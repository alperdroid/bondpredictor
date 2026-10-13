"""Deutsche Bundesbank time-series database (api.statistiken.bundesbank.de).

The Bundesbank fits a Svensson curve to listed Federal securities each day
and publishes both the parameters and the implied zero-coupon yields.
Monthly values are end-of-month (they match the last business day of the
daily series), and the monthly history starts in September 1972.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from .download import DataUnavailable, Downloader

BBK_URL = "https://api.statistiken.bundesbank.de/rest/download/{flow}/{key}?format=csv&lang=en"

# Svensson parameters of the German federal-securities curve.
_PARAM_KEY = "BBSIS.{freq}.I.ZST.{param}.EUR.S1311.B.A604._Z.R.A.A._Z._Z.A"
# Zero-coupon yield for a residual maturity of `mat` years (01..30).
_ZERO_KEY = "BBSIS.{freq}.I.ZST.ZI.EUR.S1311.B.A604.R{mat:02d}XX.R.A.A._Z._Z.A"

SVENSSON_PARAMS = {"beta0": "B0", "beta1": "B1", "beta2": "B2", "beta3": "B3", "tau1": "T1", "tau2": "T2"}

_DATE_RE = re.compile(r"^\d{4}(-\d{2}){0,2}$|^\d{4}-Q\d$")


def parse_bundesbank_csv(text: str, name: str | None = None) -> pd.Series:
    """Parse the Bundesbank CSV download (metadata rows, then ``date,value,flag``)."""
    rows = []
    for line in io.StringIO(text.lstrip("﻿")):
        parts = line.rstrip("\r\n").split(",")
        if len(parts) >= 2 and _DATE_RE.match(parts[0].strip().strip('"')):
            rows.append((parts[0].strip().strip('"'), parts[1].strip().strip('"')))
    if not rows:
        raise ValueError("no observations found in Bundesbank csv")
    dates, values = zip(*rows)
    if "Q" in dates[0]:
        idx = pd.PeriodIndex(dates, freq="Q").to_timestamp(how="start")
    else:
        idx = pd.to_datetime(list(dates), format="mixed")
    vals = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()  # "." = missing
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), name=name)
    return s.dropna().sort_index()


def fetch_bundesbank(dl: Downloader, series_key: str) -> pd.Series:
    flow, key = series_key.split(".", 1)
    url = BBK_URL.format(flow=flow, key=key)
    text = dl.get_text(url, name=f"bbk_{series_key}.csv")
    s = parse_bundesbank_csv(text, name=series_key)
    if s.empty:
        raise DataUnavailable(f"Bundesbank series {series_key} is empty")
    return s


def svensson_parameters(dl: Downloader, freq: str = "M") -> pd.DataFrame:
    """Svensson parameters (betas in percent, taus in years) of the Bund curve."""
    cols = {}
    for name, code in SVENSSON_PARAMS.items():
        cols[name] = fetch_bundesbank(dl, _PARAM_KEY.format(freq=freq, param=code))
    return pd.DataFrame(cols).sort_index()


def zero_yield(dl: Downloader, maturity: int, freq: str = "M") -> pd.Series:
    """Published zero-coupon yield (percent) for an integer maturity in years."""
    return fetch_bundesbank(dl, _ZERO_KEY.format(freq=freq, mat=maturity))


def max_maturity_from_published(dl: Downloader, freq: str = "M") -> pd.Series:
    """Longest maturity (years) for which the Bundesbank publishes a yield.

    The Svensson parameters can be evaluated at any maturity, but before 1986
    there were no bonds beyond ~10 years, so longer yields are extrapolations.
    """
    published = {}
    for mat in (10, 15, 20, 25, 30):
        try:
            published[mat] = zero_yield(dl, mat, freq)
        except DataUnavailable:
            continue
    if not published:
        raise DataUnavailable("no published Bundesbank zero yields")
    avail = pd.DataFrame(published).notna()
    mats = np.array(list(avail.columns), dtype=float)
    return pd.Series((avail.to_numpy() * mats).max(axis=1), index=avail.index).sort_index()
