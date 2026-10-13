"""United States curve sources.

* Gürkaynak–Sack–Wright (GSW) Svensson curve, Federal Reserve Board, daily
  since 1961.  This is the default.
* Kim–Wright fitted zero yields (1–10y, daily since 1990) and term premia,
  which FRED republishes.  Used as a fallback when federalreserve.gov is
  unreachable.
* Adrian–Crump–Moench (ACM) term premia, New York Fed.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .download import DataUnavailable, Downloader
from .fred import fetch_fred

GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
ACM_URL = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"


def parse_gsw_csv(text: str) -> pd.DataFrame:
    """Parse ``feds200628.csv`` into a daily frame of all published columns."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip().strip('"').startswith("Date")), None)
    if start is None:
        raise ValueError("GSW csv: header row not found")
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), na_values=["NA", "-999.99", "-999.990000"])
    df.columns = [c.strip().strip('"') for c in df.columns]
    df.index = pd.to_datetime(df.pop("Date"), errors="coerce")
    df = df[df.index.notna()].apply(pd.to_numeric, errors="coerce")
    return df.sort_index()


def gsw_parameters(raw: pd.DataFrame) -> pd.DataFrame:
    """Svensson parameters with the package's column names, plus max maturity.

    GSW only report yields out to the longest outstanding maturity; that
    limit is recovered from which ``SVENYnn`` columns are populated.
    """
    params = pd.DataFrame(
        {
            "beta0": raw["BETA0"],
            "beta1": raw["BETA1"],
            "beta2": raw["BETA2"],
            "beta3": raw.get("BETA3"),
            "tau1": raw["TAU1"],
            "tau2": raw.get("TAU2"),
        }
    )
    ycols = sorted((c for c in raw.columns if c.startswith("SVENY")), key=lambda c: int(c[5:]))
    if ycols:
        mats = np.array([int(c[5:]) for c in ycols], dtype=float)
        avail = raw[ycols].notna().to_numpy()
        params["max_maturity"] = np.where(avail.any(axis=1), (avail * mats).max(axis=1), np.nan)
    return params.dropna(subset=["beta0", "beta1", "beta2", "tau1"])


def fetch_gsw(dl: Downloader) -> pd.DataFrame:
    return gsw_parameters(parse_gsw_csv(dl.get_text(GSW_URL, name="feds200628.csv")))


def fetch_kim_wright(dl: Downloader, maturities=range(1, 11)) -> pd.DataFrame:
    """Kim–Wright fitted zero yields (percent), daily, columns = maturity in years."""
    cols = {}
    for n in maturities:
        cols[float(n)] = fetch_fred(dl, f"THREEFY{n}")
    return pd.DataFrame(cols).sort_index()


def fetch_kim_wright_term_premium(dl: Downloader, maturity: int = 10) -> pd.Series:
    return fetch_fred(dl, f"THREEFYTP{maturity}")


def parse_acm(content: bytes) -> pd.DataFrame:
    """Parse the NY Fed ACM workbook; returns yields, term premia and
    risk-neutral yields (percent) on the sheet with the highest frequency."""
    try:
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
    except ImportError as err:  # pragma: no cover - depends on optional xlrd
        raise DataUnavailable("reading the ACM .xls file needs `pip install xlrd`") from err
    best = None
    for frame in sheets.values():
        cols = [str(c).strip().upper() for c in frame.columns]
        if "ACMTP10" in cols:
            frame = frame.copy()
            frame.columns = cols
            if best is None or len(frame) > len(best):
                best = frame
    if best is None:
        raise ValueError("ACM workbook: no sheet with ACMTP10")
    date_col = next(c for c in best.columns if "DATE" in c)
    best.index = pd.to_datetime(best.pop(date_col), errors="coerce")
    best = best[best.index.notna()].apply(pd.to_numeric, errors="coerce")
    return best.sort_index()


def fetch_acm(dl: Downloader) -> pd.DataFrame:
    return parse_acm(dl.get(ACM_URL, name="ACMTermPremium.xls"))
