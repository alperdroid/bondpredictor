"""FRED / ALFRED (Federal Reserve Bank of St. Louis).

Plain series use the key-less ``fredgraph.csv`` endpoint.  Real-time vintages
(ALFRED) need the official API and a free key in ``FRED_API_KEY``.
"""

from __future__ import annotations

import io
import json
import os

import pandas as pd

from .download import DataUnavailable, Downloader

FREDGRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
API_OBS_URL = (
    "https://api.stlouisfed.org/fred/series/observations?series_id={sid}"
    "&api_key={key}&file_type=json&realtime_start=1776-07-04&realtime_end=9999-12-31"
)


def parse_fredgraph_csv(text: str, series_id: str | None = None) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        raise ValueError("unexpected FRED csv layout")
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce")  # "." marks missing
    s = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=series_id or df.columns[1])
    return s[s.index.notna()].dropna().sort_index()


def fetch_fred(dl: Downloader, series_id: str) -> pd.Series:
    text = dl.get_text(FREDGRAPH_URL.format(sid=series_id), name=f"fred_{series_id}.csv")
    s = parse_fredgraph_csv(text, series_id)
    if s.empty:
        raise DataUnavailable(f"FRED series {series_id} is empty")
    return s


def first_release(observations: list[dict]) -> pd.Series:
    """First-published value for each reference date from ALFRED observations.

    ``observations`` is the ``observations`` list of the FRED API response when
    requested with the full real-time range; each entry has ``date``,
    ``realtime_start`` and ``value``.
    """
    df = pd.DataFrame(observations)
    if df.empty:
        return pd.Series(dtype=float)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values(["date", "realtime_start"])
    first = df.groupby("date", sort=True).first()
    s = pd.Series(first["value"].to_numpy(), index=pd.to_datetime(first.index))
    s.attrs["release_date"] = pd.to_datetime(first["realtime_start"]).to_numpy()
    return s


def fetch_fred_first_release(dl: Downloader, series_id: str, api_key: str | None = None) -> pd.Series:
    """First-release (real-time) version of a revised macro series via ALFRED."""
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise DataUnavailable("set FRED_API_KEY to download ALFRED vintages")
    text = dl.get_text(API_OBS_URL.format(sid=series_id, key=key), name=f"alfred_{series_id}.json")
    payload = json.loads(text)
    s = first_release(payload.get("observations", []))
    s.name = series_id
    return s
