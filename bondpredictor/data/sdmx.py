"""SDMX REST sources: the OECD data explorer and the ECB data portal.

Both return SDMX-CSV with ``TIME_PERIOD`` and ``OBS_VALUE`` columns.  A
series key is ``"<dataflow>/<key>"``:

* OECD: ``"OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/GBR.M.N.CPI.IX._T.N._Z"``
* ECB:  ``"ICP/M.U2.N.000000.4.INX"``

The OECD API rate-limits anonymous use, so uncached OECD requests are
spaced out by :data:`OECD_PAUSE` seconds.
"""

from __future__ import annotations

import hashlib
import io
import time

import pandas as pd

from .download import DataUnavailable, Downloader

OECD_URL = "https://sdmx.oecd.org/public/rest/data/{flow}/{key}?startPeriod={start}&format=csvfile"
ECB_URL = "https://data-api.ecb.europa.eu/service/data/{flow}/{key}?format=csvdata"
OECD_PAUSE = 1.5


def parse_period(values) -> pd.DatetimeIndex:
    """SDMX time periods (``2026``, ``2026-Q1``, ``2026-08``, ``2026-08-14``) to period-start dates."""
    s = pd.Series(values, dtype=str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    q = s.str.match(r"^\d{4}-Q[1-4]$")
    if q.any():
        out[q] = pd.PeriodIndex(s[q].str.replace("-", ""), freq="Q").to_timestamp(how="start")
    a = s.str.match(r"^\d{4}$")
    if a.any():
        out[a] = pd.to_datetime(s[a], format="%Y")
    rest = ~(q | a)
    if rest.any():
        out[rest] = pd.to_datetime(s[rest], errors="coerce", format="mixed")
    return pd.DatetimeIndex(out)


def _series_ids(df: pd.DataFrame) -> pd.Series:
    """One label per series: ECB files carry a ``KEY`` column; OECD files list the dimensions."""
    if "KEY" in df:
        return df["KEY"].astype(str)
    cols = list(df.columns)
    first = cols.index("ACTION") + 1 if "ACTION" in cols else 0
    dims = cols[first : cols.index("TIME_PERIOD")]
    return df[dims].astype(str).agg(".".join, axis=1)


def parse_sdmx_csv(text: str, name: str | None = None) -> pd.Series:
    """A single SDMX-CSV series as a date-indexed float series."""
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    if "TIME_PERIOD" not in df or "OBS_VALUE" not in df:
        raise ValueError(f"not an SDMX-CSV file: {text[:80]!r}")
    ids = _series_ids(df)
    if ids.nunique() > 1:
        raise ValueError(f"key matches {ids.nunique()} series, expected one: {sorted(ids.unique())[:4]}")
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").to_numpy(), index=parse_period(df["TIME_PERIOD"]), name=name)
    s = s[s.index.notna()].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def parse_sdmx_panel(text: str, by: str = "REF_AREA") -> pd.DataFrame:
    """SDMX-CSV with several series that differ only in dimension ``by``; one column per value."""
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    if df.empty or "TIME_PERIOD" not in df:
        raise ValueError("empty SDMX-CSV panel")
    df = df.assign(_date=parse_period(df["TIME_PERIOD"]), _v=pd.to_numeric(df["OBS_VALUE"], errors="coerce"))
    wide = df.pivot_table(index="_date", columns=by, values="_v", aggfunc="last")
    wide.index.name = None
    wide.columns.name = None
    return wide.sort_index()


def _short(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def oecd_text(dl: Downloader, series_key: str, start: str = "1950-01") -> str:
    flow, key = series_key.split("/", 1)
    url = OECD_URL.format(flow=flow, key=key, start=start)
    name = f"oecd_{key[:40].replace('+', '_')}_{_short(series_key)}.csv"
    cached = dl.cache_path(url, name).exists()
    text = dl.get_text(url, name=name)
    if not cached and not dl.offline:
        time.sleep(OECD_PAUSE)
    if text.startswith("NoRecordsFound") or not text.strip():
        raise DataUnavailable(f"OECD: no records for {series_key}")
    return text


def fetch_oecd(dl: Downloader, series_key: str) -> pd.Series:
    s = parse_sdmx_csv(oecd_text(dl, series_key), name=series_key)
    if s.empty:
        raise DataUnavailable(f"OECD series {series_key} is empty")
    return s


def fetch_oecd_panel(dl: Downloader, series_key: str, by: str = "REF_AREA") -> pd.DataFrame:
    """Several countries at once, e.g. ``".../.M.N.CPI.IX._T.N._Z"`` with an empty REF_AREA."""
    return parse_sdmx_panel(oecd_text(dl, series_key), by=by)


def fetch_ecb(dl: Downloader, series_key: str) -> pd.Series:
    flow, key = series_key.split("/", 1)
    text = dl.get_text(ECB_URL.format(flow=flow, key=key), name=f"ecb_{flow}_{key}.csv")
    s = parse_sdmx_csv(text, name=series_key)
    if s.empty:
        raise DataUnavailable(f"ECB series {series_key} is empty")
    return s
