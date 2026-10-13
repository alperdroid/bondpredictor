"""Large macro panels for Ludvigson–Ng style factors.

The US uses FRED-MD (McCracken–Ng).  Its first data row holds the
transformation codes.  When FRED-MD cannot be reached, a curated panel of
individual FRED series is used instead (see ``config.py``).

Transformation codes follow FRED-MD:
1 level, 2 Δx, 3 Δ²x, 4 log x, 5 Δlog x, 6 Δ²log x, 7 Δ(x_t/x_{t-1} - 1).
Two extra codes handle seasonally unadjusted series:
8 12-month log change, 9 12-month change.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .download import DataUnavailable, Downloader

FREDMD_URLS = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv",
    "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv",
)


def transform(x: pd.Series, tcode: int) -> pd.Series:
    x = x.astype(float)
    if tcode in (4, 5, 6, 8):
        x = np.log(x.where(x > 0))
    if tcode in (1, 4):
        return x
    if tcode in (2, 5):
        return x.diff()
    if tcode in (3, 6):
        return x.diff().diff()
    if tcode == 7:
        return x.pct_change(fill_method=None).diff()
    if tcode in (8, 9):
        return x.diff(12)
    raise ValueError(f"unknown tcode {tcode}")


def parse_fredmd(text: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return (levels with monthly PeriodIndex, {column: tcode})."""
    df = pd.read_csv(io.StringIO(text))
    first = df.columns[0]
    tc_row = df.iloc[0]
    if not str(tc_row[first]).lower().startswith("transform"):
        raise ValueError("FRED-MD: transform row not found")
    tcodes = {c: int(float(tc_row[c])) for c in df.columns[1:] if pd.notna(tc_row[c])}
    body = df.iloc[1:].copy()
    dates = pd.to_datetime(body.pop(first), errors="coerce", format="mixed")
    body.index = pd.DatetimeIndex(dates).to_period("M")
    body = body[body.index.notna()].apply(pd.to_numeric, errors="coerce")
    return body.dropna(how="all"), tcodes


def fetch_fredmd(dl: Downloader) -> tuple[pd.DataFrame, dict[str, int]]:
    last: Exception | None = None
    for url in FREDMD_URLS:
        try:
            return parse_fredmd(dl.get_text(url, name="fredmd_current.csv"))
        except (DataUnavailable, ValueError) as err:
            last = err
    raise DataUnavailable(f"FRED-MD unavailable: {last}")


def transform_panel(levels: pd.DataFrame, tcodes: dict[str, int]) -> pd.DataFrame:
    out = {c: transform(levels[c], tcodes.get(c, 1)) for c in levels.columns}
    return pd.DataFrame(out, index=levels.index)
