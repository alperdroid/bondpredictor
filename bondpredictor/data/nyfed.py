"""New York Fed Global Supply Chain Pressure Index (Benigno, di Giovanni, Groen and Noble 2022).

Monthly from 1998, in standard deviations from its average; published in the
first week of the following month.  The download is a legacy Excel workbook
(needs ``xlrd``).
"""

from __future__ import annotations

import io

import pandas as pd

from .download import DataUnavailable, Downloader

GSCPI_URL = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"


def parse_gscpi(raw: bytes) -> pd.Series:
    df = pd.read_excel(io.BytesIO(raw), sheet_name="GSCPI Monthly Data")
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce", format="%d-%b-%Y")
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s = pd.Series(vals.to_numpy(), index=pd.DatetimeIndex(dates), name="gscpi").dropna()
    s = s[s.index.notna()]
    # Month-end dates -> month start, the convention used elsewhere.
    s.index = s.index.to_period("M").to_timestamp()
    return s.sort_index()


def fetch_nyfed(dl: Downloader, key: str) -> pd.Series:
    if key != "gscpi":
        raise ValueError(f"unknown New York Fed series {key!r}")
    s = parse_gscpi(dl.get(GSCPI_URL, name="nyfed_gscpi.xlsx"))
    if s.empty:
        raise DataUnavailable("GSCPI download is empty")
    return s
