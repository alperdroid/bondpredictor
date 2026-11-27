"""Zillow Observed Rent Index (ZORI), national, monthly from 2015.

Asking rents on new leases, smoothed; they lead the rents measured in the CPI
(which average over all tenants) by roughly a year.  Each month's value is
published in the middle of the following month.
"""

from __future__ import annotations

import csv
import io

import pandas as pd

from .download import DataUnavailable, Downloader

ZORI_URL = "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_month.csv"


def parse_zori(text: str, region: str = "United States") -> pd.Series:
    rows = csv.reader(io.StringIO(text))
    head = next(rows)
    name = head.index("RegionName")
    dates = [(i, c) for i, c in enumerate(head) if len(c) == 10 and c[4] == "-" and c[7] == "-"]
    for row in rows:
        if row[name] == region:
            vals = {pd.Timestamp(d).to_period("M").to_timestamp(): float(row[i]) for i, d in dates if row[i] not in ("", "NA")}
            return pd.Series(vals, name="zori").sort_index()
    raise DataUnavailable(f"ZORI has no row for {region!r}")


def fetch_zillow(dl: Downloader, key: str) -> pd.Series:
    if key != "zori":
        raise ValueError(f"unknown Zillow series {key!r}")
    return parse_zori(dl.get_text(ZORI_URL, name="zillow_zori_metro.csv"))
