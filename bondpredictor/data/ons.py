"""UK Office for National Statistics time series (www.ons.gov.uk).

A series key is the series' URI path, e.g.
``"/economy/inflationandpriceindices/timeseries/gb7s/mm22"`` (producer
output prices) or
``"/employmentandlabourmarket/peopleinwork/earningsandworkinghours/timeseries/kab9/lms"``
(average weekly earnings).  The CSV download lists annual, quarterly and
monthly observations one after the other; only the highest frequency
present is returned.
"""

from __future__ import annotations

import csv
import io
import re

import pandas as pd

from .download import DataUnavailable, Downloader

ONS_URL = "https://www.ons.gov.uk/generator?format=csv&uri={uri}"

_MONTH = re.compile(r"^(\d{4}) ([A-Z]{3})$")
_QUARTER = re.compile(r"^(\d{4}) Q([1-4])$")


def parse_ons_csv(text: str, name: str | None = None) -> pd.Series:
    monthly, quarterly = {}, {}
    for row in csv.reader(io.StringIO(text.lstrip("﻿"))):
        if len(row) < 2:
            continue
        key, val = row[0].strip(), row[1].strip()
        try:
            v = float(val)
        except ValueError:
            continue
        if m := _MONTH.match(key):
            monthly[pd.Timestamp(f"{m.group(1)}-{m.group(2).title()}-01")] = v
        elif q := _QUARTER.match(key):
            quarterly[pd.Period(f"{q.group(1)}Q{q.group(2)}", "Q").start_time] = v
    data = monthly or quarterly
    if not data:
        raise ValueError("ONS csv: no monthly or quarterly observations")
    return pd.Series(data, name=name, dtype=float).sort_index()


def fetch_ons(dl: Downloader, uri: str) -> pd.Series:
    sid = uri.rstrip("/").split("/")[-2]
    text = dl.get_text(ONS_URL.format(uri=uri), name=f"ons_{sid}.csv")
    if text.lstrip().startswith("<"):
        raise DataUnavailable(f"ONS series {uri} not found")
    return parse_ons_csv(text, name=uri)
