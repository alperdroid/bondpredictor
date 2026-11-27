"""Japanese consumer price tables from e-Stat's public file listing (www.e-stat.go.jp).

A series key is ``"<area>/<item code>"``, e.g. ``"tokyo/0001"``: all items in
the Ku-area of Tokyo, monthly from January 1970.  The Tokyo index for a month
is published at the end of that same month, about four weeks before the
national one, so it is the standard nowcast of the national print.  Useful
item codes: 0001 all items, 0161 less fresh food, 0168 less food (less
alcoholic beverages) and energy, 0167 energy, 0002 food.

The files are plain downloads and need no application ID.  Their identifiers
change with every release, so the loader reads the listing of the long time
series for the current one.  The latest month is a preliminary figure that is
revised a month later; the file holds the revised values for earlier months.
"""

from __future__ import annotations

import csv
import html
import io
import re

import pandas as pd

from .download import DataUnavailable, Downloader

# 2025-base CPI, long time series, price indices by item, monthly; ``area`` picks the region.
LISTING = ("https://www.e-stat.go.jp/stat-search/files?tstat=000001243876&toukei=00200573&cycle=0"
           "&tclass1=000001243880&tclass2=000001243881&tclass3={area}&tclass4=000001243888"
           "&layout=datalist&page=1&tclass5val=0")
AREAS = {"tokyo": "000001243884"}
TABLE = "中分類指数（1970年1月～最新月）"  # indices of subgroups, January 1970 to the latest month
DOWNLOAD = "https://www.e-stat.go.jp/stat-search/file-download?statInfId={id}&fileKind=1"


def table_id(page: str, title: str = TABLE) -> str:
    """Identifier of the CSV file listed under ``title`` on a listing page."""
    for m in re.finditer(r"statInfId=(\d+)&(?:amp;)?fileKind=1", page):
        before = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", page[max(0, m.start() - 3000):m.start()])).split())
        if title in before[-300:]:
            return m.group(1)
    raise DataUnavailable(f"e-Stat listing has no CSV file titled {title!r}")


def parse_cpi_csv(text: str, code: str) -> pd.Series:
    """One item's monthly index from a long-time-series file (one column per item, one row per month)."""
    rows = list(csv.reader(io.StringIO(text)))
    codes = next((r for r in rows if r and "Group/Item code" in r[0]), None)
    if codes is None or code not in codes:
        raise DataUnavailable(f"e-Stat CPI file has no item {code!r}")
    col = codes.index(code)
    data = {}
    for r in rows:
        if r and re.fullmatch(r"\d{6}", r[0].strip()) and len(r) > col and r[col].strip():
            try:
                data[pd.Timestamp(int(r[0][:4]), int(r[0][4:6]), 1)] = float(r[col])
            except ValueError:
                continue
    if not data:
        raise DataUnavailable(f"e-Stat CPI file has no observations for item {code!r}")
    return pd.Series(data, dtype=float).sort_index()


def fetch_jstat(dl: Downloader, key: str) -> pd.Series:
    area, _, code = key.partition("/")
    if area not in AREAS or not code:
        raise ValueError(f"unknown e-Stat CPI series {key!r}")
    page = dl.get_text(LISTING.format(area=AREAS[area]), name=f"jstat_{area}_listing.html")
    text = dl.get_text(DOWNLOAD.format(id=table_id(page)), name=f"jstat_{area}_subgroups.csv", encoding="cp932")
    return parse_cpi_csv(text, code).rename(f"jstat_{area}_{code}")
