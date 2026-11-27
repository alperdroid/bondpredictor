"""Weekly retail fuel prices, published within days of each survey.

* ``ukfuel``: UK pump prices of petrol and diesel, pence per litre, Mondays
  from June 2003 (Department for Energy Security and Net Zero).
* ``ecoil``: euro-area weighted average consumer prices including taxes,
  euro per 1000 litres, Mondays from 2005 (European Commission, Weekly Oil
  Bulletin).
* ``meti``: Japanese national average retail prices, yen per litre (kerosene:
  per 18 litres), Mondays from August 1990 (Agency for Natural Resources and
  Energy).  The agency's server refuses automated clients (HTTP 403), so this
  loader only works from copies of the results page and the weekly workbook
  saved by hand in the cache as ``meti_page.html`` and ``meti_weekly.xlsx``.

The data files change name with every release, so each loader reads the
publisher's landing page for the current link.
"""

from __future__ import annotations

import csv
import html
import io
import re
from urllib.parse import urljoin

import pandas as pd

from .download import DataUnavailable, Downloader

UK_PAGE = "https://www.gov.uk/government/statistics/weekly-road-fuel-prices"
EC_PAGE = "https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en"
METI_PAGE = "https://www.enecho.meti.go.jp/statistics/petroleum_and_lpgas/pl007/results.html"

UK_COLUMNS = {"petrol": "ULSP", "diesel": "ULSD"}
EC_COLUMNS = {"euro95": "EUR_price_with_tax_euro95", "diesel": "EUR_price_with_tax_diesel",
              "heating_oil": "EUR_price_with_tax_heating_oil"}
METI_SHEETS = {"regular": "レギュラー", "diesel": "軽油", "kerosene": "灯油店頭"}


def page_links(page: str, base: str, pattern: str) -> list[str]:
    """Absolute URLs of the links on ``page`` whose address matches ``pattern``, in page order, without repeats."""
    out = []
    for href in re.findall(r'href="([^"]+)"', page):
        url = urljoin(base, html.unescape(href))
        if re.search(pattern, url) and url not in out:
            out.append(url)
    return out


def parse_uk_fuel(text: str, key: str) -> pd.Series:
    rows = csv.reader(io.StringIO(text.lstrip("﻿")))
    head = next(rows)
    col = next(i for i, c in enumerate(head) if c.startswith(UK_COLUMNS[key]) and "pump price" in c.lower())
    data = {}
    for row in rows:
        try:
            data[pd.Timestamp(pd.to_datetime(row[0], format="%d/%m/%Y"))] = float(row[col])
        except (ValueError, IndexError):
            continue
    return pd.Series(data, dtype=float).sort_index()


def fetch_uk_fuel(dl: Downloader, key: str) -> pd.Series:
    page = dl.get_text(UK_PAGE, name="ukfuel_page.html")
    links = page_links(page, UK_PAGE, r"assets\.publishing\.service\.gov\.uk/.+\.csv$")
    if not links:
        raise DataUnavailable("UK weekly road fuel prices: no CSV link on the statistics page")
    parts = [parse_uk_fuel(dl.get_text(url, name=f"ukfuel_{i}.csv"), key) for i, url in enumerate(links)]
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="last")].rename(f"ukfuel_{key}")


def _sheet_series(raw: bytes, sheet: str, date_col: int, value_col, first_row: int) -> pd.Series:
    """Dated numbers from one worksheet; ``value_col`` is a column number or a first-row header to look up."""
    import openpyxl

    ws = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)[sheet]
    rows = ws.iter_rows(values_only=True)
    head = next(rows)
    if isinstance(value_col, str):
        value_col = next(i for i, c in enumerate(head) if str(c or "").strip() == value_col)
    data = {}
    for n, row in enumerate(rows, start=2):
        if n < first_row or len(row) <= value_col:
            continue
        d, v = row[date_col], row[value_col]
        if hasattr(d, "year") and isinstance(v, (int, float)) and v > 0:
            data[pd.Timestamp(d)] = float(v)
    return pd.Series(data, dtype=float).sort_index()


def parse_ec_oil(raw: bytes, key: str) -> pd.Series:
    return _sheet_series(raw, "Prices with taxes", 0, EC_COLUMNS[key], 4)


def fetch_ec_oil(dl: Downloader, key: str) -> pd.Series:
    page = dl.get_text(EC_PAGE, name="ecoil_page.html")
    links = page_links(page, EC_PAGE, r"Prices_History")
    if not links:
        raise DataUnavailable("Weekly Oil Bulletin: no price history link on the bulletin page")
    return parse_ec_oil(dl.get(links[0], name="ecoil_history.xlsx"), key).rename(f"ecoil_{key}")


def parse_meti(raw: bytes, key: str) -> pd.Series:
    return _sheet_series(raw, METI_SHEETS[key], 1, 2, 2)


def fetch_meti(dl: Downloader, key: str) -> pd.Series:
    page = dl.get_text(METI_PAGE, name="meti_page.html")
    links = page_links(page, METI_PAGE, r"/\d{6}s5\.xlsx$")
    if not links:
        raise DataUnavailable("METI retail price survey: no weekly history link on the results page")
    return parse_meti(dl.get(links[0], name="meti_weekly.xlsx"), key).rename(f"meti_{key}")


def fetch_fuel(dl: Downloader, source: str, key: str) -> pd.Series:
    fetch, keys = {"ukfuel": (fetch_uk_fuel, UK_COLUMNS), "ecoil": (fetch_ec_oil, EC_COLUMNS),
                   "meti": (fetch_meti, METI_SHEETS)}[source]
    if key not in keys:
        raise ValueError(f"unknown {source} series {key!r}")
    return fetch(dl, key)
