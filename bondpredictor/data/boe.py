"""Bank of England interactive database (IADB).

A series key is an IADB code, e.g. ``"IUMAMIZC"`` (10-year implied
inflation, monthly average) or ``"IUMBEDR"`` (Bank Rate, month end).  The
CSV endpoint redirects to an error page (HTTP 200) for unknown codes and for
start dates before 1963.
"""

from __future__ import annotations

import io

import pandas as pd

from .download import DataUnavailable, Downloader

BOE_URL = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes"
           "&Datefrom=01/Jan/1963&Dateto=now&SeriesCodes={code}&CSVF=TN&UsingCodes=Y")


def parse_boe_csv(text: str, name: str | None = None) -> pd.Series:
    if not text.lstrip().upper().startswith("DATE"):
        raise ValueError("BoE csv: unexpected response (unknown code, or start date before 1963)")
    df = pd.read_csv(io.StringIO(text))
    dates = pd.to_datetime(df.iloc[:, 0], format="%d %b %Y", errors="coerce")
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s = pd.Series(vals.to_numpy(), index=pd.DatetimeIndex(dates), name=name or df.columns[1])
    return s[s.index.notna()].dropna().sort_index()


def fetch_boe(dl: Downloader, code: str) -> pd.Series:
    url, name = BOE_URL.format(code=code), f"boe_{code}.csv"
    text = dl.get_text(url, name=name)
    try:
        s = parse_boe_csv(text, name=code)
    except ValueError as err:
        dl.cache_path(url, name).unlink(missing_ok=True)  # do not keep the error page as data
        raise DataUnavailable(f"BoE series {code}: {err}") from err
    if s.empty:
        raise DataUnavailable(f"BoE series {code} is empty")
    return s
