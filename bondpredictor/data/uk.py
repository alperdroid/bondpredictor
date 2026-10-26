"""United Kingdom: Bank of England nominal government zero-coupon (spot) curves.

The BoE publishes month-end curves fitted to gilts (variable roughness
penalty spline) from 1970.  They come as a zip of Excel workbooks.  Each
workbook has a long-end sheet (maturities 0.5–25 or 40 years in half-year
steps) and a short-end sheet (1–60 months).  Rates are in percent and are
treated as continuously compounded.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from .download import DataUnavailable, Downloader

BOE_MONTH_END_URL = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/glcnominalmonthedata.zip"
_SHEETS = ("4. spot curve", "3. spot, short end")


def parse_boe_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """One curve sheet (read with ``header=None``) -> dates x maturities (years), percent."""
    first = df.iloc[:, 0].astype(str).str.strip().str.lower()
    hits = first.index[first.str.startswith("years")]
    if len(hits) == 0:
        raise ValueError("BoE sheet: no 'years:' row")
    mats = pd.to_numeric(df.loc[hits[0]].iloc[1:], errors="coerce")
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce", format="mixed")
    body = df.loc[dates.notna()].iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    body.index = pd.DatetimeIndex(dates[dates.notna()])
    keep = mats.notna().to_numpy()
    body = body.loc[:, keep]
    body.columns = [round(float(m), 6) for m in mats[keep]]
    return body.dropna(how="all")


def parse_boe_zip(content: bytes) -> pd.DataFrame:
    """All month-end workbooks in the archive, short and long end merged."""
    z = zipfile.ZipFile(io.BytesIO(content))
    names = sorted(n for n in z.namelist() if "month end" in n.lower() and n.lower().endswith(".xlsx"))
    if not names:
        raise ValueError("BoE zip: no month-end workbooks")
    frames = []
    for name in names:
        book = pd.ExcelFile(io.BytesIO(z.read(name)))
        parts = [parse_boe_sheet(pd.read_excel(book, sheet_name=s, header=None)) for s in _SHEETS if s in book.sheet_names]
        if not parts:
            continue
        merged = parts[0]
        for p in parts[1:]:
            merged = merged.combine_first(p)  # overlapping maturities agree; long sheet wins
        frames.append(merged)
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.reindex(columns=sorted(out.columns))


def fetch_boe_curve(dl: Downloader) -> pd.DataFrame:
    try:
        return parse_boe_zip(dl.get(BOE_MONTH_END_URL, name="boe_glcnominalmonthedata.zip"))
    except (zipfile.BadZipFile, ValueError) as err:
        raise DataUnavailable(f"Bank of England curve: {err}") from err
