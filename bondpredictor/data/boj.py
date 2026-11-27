"""Bank of Japan time-series API (www.stat-search.boj.or.jp/api/v1).

A series key is ``"<db>/<series code>"``, e.g. ``"PR01/PRCG20_2200000000"``
(producer price index) or ``"CO/TK99F0000614GCQ00000"`` (Tankan output-price
DI).  Monthly dates come as ``YYYYMM``; quarterly ones as ``YYYYQQ`` with
``QQ`` = 01..04.
"""

from __future__ import annotations

import io

import pandas as pd

from .download import DataUnavailable, Downloader

BOJ_URL = "https://www.stat-search.boj.or.jp/api/v1/getDataCode?format=csv&lang=en&db={db}&startDate=195001&code={code}"


def parse_boj_csv(text: str, name: str | None = None) -> pd.Series:
    lines = text.lstrip("﻿").splitlines()
    status = next((ln.split(",", 1)[1] for ln in lines[:3] if ln.startswith("STATUS")), "200")
    if status.strip() != "200":
        raise ValueError(f"BoJ API status {status}: {lines[2] if len(lines) > 2 else ''}")
    head = next((i for i, ln in enumerate(lines) if ln.startswith("SERIES_CODE")), None)
    if head is None:
        raise ValueError("BoJ csv: header row not found")
    df = pd.read_csv(io.StringIO("\n".join(lines[head:])), dtype={"SURVEY_DATES": str})
    if df.empty:
        return pd.Series(dtype=float, name=name)
    freq = str(df["FREQUENCY"].iloc[0]).upper()
    d = df["SURVEY_DATES"].str.strip()
    year, sub = d.str[:4].astype(int), d.str[4:6].astype(int)
    if freq.startswith("QUARTER"):
        idx = pd.PeriodIndex.from_fields(year=year, quarter=sub, freq="Q").to_timestamp(how="start")
    elif freq.startswith("MONTH"):
        idx = pd.PeriodIndex.from_fields(year=year, month=sub, freq="M").to_timestamp(how="start")
    else:
        raise ValueError(f"unsupported BoJ frequency {freq}")
    vals = pd.to_numeric(df["VALUES"], errors="coerce").to_numpy()
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), name=name)
    return s.dropna().sort_index()


def fetch_boj(dl: Downloader, series_key: str) -> pd.Series:
    db, code = series_key.split("/", 1)
    text = dl.get_text(BOJ_URL.format(db=db, code=code), name=f"boj_{db}_{code}.csv")
    s = parse_boj_csv(text, name=series_key)
    if s.empty:
        raise DataUnavailable(f"BoJ series {series_key} is empty")
    return s
