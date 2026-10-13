"""User-supplied files for series that are paid, discontinued or blocked.

A local CSV needs two columns: a date and a value.  Use it, for example, to
extend Japanese CPI past mid-2021 with the e-Stat release, or to add a
survey such as Consensus Economics.  Point a series at a file with the
source ``"csv"`` and the path as key (see ``config.py``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .download import DataUnavailable


def read_local_series(path: str | Path) -> pd.Series:
    p = Path(path)
    if not p.exists():
        raise DataUnavailable(f"local file not found: {p}")
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce", format="mixed")
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s = pd.Series(vals.to_numpy(), index=pd.DatetimeIndex(dates), name=p.stem)
    return s[s.index.notna()].dropna().sort_index()
