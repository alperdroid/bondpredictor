"""Japan: Ministry of Finance JGB par yields.

MOF publishes daily par (constant-maturity) yields for 1–40 years.  The full
history (1974 to last month) and the current month are separate files.
"""

from __future__ import annotations

import io
import re

import pandas as pd

from .download import DataUnavailable, Downloader

MOF_HISTORY_URL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv"
MOF_CURRENT_URL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv"

_MAT_RE = re.compile(r"^\s*(\d+)\s*Y\s*$", re.IGNORECASE)


def parse_mof_csv(text: str) -> pd.DataFrame:
    """Parse a MOF ``jgbcme*.csv`` file into percent par yields by maturity (years)."""
    lines = text.splitlines()
    header_idx = None
    for i, ln in enumerate(lines[:20]):
        cells = [c.strip().strip('"') for c in ln.split(",")]
        if sum(bool(_MAT_RE.match(c)) for c in cells) >= 3:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("MOF csv: maturity header not found")
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), dtype=str)
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    date_col = df.columns[0]
    dates = pd.to_datetime(df[date_col].str.strip(), errors="coerce", format="mixed")
    out = {}
    for col in df.columns[1:]:
        m = _MAT_RE.match(col)
        if m:
            out[float(m.group(1))] = pd.to_numeric(df[col].str.strip(), errors="coerce")  # "-" = missing
    frame = pd.DataFrame(out)
    frame.index = pd.DatetimeIndex(dates)
    frame = frame[frame.index.notna()].dropna(how="all")
    return frame.sort_index()


def fetch_mof_par_yields(dl: Downloader) -> pd.DataFrame:
    hist = parse_mof_csv(dl.get_text(MOF_HISTORY_URL, name="mof_jgbcme_all.csv"))
    try:
        cur = parse_mof_csv(dl.get_text(MOF_CURRENT_URL, name="mof_jgbcme.csv"))
        hist = pd.concat([hist, cur[~cur.index.isin(hist.index)]]).sort_index()
    except (DataUnavailable, ValueError):
        pass
    return hist
