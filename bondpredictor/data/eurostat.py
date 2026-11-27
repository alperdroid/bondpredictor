"""Eurostat dissemination API (JSON-stat 2.0).

A series key is a dataset code plus filters, e.g.
``"prc_hicp_minr?geo=EA&unit=I15&coicop18=TOTAL"``.  The filters must pin
every dimension except time to a single value.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .download import DataUnavailable, Downloader
from .sdmx import parse_period

EUROSTAT_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{query}"


def parse_jsonstat(payload: dict, name: str | None = None) -> pd.Series:
    if "error" in payload:
        raise ValueError(f"Eurostat error: {payload['error']}")
    dims, sizes = payload["id"], payload["size"]
    if "time" not in dims:
        raise ValueError("JSON-stat payload has no time dimension")
    if 0 in sizes:
        return pd.Series(dtype=float, name=name)
    extra = [d for d, n in zip(dims, sizes) if d != "time" and n != 1]
    if extra:
        raise ValueError(f"filters leave several values for {extra}")
    tpos = dims.index("time")
    stride = int(np.prod(sizes[tpos + 1 :])) if tpos + 1 < len(sizes) else 1
    times = payload["dimension"]["time"]["category"]["index"]
    by_pos = {pos: t for t, pos in times.items()}
    values = payload.get("value", {})
    if isinstance(values, list):  # dense form
        values = {str(i): v for i, v in enumerate(values) if v is not None}
    rows = {by_pos[int(k) // stride % sizes[tpos]]: v for k, v in values.items() if v is not None}
    if not rows:
        return pd.Series(dtype=float, name=name)
    s = pd.Series(list(rows.values()), index=parse_period(list(rows.keys())), name=name, dtype=float)
    return s.sort_index()


def fetch_eurostat(dl: Downloader, query: str) -> pd.Series:
    digest = hashlib.sha1(query.encode()).hexdigest()[:12]
    text = dl.get_text(EUROSTAT_URL.format(query=query), name=f"eurostat_{query.split('?')[0]}_{digest}.json")
    s = parse_jsonstat(json.loads(text), name=query)
    if s.empty:
        raise DataUnavailable(f"Eurostat query {query} returned no data")
    return s
