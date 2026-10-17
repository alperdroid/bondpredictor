"""Raw data access.  ``fetch_series`` dispatches a :class:`~bondpredictor.config.Series`
spec to the right provider."""

from __future__ import annotations

import pandas as pd

from .boj import fetch_boj
from .bundesbank import fetch_bundesbank
from .download import DataUnavailable, Downloader
from .eurostat import fetch_eurostat
from .fred import fetch_fred
from .local import read_local_series
from .sdmx import fetch_ecb, fetch_oecd

__all__ = ["DataUnavailable", "Downloader", "fetch_raw"]


def fetch_raw(dl: Downloader, source: str, key: str) -> pd.Series:
    if source == "fred":
        return fetch_fred(dl, key)
    if source == "bundesbank":
        return fetch_bundesbank(dl, key)
    if source == "oecd":
        return fetch_oecd(dl, key)
    if source == "ecb":
        return fetch_ecb(dl, key)
    if source == "eurostat":
        return fetch_eurostat(dl, key)
    if source == "boj":
        return fetch_boj(dl, key)
    if source == "csv":
        return read_local_series(key)
    raise ValueError(f"unknown source {source!r}")
