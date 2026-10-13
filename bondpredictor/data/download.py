"""HTTP downloads with an on-disk cache.

Every loader goes through :class:`Downloader`, so a pipeline run can be
repeated offline once the raw files are cached.  When a refresh fails, a stale
cached copy is used with a warning rather than aborting the run.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(os.environ.get("BONDPREDICTOR_CACHE", "data/cache"))
USER_AGENT = "bondpredictor/0.1 (research; https://github.com/alperdroid/bondpredictor)"


class DataUnavailable(RuntimeError):
    """Raised when a source cannot be downloaded and no cached copy exists."""


class Downloader:
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_age_days: float = 1.0,
        offline: bool = False,
        timeout: float = 60.0,
        retries: int = 3,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.max_age = max_age_days * 86400.0
        self.offline = offline
        self.timeout = timeout
        self.retries = retries
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT

    def cache_path(self, url: str, name: str | None = None) -> Path:
        if name is None:
            digest = hashlib.sha1(url.encode()).hexdigest()[:16]
            tail = re.sub(r"[^A-Za-z0-9._-]+", "_", url.rsplit("/", 1)[-1])[:60]
            name = f"{tail}_{digest}"
        return self.cache_dir / name

    def get(self, url: str, name: str | None = None) -> bytes:
        path = self.cache_path(url, name)
        if path.exists():
            age = time.time() - path.stat().st_mtime
            if self.offline or age < self.max_age:
                return path.read_bytes()
        if self.offline:
            raise DataUnavailable(f"offline and not cached: {url}")

        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self._session.get(url, timeout=self.timeout)
                if resp.status_code == 200 and resp.content:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(resp.content)
                    return resp.content
                last_err = DataUnavailable(f"HTTP {resp.status_code} for {url}")
                if resp.status_code in (400, 401, 403, 404, 410):
                    break  # not transient
            except requests.RequestException as err:
                last_err = err
            time.sleep(min(2 ** attempt, 8))

        if path.exists():
            log.warning("refresh failed (%s); using cached copy of %s", last_err, url)
            return path.read_bytes()
        raise DataUnavailable(f"could not download {url}: {last_err}")

    def get_text(self, url: str, name: str | None = None, encoding: str = "utf-8") -> str:
        raw = self.get(url, name)
        for enc in (encoding, "utf-8-sig", "cp932", "latin-1"):
            try:
                return raw.decode(enc).lstrip("﻿")
            except UnicodeDecodeError:
                continue
        return raw.decode("latin-1")
