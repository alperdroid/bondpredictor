"""Download and assemble the raw monthly inputs for the inflation models.

Everything sits on a "known-at" monthly grid: row ``k`` holds what was
published by the end of month ``k`` (each source shifted by its
publication lag, see :mod:`bondpredictor.inflation.config`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Series
from ..data import DataUnavailable, Downloader
from ..data.sdmx import fetch_oecd_panel
from ..dataset import load_series
from . import config as cfg

log = logging.getLogger(__name__)


def splice(parts: list[pd.Series], how: str = "ratio") -> pd.Series:
    """Combine series in priority order.

    The first series is kept wherever it has data; each later one fills the
    remaining months, rescaled to match the combined series at their first
    common month (``how="ratio"``) or taken as is (``how="level"``).
    """
    parts = [p.dropna() for p in parts if p is not None and p.notna().any()]
    if not parts:
        return pd.Series(dtype=float)
    out = parts[0]
    for p in parts[1:]:
        common = out.index.intersection(p.index)
        if how == "ratio":
            if len(common) == 0:
                continue  # cannot be put on the same scale
            first = common.min()
            if p[first] == 0:
                continue
            p = p * (out[first] / p[first])
        idx = out.index.union(p.index)
        out = out.reindex(idx).combine_first(p.reindex(idx))
    return out


def load_sources(dl: Downloader, sources: tuple[Series, ...], how: str, notes: list[str], label: str) -> pd.Series | None:
    parts = []
    for spec in sources:
        try:
            parts.append(load_series(dl, spec))
        except (DataUnavailable, ValueError, KeyError) as err:
            notes.append(f"{label}: {spec.source}:{spec.key} unavailable ({str(err)[:120]})")
    s = splice(parts, how)
    return s if not s.empty else None


@dataclass
class CountryData:
    code: str
    cpi: pd.Series  # headline index, known-at grid
    core: pd.Series | None
    components: dict[str, pd.Series]
    oil: pd.Series | None
    predictors: dict[str, pd.Series]  # raw levels, known-at grid
    notes: list[str] = field(default_factory=list)


def build_country(dl: Downloader, c: cfg.InflationCountry) -> CountryData:
    notes: list[str] = []
    cpi = load_sources(dl, c.cpi, "ratio", notes, f"{c.code} CPI")
    if cpi is None:
        raise DataUnavailable(f"{c.code}: no CPI source available")
    core = load_sources(dl, c.core, "ratio", notes, f"{c.code} core") if c.core else None
    comps = {}
    for name, srcs in c.components.items():
        s = load_sources(dl, srcs, "ratio", notes, f"{c.code} {name}")
        if s is not None:
            comps[name] = s
    oil = load_sources(dl, c.oil_local, "ratio", notes, f"{c.code} oil") if c.oil_local else None
    preds = {}
    for name, p in {**c.predictors, **cfg.GLOBAL_PREDICTORS}.items():
        s = load_sources(dl, p.sources, p.splice, notes, f"{c.code} {name}")
        if s is not None:
            preds[name] = s
    return CountryData(c.code, cpi, core, comps, oil, preds, notes)


def oecd_cpi_panel(dl: Downloader) -> pd.DataFrame:
    """Headline CPI index for every country in the OECD prices database (reference months).

    The COICOP 2018 table is used where it exists (it runs to date for the
    countries that switched in 2026); the COICOP 1999 table fills the rest.
    """
    old = fetch_oecd_panel(dl, cfg.OECD_CPI.format(area="", item="_T"))
    try:
        new = fetch_oecd_panel(dl, cfg.OECD_CPI18.format(area="", item="_T"))
    except DataUnavailable:
        new = pd.DataFrame()
    cols = {}
    for area in sorted(set(old.columns) | set(new.columns)):
        parts = [df[area] for df in (new, old) if area in df]
        s = splice(parts, "ratio")
        s.index = pd.DatetimeIndex(s.index).to_period("M")
        cols[area] = s
    return pd.DataFrame(cols).sort_index()


def yoy(index: pd.Series) -> pd.Series:
    """Year-on-year inflation, % (100 x 12-month log change)."""
    return 100.0 * np.log(index.where(index > 0)).diff(12)


def global_factor(panel: pd.DataFrame, areas=cfg.GLOBAL_FACTOR_AREAS, lag: int = cfg.GLOBAL_FACTOR_LAG,
                  min_count: int = 10) -> pd.DataFrame:
    """Median and dispersion of y/y inflation across advanced economies, on the known-at grid."""
    infl = pd.DataFrame({a: yoy(panel[a]) for a in areas if a in panel})
    n = infl.notna().sum(axis=1)
    med = infl.median(axis=1).where(n >= min_count)
    iqr = (infl.quantile(0.75, axis=1) - infl.quantile(0.25, axis=1)).where(n >= min_count)
    out = pd.DataFrame({"g_infl": med, "g_infl_iqr": iqr})
    out.index = out.index + lag
    return out


@dataclass
class InflationData:
    countries: dict[str, CountryData]
    oecd_panel: pd.DataFrame  # CPI index by country, reference months
    global_block: pd.DataFrame  # known-at grid
    last_month: pd.Period  # last complete month (forecast origins stop here)


def build_data(codes=("US", "UK", "JP", "EA"), dl: Downloader | None = None, end: str | None = None) -> InflationData:
    dl = dl or Downloader()
    countries = {}
    for code in codes:
        log.info("loading %s", code)
        countries[code] = build_country(dl, cfg.COUNTRIES[code])
    panel = oecd_cpi_panel(dl)
    glob = global_factor(panel)
    last = pd.Period(end, "M") if end else pd.Timestamp.today().to_period("M") - 1
    return InflationData(countries, panel, glob, last)
