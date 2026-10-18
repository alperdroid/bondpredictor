"""What-if scenarios for the latest forecasts.

A scenario moves drivers *today*: oil 30% dearer than it is, the currency
10% weaker, supply chains one index point more stretched.  Each tier-1
country model is refitted on all realised data (as for the latest
forecasts), the latest predictor row is shifted the way the shock would show
up in this month's data, and the model forecasts again.  Its response is the
change in its forecast.

* Price shocks (oil, producer prices, the currency) are level shifts this
  month, so they move the 1-, 3- and 12-month changes alike, in log points.
* Rates, survey levels and growth rates shift by the stated number of
  points, together with their recent changes.
* Levers combine additively within each model; the ensemble is the median
  across models of the combined forecasts.  ``run_scenarios`` also solves
  the preset scenarios jointly, so the error of that shortcut is measured.

These are reduced-form what-ifs: the models learned correlations, not causal
effects, and a shock does not move the other drivers.  The policy rate is
left out because central banks set it in response to the inflation outlook,
so its correlation with future inflation has the wrong sign for a policy
lever.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import design, fit_at, model_zoo

ECONOMIES = ("US", "UK", "JP", "EA")
# The ensemble's members: every tier-1 country model except the benchmarks.
MODELS = tuple(s.name for s in model_zoo(1) if s.name not in ("rw", "ucsv", "ar"))


def _same(*cols: str) -> dict[str, dict[str, float]]:
    return {c: {col: 1.0 for col in cols} for c in ECONOMIES}


@dataclass(frozen=True)
class Lever:
    key: str
    label: str
    unit: str
    grid: tuple[float, ...]
    columns: dict[str, dict[str, float]]  # economy -> {predictor: weight}
    log: bool = False  # the shock is a % change in a price level, applied as 100 * log(1 + s / 100)
    note: str = ""

    def shift(self, economy: str, value: float) -> dict[str, float]:
        size = 100.0 * math.log1p(value / 100.0) if self.log else value
        return {col: w * size for col, w in self.columns.get(economy, {}).items()}


LEVERS = (
    Lever("oil", "Oil price", "% vs today", (-50, -30, -15, 0, 15, 30, 50, 75, 100),
          _same("oil_yoy", "oil_c3", "oil_c1"), log=True,
          note="Brent jumps (or drops) this month; its 1-, 3- and 12-month changes all move."),
    Lever("supply", "Supply-chain pressure", "GSCPI points", (-2, -1, 0, 1, 2, 3, 4),
          _same("gscpi_lvl", "gscpi_d3"),
          note="New York Fed Global Supply Chain Pressure Index, in standard deviations from its average."),
    Lever("ppi", "Producer prices", "% vs today", (-10, -5, 0, 5, 10, 15, 20),
          {"US": {"ppi_yoy": 1.0, "ppi_c3": 1.0}, "UK": {"ppi_yoy": 1.0}, "JP": {"ppi_yoy": 1.0, "ppi_c3": 1.0},
           "EA": {"ppi_yoy": 1.0}}, log=True,
          note="Factory-gate prices jump this month."),
    Lever("currency", "Currency", "% stronger", (-20, -10, -5, 0, 5, 10, 20),
          {"US": {"dollar_yoy": 1.0}, "UK": {"reer_yoy": 1.0, "gbpusd_yoy": 1.0},
           "JP": {"reer_yoy": 1.0, "usdjpy_yoy": -1.0}, "EA": {"neer_yoy": 1.0, "eurusd_yoy": 1.0}}, log=True,
          note="Trade-weighted and dollar exchange rates; negative = depreciation."),
    Lever("wages", "Wage growth", "pp", (-3, -2, -1, 0, 1, 2, 3, 4),
          {"US": {"wages_yoy": 1.0, "wage_tracker_lvl": 1.0}, "UK": {"earnings_yoy": 1.0},
           "JP": {"earnings_yoy": 1.0}},
          note="Year-on-year growth of average earnings."),
    Lever("unemployment", "Unemployment rate", "pp", (-2, -1, 0, 1, 2, 3, 4),
          _same("unemp_lvl", "unemp_d12", "unemp_gap"),
          note="The jobless rate jumps this month."),
    Lever("expectations", "Inflation expectations", "pp", (-1, -0.5, 0, 0.5, 1, 1.5, 2),
          {"US": {"cleveland1y_lvl": 1.0, "mich_lvl": 1.0, "mich_d12": 1.0, "breakeven5y_lvl": 1.0,
                  "breakeven5y_d3": 1.0},
           "UK": {"breakeven10_lvl": 1.0, "breakeven10_d12": 1.0},
           "JP": {"tankan_infl_outlook_1y_lvl": 1.0}},
          note="Survey and market measures of expected inflation all move by the same amount."),
    Lever("abroad", "Inflation abroad", "pp", (-2, -1, 0, 1, 2, 3, 4),
          {c: {"g_infl": 1.0, "g_infl_d6": 1.0, **{f"x_{o}": 1.0 for o in ECONOMIES if o != c}} for c in ECONOMIES},
          note="Inflation in the other three economies and the 20-economy median."),
)
LEVER_BY_KEY = {lv.key: lv for lv in LEVERS}


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    blurb: str
    values: dict[str, float] = field(default_factory=dict)


PRESETS = (
    Preset("oil_shock", "Oil shock", "Oil 50% dearer and supply chains under strain.",
           {"oil": 50, "supply": 1}),
    Preset("supply_crunch", "Supply crunch", "A 2021-style squeeze: shipping and factory-gate prices surge.",
           {"supply": 3, "ppi": 8, "abroad": 1.5}),
    Preset("weak_currency", "Currency slide", "The currency loses 15% against its trading partners.",
           {"currency": -15}),
    Preset("wage_spiral", "Wage-price spiral", "A tight labour market: wages up 3 points, expectations up 1.",
           {"wages": 3, "expectations": 1, "unemployment": -1}),
    Preset("recession", "Recession", "Jobless rate up 2.5 points, oil down 35%, supply chains slack.",
           {"unemployment": 2.5, "oil": -35, "wages": -1.5, "supply": -1, "abroad": -1}),
    Preset("disinflation", "Soft landing", "Oil eases, supply chains normalise, the currency firms.",
           {"oil": -15, "supply": -0.5, "currency": 5, "expectations": -0.3}),
)


def _shifted_rows(base: np.ndarray, names: list[str], economy: str, shocks: list[dict[str, float]]) -> np.ndarray:
    """One row per shock (lever key -> value), the base row moved by each lever's shift."""
    pos = {n: i for i, n in enumerate(names)}
    rows = np.repeat(base[None, :], len(shocks), axis=0)
    for r, shock in enumerate(shocks):
        for key, value in shock.items():
            for col, d in LEVER_BY_KEY[key].shift(economy, value).items():
                if col in pos:
                    rows[r, pos[col]] += d
    return rows


def model_responses(frame: pd.DataFrame, groups: dict[str, str], economy: str, h: int, spec, min_train: int = 120,
                    min_hist: int = 60) -> dict:
    """Refit one model at the latest origin and evaluate every lever's grid and every preset."""
    X, g, pi, tgt = design(frame, groups, h)
    names = list(X.columns)
    Xv = X.to_numpy(dtype=float)
    piv = pi.to_numpy(dtype=float)
    y = (tgt - pi).to_numpy()
    k = int(np.flatnonzero(np.isfinite(piv))[-1])
    t0 = max(frame.index.searchsorted(pd.Period(spec.train_start, "M")), 0)
    fit = fit_at(Xv, names, g, y, piv, k, h, spec, t0, min_train, min_hist)
    if fit is None:
        return {}
    model, cols, n_train = fit
    used = [names[i] for i in cols]
    base_row = Xv[k, cols]
    c = economy
    shocks = [{}]
    for lv in LEVERS:
        shocks += [{lv.key: v} for v in lv.grid]
    for p in PRESETS:
        shocks.append(dict(p.values))
        shocks += [{key: v} for key, v in p.values.items()]
    pred = np.asarray(model.predict(_shifted_rows(base_row, used, c, shocks)), dtype=float)
    base = pred[0]
    resp = pred - base
    out = {"base": float(piv[k] + base), "pi": float(piv[k]), "origin": str(frame.index[k]), "n_train": n_train,
           "curves": {}, "joint": {}, "additive": {}}
    i = 1
    for lv in LEVERS:
        out["curves"][lv.key] = [float(v) for v in resp[i:i + len(lv.grid)]]
        i += len(lv.grid)
    for p in PRESETS:
        out["joint"][p.key] = float(resp[i])
        out["additive"][p.key] = float(resp[i + 1:i + 1 + len(p.values)].sum())
        i += 1 + len(p.values)
    out["active"] = {lv.key: sorted(set(lv.columns.get(c, {})) & set(used)) for lv in LEVERS}
    return out


_FRAMES: dict = {}


def _job(job):
    c, h, spec = job
    import threadpoolctl

    frame, groups = _FRAMES[c]
    with threadpoolctl.threadpool_limits(limits=1):
        return (c, h, spec.name), model_responses(frame, groups, c, h, spec)


def run_scenarios(frames: dict, horizons=(3, 6, 12, 24), n_jobs: int = 1, progress=None, done: dict | None = None) -> dict:
    """``{(economy, horizon, model): responses}`` for every tier-1 country model; skips keys in ``done``."""
    global _FRAMES
    _FRAMES = frames
    cost = {"hnn": 6, "mlp": 3, "bart": 5, "mrf": 3, "krr": 2, "rf": 2, "lgbm": 2}
    specs = [s for s in model_zoo(1) if s.name in MODELS]
    jobs = [(c, h, s) for c in frames for h in horizons for s in specs if (c, h, s.name) not in (done or {})]
    jobs.sort(key=lambda j: -cost.get(j[2].model, 0))
    out = dict(done or {})

    def keep(key, r):
        out[key] = r
        if progress:
            progress(key, r)

    if n_jobs <= 1:
        for j in jobs:
            keep(*_job(j))
        return out
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("fork")) as pool:
        for f in as_completed([pool.submit(_job, j) for j in jobs]):
            keep(*f.result())
    return out


def summarise(responses: dict) -> pd.DataFrame:
    """Median-of-models response per economy, horizon and preset: joint versus lever-by-lever."""
    rows = []
    for (c, h, m), r in responses.items():
        for p in r.get("joint", {}):
            rows.append({"economy": c, "horizon": h, "model": m, "preset": p, "base": r["base"],
                         "joint": r["base"] + r["joint"][p], "additive": r["base"] + r["additive"][p]})
    df = pd.DataFrame(rows)
    med = df.groupby(["economy", "horizon", "preset"])[["base", "joint", "additive"]].median()
    med["joint change"] = med["joint"] - med["base"]
    med["additive change"] = med["additive"] - med["base"]
    med["gap"] = med["additive change"] - med["joint change"]
    return med.reset_index()
