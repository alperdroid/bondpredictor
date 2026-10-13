"""Model zoo and run orchestration.

The models follow the suggested build order: curve factors first, then
macro and inflation-uncertainty factors, then supply, central-bank and
regime variables.  Each step adds a block of predictors to a shrinkage
estimator, so the out-of-sample table shows what each block contributes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .backtest import ModelSpec, PanelSpec, run_backtest, run_panel_backtest
from .evaluation import evaluate_strategy, summarize
from .learned import GlobalAverage, PCAFactors, cp_factor, cycle_factor
from .returns import rx_column

log = logging.getLogger(__name__)

N_MACRO_FACTORS = 5

# Country-specific risk/safe-haven variables.
EXTRA_RISK = {"US": ["US.credit"], "DE": ["DE.periphery"], "JP": []}


def _regimes(frame: pd.DataFrame, c: str) -> list[str]:
    return [col for col in frame.columns if col.startswith(f"{c}.reg_")]


def feature_blocks(c: str, countries: list[str], frame: pd.DataFrame) -> dict[str, tuple[list[str], list]]:
    """Predictor blocks for country ``c``: name -> (columns, transformers)."""
    cp = cp_factor(c)
    glob = GlobalAverage([cp_factor(k) for k in countries], output="G.cp")
    blocks = {
        "slope": (["{c}.slope"], []),
        "cp": (["{c}.cp"], [cp]),
        "global": (["G.cp"] + ([] if c == "US" else ["US.slope"]), [glob]),
        "cycle": (["{c}.cycle"], [cycle_factor(c)]),
        "carry": (["{c}.carry"], []),
        # Raw one-year forwards (inputs of Bianchi-Buchner-Tamoni's neural nets).
        "fwd": ([f"{{c}}.fwd_{k}" for k in range(1, 6)] + ["{c}.slope"], []),
        "macro": (
            [f"{{c}}.mf{i + 1}" for i in range(N_MACRO_FACTORS)] + ["{c}.output_gap", "{c}.infl_unc"],
            [PCAFactors(rf"^{c}\.m_", N_MACRO_FACTORS, output_prefix=f"{c}.mf")],
        ),
        "value_mom": (["{c}.real_yield", "{c}.mom"], []),
        "risk": (["{c}.rvol", "G.vix"] + EXTRA_RISK.get(c, []), []),
        "supply_policy": (
            ["{c}.debt_gdp_chg", "{c}.cb_gdp_chg", "{c}.policy_chg", "{c}.zlb"] + _regimes(frame, c),
            [],
        ),
    }
    return {k: ([f.replace("{c}", c) for f in cols], trs) for k, (cols, trs) in blocks.items()}


def _combine(blocks, names):
    feats, trs = [], []
    for n in names:
        f, t = blocks[n]
        feats += f
        trs += t
    return feats, trs


CURVE = ["slope", "cp", "global", "cycle", "carry"]
MACRO = CURVE + ["macro"]
ALL = MACRO + ["value_mom", "risk", "supply_policy"]

# (name, blocks, estimator, estimator kwargs)
MODEL_ZOO = [
    ("slope", ["slope"], "ols", {}),
    ("cp", ["cp"], "ols", {}),
    ("cp+global", ["cp", "global"], "ols", {}),
    ("curve", CURVE, "ridge", {}),
    ("curve+macro", MACRO, "ridge", {}),
    ("all:ridge", ALL, "ridge", {}),
    ("all:lasso", ALL, "lasso", {}),
    ("all:pcr", ALL, "pcr", {"k": 3}),
    ("all:combo", ALL, "combo", {}),
]


ML_ALL = ALL + ["fwd"]
ML_REFIT_MONTHS = 12

# Machine-learning models: refitted yearly, hyperparameters tuned by CV.
ML_ZOO = [
    ("ml:rf", ML_ALL, "rf", {}),
    ("ml:gbm", ML_ALL, "gbm", {}),
    ("ml:nn", ML_ALL, "nn", {}),
    ("ml:nn_fwd_macro", ["fwd", "macro"], "nn", {}),
]

# Equal-weighted averages of other models' forecasts, formed date by date.
ENSEMBLES = {"ml:ensemble": ["ml:rf", "ml:gbm", "ml:nn"]}


def country_specs(c: str, countries: list[str], frame: pd.DataFrame, maturity: float = 10, horizon: int = 12,
                  models=None, exclude_regimes=(), ml: bool = False) -> list[ModelSpec]:
    blocks = feature_blocks(c, countries, frame)
    target = f"{c}.{rx_column(horizon, maturity)}"
    zoo = [(n, b, e, k, 1) for n, b, e, k in MODEL_ZOO]
    if ml or (models and any(m.startswith("ml:") for m in models)):
        zoo += [(n, b, e, k, ML_REFIT_MONTHS) for n, b, e, k in ML_ZOO]
    specs = []
    for name, names, est, kw, refit in zoo:
        if models and name not in models:
            continue
        feats, trs = _combine(blocks, names)
        feats = list(dict.fromkeys(feats))
        specs.append(ModelSpec(name, target, feats, trs, est, dict(kw),
                               [f"{c}.reg_{r}" for r in exclude_regimes], refit_every=refit))
    return specs


def panel_spec(countries: list[str], frame: pd.DataFrame, maturity: float = 10, horizon: int = 12,
               blocks=tuple(ALL), estimator="ridge", exclude_regimes=()) -> PanelSpec:
    """Pooled model with common slopes and country fixed effects.

    Feature templates come from the first country's blocks with its code
    replaced by ``{c}``.  Regime dummies and the US-slope spillover are
    country-specific, so the pooled model leaves them out.
    """
    ref = countries[0]
    generic = []
    for name in blocks:
        for col in feature_blocks(ref, countries, frame)[name][0]:
            if col.startswith(f"{ref}.reg_") or col == "US.slope":
                continue
            generic.append(col.replace(f"{ref}.", "{c}.", 1) if col.startswith(f"{ref}.") else col)
    # One set of fitted factors per country; the global factor only once.
    uniq: dict[tuple, object] = {}
    for c in countries:
        cb = feature_blocks(c, countries, frame)
        for name in blocks:
            for t in cb[name][1]:
                uniq.setdefault(tuple(t.outputs), t)
    uniq = list(uniq.values())
    return PanelSpec(
        name=f"panel:{estimator}",
        countries=list(countries),
        target="{c}." + rx_column(horizon, maturity),
        features=list(dict.fromkeys(generic)),
        transformers=uniq,
        estimator=estimator,
        exclude_regimes=[f"{{c}}.reg_{r}" for r in exclude_regimes],
    )


@dataclass
class RunResult:
    forecasts: dict[tuple[str, str], pd.DataFrame]
    summary: pd.DataFrame


_FRAME: pd.DataFrame | None = None  # shared with forked worker processes


def _run_job(job):
    key, spec, kw = job
    if isinstance(spec, PanelSpec):
        return key, run_panel_backtest(_FRAME, spec, **kw)
    return key, run_backtest(_FRAME, spec, **kw)


def _run_job_single_threaded(job):
    """Worker entry point: one BLAS thread per process so workers don't oversubscribe the CPUs."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl ships with scikit-learn
        return _run_job(job)
    with threadpool_limits(limits=1):
        return _run_job(job)


def run_jobs(frame: pd.DataFrame, jobs: list, n_jobs: int = 1) -> dict:
    """Run backtest jobs ``(key, spec, kwargs)``, in parallel processes if ``n_jobs > 1``."""
    global _FRAME
    _FRAME = frame
    # Slow jobs first so the pool stays busy.
    jobs = sorted(jobs, key=lambda j: -(getattr(j[1], "refit_every", 1) > 1 or isinstance(j[1], PanelSpec)))
    out = {}
    if n_jobs <= 1:
        for job in jobs:
            log.info("backtest %s", job[0])
            k, r = _run_job(job)
            out[k] = r
        return out
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("fork")) as pool:
        for k, r in pool.map(_run_job_single_threaded, jobs):
            log.info("done %s", k)
            out[k] = r
    return out


def add_ensembles(forecasts: dict) -> dict:
    """Equal-weighted combinations of member forecasts on the dates all members cover."""
    countries = {c for c, _ in forecasts}
    for name, members in ENSEMBLES.items():
        for c in countries:
            parts = [forecasts[(c, m)] for m in members if (c, m) in forecasts]
            if len(parts) < 2:
                continue
            dates = parts[0].index
            for p in parts[1:]:
                dates = dates.intersection(p.index)
            frame = pd.DataFrame({
                "forecast": sum(p.loc[dates, "forecast"] for p in parts) / len(parts),
                "benchmark": sum(p.loc[dates, "benchmark"] for p in parts) / len(parts),
                "actual": parts[0].loc[dates, "actual"],
                "n_train": parts[0].loc[dates, "n_train"],
                "features": "+".join(members),
            })
            forecasts[(c, name)] = frame
    return forecasts


def run_all(frame: pd.DataFrame, countries: list[str], start: str = "1990-01", min_train: int = 120,
            maturity: float = 10, horizon: int = 12, models=None, panel: bool = True,
            exclude_regimes=(), common_sample: bool = True, ml: bool = False, n_jobs: int = 1,
            window: int | None = None) -> RunResult:
    kw = dict(start=start, min_train=min_train, window=window)
    jobs = []
    for c in countries:
        for spec in country_specs(c, countries, frame, maturity, horizon, models, exclude_regimes, ml):
            jobs.append(((c, spec.name), spec, kw))
    if panel and len(countries) > 1 and (not models or "panel:ridge" in models):
        spec = panel_spec(countries, frame, maturity, horizon, exclude_regimes=exclude_regimes)
        jobs.append((("*", spec.name), spec, kw))
    raw = run_jobs(frame, jobs, n_jobs)
    forecasts: dict[tuple[str, str], pd.DataFrame] = {}
    for (c, name), res in raw.items():
        if c == "*":
            for cc, r in res.items():
                forecasts[(cc, name)] = r
        elif not res.empty:
            forecasts[(c, name)] = res
    forecasts = add_ensembles(forecasts)
    summary = summarize_runs(frame, forecasts, maturity, horizon, common_sample)
    return RunResult(forecasts, summary)


def summarize_runs(frame, forecasts, maturity=10, horizon=12, common_sample=True) -> pd.DataFrame:
    rows = []
    by_country: dict[str, list] = {}
    for (c, name), res in forecasts.items():
        by_country.setdefault(c, []).append((name, res))
    for c, items in by_country.items():
        dates = None
        # Skip models whose predictors were never available (e.g. CP without a zero curve).
        items = [(n, r) for n, r in items if r["features"].astype(bool).any()]
        if common_sample:
            for _, res in items:
                d = res.dropna(subset=["actual"]).index
                dates = d if dates is None else dates.intersection(d)
        monthly = frame[f"{c}.{rx_column(1, maturity)}"]
        for name, res in items:
            r = res.loc[res.index.isin(dates)] if dates is not None else res
            stats = summarize(r, horizon)
            if stats.get("N", 0) >= 12:
                stats.update(evaluate_strategy(r.dropna(subset=["forecast"]), monthly, horizon))
            rows.append({"country": c, "model": name, **stats})
    return pd.DataFrame(rows)
