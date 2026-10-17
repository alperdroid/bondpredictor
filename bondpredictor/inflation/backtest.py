"""Real-time expanding-window backtest of the inflation models.

At each forecast origin ``k`` (end of month ``k``):

1. training rows are the months ``j`` whose target has been published by
   ``k`` (``j + h <= k``) and that start no earlier than ``train_start``;
2. at refit dates the model keeps the predictors observed at ``k`` with at
   least ``min_hist`` training observations, and is fitted on those rows;
3. between refits (``refit_every`` months, 12 for the ML models) the frozen
   model is applied to each new month's predictors, refitting early if a
   predictor it uses goes missing;
4. the forecast of y/y inflation is the latest rate plus the forecast
   change.

The random walk forecasts no change; UCSV uses its filtered trend (see
``ucsv.py``) for the months not yet observed.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import generic_name, horizon_columns

log = logging.getLogger(__name__)

# Own-inflation terms for the AR-type benchmark.
AR_FEATURES = ("pi", "pi_d3", "pi_d12", "sa3", "sa6", "mech", "drop", "tax")


def make_model(name: str, horizon: int, **kw):
    from . import bart, econ, hnn, mrf

    registry = {
        "ols": econ.OLS, "ridge": econ.Ridge, "adalasso": econ.AdaptiveLasso, "enet": econ.ElasticNet,
        "di": econ.DiffusionIndex, "3prf": econ.ThreePassFilter, "csr": econ.SubsetRegression,
        "krr": econ.KernelRidge, "rf": econ.RandomForest, "lgbm": econ.Boosting,
        "mrf": mrf.MacroRandomForest, "bart": bart.BART, "hnn": hnn.HemisphereNN, "mlp": hnn.MLP,
    }
    return registry[name](horizon=horizon, **kw)


@dataclass
class Spec:
    name: str
    model: str  # "rw", "ucsv" or a key of make_model
    kw: dict = field(default_factory=dict)
    features: tuple[str, ...] | None = None  # generic feature names; None = all
    refit_every: int = 12
    window: int | None = None  # rolling window in months (None = expanding)
    train_start: str = "1960-01"
    tier: int = 1  # 1 = econometric ML and benchmarks, 2 = generic deep learning


def design(frame: pd.DataFrame, groups: dict[str, str], h: int):
    """Predictors with horizon-free names, their groups, the latest inflation rate and the target."""
    cols = horizon_columns(frame, groups, h)
    gen = [generic_name(c, h) for c in cols]
    X = frame[cols].copy()
    X.columns = gen
    g = {generic_name(c, h): groups[c] for c in cols}
    return X, g, frame["pi"], frame[f"tgt{h}"]


def _benchmark_change(spec: Spec, X: pd.DataFrame, frame: pd.DataFrame, pi: pd.Series, h: int) -> pd.Series:
    if spec.model == "rw":
        return pd.Series(0.0, index=frame.index)
    tau = frame["ucsv_tau"]
    if h < 12:
        return (h / 12.0) * (tau - X["drop"])
    return tau - pi


def run_backtest(frame: pd.DataFrame, groups: dict[str, str], h: int, spec: Spec, start: str = "2000-01",
                 end: str | None = None, min_train: int = 120, min_hist: int = 60,
                 explain: bool = False) -> pd.DataFrame:
    """One row per origin: ``forecast``, ``rw``, ``actual`` (NaN until realised) and diagnostics."""
    X, g, pi, tgt = design(frame, groups, h)
    if spec.features is not None:
        X = X[[c for c in spec.features if c in X]]
    y = (tgt - pi).to_numpy()
    idx = frame.index
    first = idx.searchsorted(pd.Period(start, "M"))
    last = len(idx) - 1 if end is None else idx.searchsorted(pd.Period(end, "M"), side="right") - 1
    t0 = max(idx.searchsorted(pd.Period(spec.train_start, "M")), 0)
    Xv = X.to_numpy(dtype=float)
    piv = pi.to_numpy(dtype=float)
    bench = _benchmark_change(spec, X, frame, pi, h).to_numpy() if spec.model in ("rw", "ucsv") else None
    rows = []
    model, feats, cols, fitted_at, n_train = None, [], [], -10**9, 0
    for k in range(first, last + 1):
        if not np.isfinite(piv[k]):
            continue
        extra = {}
        if bench is not None:
            change = bench[k]
            if not np.isfinite(change):
                continue
        else:
            lo = t0 if spec.window is None else max(t0, k - h - spec.window + 1)
            train = np.arange(lo, k - h + 1)
            train = train[np.isfinite(y[train]) & np.isfinite(piv[train])]
            due = model is None or k - fitted_at >= spec.refit_every or not np.all(np.isfinite(Xv[k, cols]))
            if due:
                if len(train) < min_train:
                    continue
                ok_now = np.isfinite(Xv[k])
                hist = np.isfinite(Xv[train]).sum(axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-missing columns
                    sd = np.nanstd(Xv[train], axis=0)
                cols = [i for i in range(Xv.shape[1]) if ok_now[i] and hist[i] >= min_hist and sd[i] > 1e-12]
                feats = [X.columns[i] for i in cols]
                if not cols:
                    continue
                model = make_model(spec.model, h, **spec.kw)
                model.fit(Xv[np.ix_(train, cols)], y[train], feats, [g[f] for f in feats])
                fitted_at, n_train = k, len(train)
            row = Xv[k, cols][None, :]
            change = float(model.predict(row)[0])
            if hasattr(model, "predict_quantiles"):
                q = model.predict_quantiles(row, (0.1, 0.5, 0.9))[0]
                extra.update({"q10": piv[k] + q[0], "q50": piv[k] + q[1], "q90": piv[k] + q[2]})
            if explain and hasattr(model, "contributions"):
                labels, parts = model.contributions(row)
                extra.update({f"c_{lab}": float(v) for lab, v in zip(labels, parts[0])})
            if explain and hasattr(model, "gtvp"):
                names, b = model.gtvp(row)
                extra.update({f"b_{n}": float(v) for n, v in zip(names, b[0])})
        rows.append({
            "date": idx[k],
            "forecast": piv[k] + change,
            "rw": piv[k],
            "actual": tgt.iloc[k],
            "n_train": n_train if bench is None else 0,
            "n_features": len(cols) if bench is None else 0,
            "refit": bench is None and fitted_at == k,
            **extra,
        })
    out = pd.DataFrame(rows)
    return out.set_index("date") if not out.empty else out


# ------------------------------------------------------------------ zoo

def model_zoo(tier: int = 1) -> list[Spec]:
    """Tier 1: benchmarks and econometric ML (run first, full robustness).  Tier 2 lives in ``deep.py``."""
    specs = [
        Spec("rw", "rw", refit_every=1),
        Spec("ucsv", "ucsv", refit_every=1),
        Spec("ar", "ols", features=AR_FEATURES, refit_every=1),
        Spec("ridge", "ridge", refit_every=1),
        Spec("adalasso", "adalasso", refit_every=12),
        Spec("enet", "enet", refit_every=12),
        Spec("di", "di", refit_every=1),
        Spec("3prf", "3prf", refit_every=1),
        Spec("csr", "csr", refit_every=12),
        Spec("krr", "krr", refit_every=12),
        Spec("rf", "rf", refit_every=12),
        Spec("lgbm", "lgbm", refit_every=12),
        Spec("mrf", "mrf", refit_every=12),
        Spec("bart", "bart", refit_every=12),
        Spec("hnn", "hnn", refit_every=12),
        Spec("mlp", "mlp", refit_every=12),
    ]
    return [s for s in specs if s.tier <= tier]


# ------------------------------------------------------------- running

_FRAMES: dict = {}


def _run_job(job):
    key, spec, kw = job
    country, h = key[0], key[1]
    frame, groups = _FRAMES[country]
    import threadpoolctl

    with threadpoolctl.threadpool_limits(limits=1):
        try:
            return key, run_backtest(frame, groups, h, spec, **kw)
        except Exception as err:  # keep the other jobs running
            log.exception("job %s failed", key)
            return key, err


def run_jobs(frames: dict, jobs: list, n_jobs: int = 1, progress=None) -> dict:
    """Run ``(key, spec, kwargs)`` jobs; ``key`` = (country, horizon, label).  ``frames``: country -> (frame, groups)."""
    global _FRAMES
    _FRAMES = frames
    cost = {"hnn": 6, "mlp": 3, "bart": 5, "mrf": 3, "krr": 2, "rf": 2, "lgbm": 2, "adalasso": 1, "csr": 1}
    jobs = sorted(jobs, key=lambda j: -cost.get(j[1].model, 0))
    out = {}
    if n_jobs <= 1:
        for job in jobs:
            k, r = _run_job(job)
            out[k] = r
            if progress:
                progress(k, r)
        return out
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("fork")) as pool:
        futs = [pool.submit(_run_job, j) for j in jobs]
        for f in as_completed(futs):
            k, r = f.result()
            out[k] = r
            if progress:
                progress(k, r)
    return out
