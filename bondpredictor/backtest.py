"""Real-time out-of-sample backtest (expanding or rolling window).

At each forecast origin ``t``:

1. Build the information set: all rows up to ``t``, with every target
   ``rx{h}_*`` blanked for rows after ``t - h``.  A return stored at ``s``
   is realised at ``s + h``.  With a rolling ``window`` only its last
   ``window`` months are kept.
2. Re-fit the learned factors (CP, cycle, global CP, macro PCs) on that
   information set.
3. Keep the predictors observed at ``t`` with enough history, fit the
   estimator on rows with a realised target, and forecast from row ``t``.
4. Record the prevailing historical mean over the same training rows as
   the benchmark.

With ``refit_every > 1`` (used for the ML models) steps 2–3 run only at
refit dates.  In between, the frozen factors and model are applied to each
new month's data, so the model never sees data from after its own refit
date.  Tuned hyperparameters (penalties, ML grids) are re-selected every
``cv_every`` months.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .learned import Transformer
from .models import make_estimator
from .returns import target_horizon

log = logging.getLogger(__name__)


def information_set(frame: pd.DataFrame, pos: int, window: int | None = None) -> pd.DataFrame:
    """Rows ``0..pos`` (or the last ``window`` of them) with unrealised returns blanked."""
    info = frame.iloc[: pos + 1].copy()
    for j, col in enumerate(info.columns):
        h = target_horizon(col)
        if h:
            info.iloc[max(pos - h + 1, 0) :, j] = np.nan
    if window:
        info = info.iloc[-window:]
    return info


@dataclass
class ModelSpec:
    name: str
    target: str
    features: list[str]
    transformers: list[Transformer] = field(default_factory=list)
    estimator: str = "ols"
    estimator_kw: dict = field(default_factory=dict)
    exclude_regimes: list[str] = field(default_factory=list)
    refit_every: int = 1

    @property
    def horizon(self) -> int:
        return target_horizon(self.target) or 1


def _select_features(train_ok: pd.Series, X: pd.DataFrame, candidates: list[str], min_train: int) -> list[str]:
    """Drop the sparsest predictors until enough complete training rows remain."""
    feats = list(candidates)
    while feats:
        n = (train_ok & X[feats].notna().all(axis=1)).sum()
        if n >= min_train:
            return feats
        coverage = X.loc[train_ok, feats].notna().sum()
        feats.remove(coverage.idxmin())
    return feats


def _exclusion_mask(data: pd.DataFrame, regimes: list[str]) -> pd.Series:
    mask = pd.Series(False, index=data.index)
    for col in regimes:
        if col in data:
            mask |= data[col].fillna(0.0) > 0.5
    return mask


def _prune(frame: pd.DataFrame, columns, transformers: list[Transformer]) -> pd.DataFrame:
    """Keep only the columns a spec reads (copying the full frame each month is slow)."""
    needed = set(columns)
    for tr in transformers:
        needed.update(tr.required_columns(frame.columns))
    return frame[[c for c in frame.columns if c in needed]]


def _apply(info: pd.DataFrame, fitted: list[Transformer]) -> pd.DataFrame:
    if not fitted:
        return info
    return pd.concat([info] + [tr.transform(info) for tr in fitted], axis=1)


def _fit_estimator(name, kw, horizon, X, y, cache: dict, pos: int, cv_every: int):
    """Fit an estimator, re-tuning its hyperparameter only every ``cv_every`` months."""
    kw = dict(kw)
    est = make_estimator(name, horizon=horizon, **kw)
    key = est.tune_key
    reuse = key is not None and key not in kw and "tuned" in cache and pos - cache["pos"] < cv_every
    if reuse:
        kw[key] = cache["tuned"]
        est = make_estimator(name, horizon=horizon, **kw)
    est.fit(X, y)
    if key is not None and key not in kw:
        cache.update(tuned=est.tuned_, pos=pos)
    return est


def _origins(idx: pd.PeriodIndex, start: str, end: str | None) -> range:
    first = idx.searchsorted(pd.Period(start, "M"))
    last = len(idx) - 1 if end is None else idx.searchsorted(pd.Period(end, "M"), side="right") - 1
    return range(first, last + 1)


def run_backtest(
    frame: pd.DataFrame,
    spec: ModelSpec,
    start: str = "1990-01",
    end: str | None = None,
    min_train: int = 120,
    cv_every: int = 12,
    window: int | None = None,
) -> pd.DataFrame:
    """Forecasts for one country/target, one row per forecast origin.

    Columns: ``forecast``, ``benchmark`` (prevailing mean), ``actual`` (NaN
    until realised), ``n_train`` and the predictors used.
    """
    h = spec.horizon
    frame = _prune(frame, [spec.target, *spec.features, *spec.exclude_regimes], spec.transformers)
    idx = frame.index
    rows, cache = [], {}
    model = None  # (fitted transformers, estimator, features, refit position)
    for pos in _origins(idx, start, end):
        info = information_set(frame, pos, window)
        due = model is None or pos - model[3] >= spec.refit_every
        if not due:
            data = _apply(info, model[0])
            due = data[model[2]].iloc[-1].isna().any()  # a predictor went missing: refit now
        if due:
            fitted = [tr.clone().fit(info) for tr in spec.transformers]
            data = _apply(info, fitted)
        y = data[spec.target]
        train_ok = y.notna() & ~_exclusion_mask(data, spec.exclude_regimes)
        if due:
            cands = [f for f in spec.features if f in data and pd.notna(data[f].iloc[-1])]
            feats = _select_features(train_ok, data, cands, min_train)
        else:
            feats = model[2]
        ok = (train_ok & data[feats].notna().all(axis=1)) if feats else train_ok
        if ok.sum() < min_train:
            model = None
            continue
        ytr = y[ok].to_numpy()
        bench = float(ytr.mean())
        if due:
            est = None
            if feats:
                est = _fit_estimator(spec.estimator, spec.estimator_kw, h,
                                     data.loc[ok, feats].to_numpy(dtype=float), ytr, cache, pos, cv_every)
            model = (fitted, est, feats, pos)
        est = model[1]
        fc = float(est.predict(data[feats].iloc[[-1]].to_numpy(dtype=float))[0]) if est is not None else bench
        rows.append(
            {
                "date": idx[pos],
                "forecast": fc,
                "benchmark": bench,
                "actual": frame[spec.target].iloc[pos],
                "n_train": int(ok.sum()),
                "features": ",".join(feats),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.set_index("date")


@dataclass
class PanelSpec:
    """Pooled model across countries with country fixed effects.

    ``target`` and ``features`` are templates with ``{c}`` for the country
    code, e.g. ``"{c}.rx12_n10"`` or ``"{c}.slope"``.  Global columns
    (``"G.cp"``) are used as is.
    """

    name: str
    countries: list[str]
    target: str
    features: list[str]
    transformers: list[Transformer] = field(default_factory=list)
    estimator: str = "ridge"
    estimator_kw: dict = field(default_factory=dict)
    exclude_regimes: list[str] = field(default_factory=list)

    @property
    def horizon(self) -> int:
        return target_horizon(self.target) or 1


def _generic(name: str) -> str:
    return name.replace("{c}.", "")


def run_panel_backtest(
    frame: pd.DataFrame,
    spec: PanelSpec,
    start: str = "1990-01",
    end: str | None = None,
    min_train: int = 120,
    cv_every: int = 12,
    window: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Forecasts from a pooled panel; one result frame per country."""
    h = spec.horizon
    templates = [spec.target, *spec.features, *spec.exclude_regimes]
    frame = _prune(frame, [t.format(c=c) for t in templates for c in spec.countries], spec.transformers)
    idx = frame.index
    generic = [_generic(f) for f in spec.features]
    rows: dict[str, list] = {c: [] for c in spec.countries}
    cache: dict = {}
    for pos in _origins(idx, start, end):
        info = information_set(frame, pos, window)
        data = _apply(info, [tr.clone().fit(info) for tr in spec.transformers])
        blocks, origin_x, means = [], {}, {}
        for c in spec.countries:
            tcol = spec.target.format(c=c)
            if tcol not in data:
                continue
            cols = [f.format(c=c) for f in spec.features]
            X = pd.DataFrame({g: data[col] if col in data else np.nan for g, col in zip(generic, cols)})
            y = data[tcol]
            excl = _exclusion_mask(data, [r.format(c=c) for r in spec.exclude_regimes])
            train_ok = y.notna() & ~excl
            present = _select_features(train_ok, X, [g for g in generic if pd.notna(X[g].iloc[-1])], min_train)
            ok = train_ok & X[present].notna().all(axis=1)
            if ok.sum() < min_train:
                continue
            Xc = X[ok].copy()
            xbar, ybar = Xc.mean(), float(y[ok].mean())
            Xd = Xc - xbar
            Xd[[g for g in generic if g not in present]] = 0.0
            blocks.append(pd.DataFrame({"_y": y[ok] - ybar, "_c": c, **{g: Xd[g] for g in generic}}))
            origin_x[c] = (X.iloc[-1] - xbar).where(X.columns.isin(present), 0.0)
            means[c] = (ybar, int(ok.sum()), present)
        if not blocks:
            continue
        stacked = pd.concat(blocks).sort_index(kind="stable")
        use = [g for g in generic if stacked[g].abs().sum() > 0]
        if len(stacked) < min_train:
            continue
        if use:
            # Rows are stacked by date, so the CV gap is horizon x countries rows.
            est = _fit_estimator(spec.estimator, spec.estimator_kw, h * len(blocks),
                                 stacked[use].to_numpy(dtype=float), stacked["_y"].to_numpy(), cache, pos, cv_every)
        for c, xt in origin_x.items():
            ybar, n, present = means[c]
            fc = ybar + (float(est.predict(xt[use].to_numpy(dtype=float)[None, :])[0]) if use else 0.0)
            rows[c].append(
                {
                    "date": idx[pos],
                    "forecast": fc,
                    "benchmark": ybar,
                    "actual": frame[spec.target.format(c=c)].iloc[pos],
                    "n_train": n,
                    "features": ",".join(g for g in use if g in present),
                }
            )
    return {c: pd.DataFrame(r).set_index("date") for c, r in rows.items() if r}
