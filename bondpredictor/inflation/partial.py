"""Partial-horizon training for the direct models (Montes-Galdón 2026, ECB Working Paper 3307, section 4.2.2).

A direct 12-month model learns only from origins whose target is published, so at origin ``k`` its newest
training row is ``k - 12``: in February 2021 it had seen nothing after February 2020.  Here every training
origin ``j`` enters once for each horizon ``m = 1..12`` whose outcome is out by ``k``, on a scale that is
comparable across horizons:

    z[j, m] = (12 / m) (pi[j+m] - pi[j]) + drop_m[j] - pi[j]

``(12 / m) (pi[j+m] - pi[j]) + drop_m`` is the average annualised inflation of the next ``m`` months (the
seasonal factors of the same calendar months cancel against ``drop_m``, the seasonally adjusted rate of the
months leaving the 12-month window), so ``z[j, m]`` is "average inflation over the next m months minus
today's y/y rate".  At ``m = 12`` (``drop_12 = pi``) it is exactly the usual target ``pi[j+12] - pi[j]``.
Each cell is weighted by ``m / 12``: an m-month average carries 1/m of one month's noise variance.  The
model also sees the horizon (``hfrac = m / 12``; for the linear models also its product with every
predictor, so coefficients can change with the horizon) and that horizon's base-effect and tax columns.
Forecasts are made at ``m = 12``.

* ``stack="partial"`` keeps origins whose 12 months are only partly published (``k - 12 < j < k``), scored
  on the months already out: the paper's partial horizons.
* ``stack="complete"`` uses horizons 1..12 only for origins whose 12 months are all out.  Comparing the two
  separates the effect of the partial rows from that of training on shorter horizons.

Hyperparameters are tuned every 12 months by expanding-window validation on the 12-month cells, with each
fold's training cells restricted to those published by the start of its validation block; coefficients
are refitted every ``refit_every`` months like the model's standard version.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

H_STACK = 12
LINEAR = ("ols", "ridge", "enet")
SUPPORTED = ("ols", "ridge", "enet", "rf", "lgbm")
TUNE_EVERY = 12


def horizon_parts(frame: pd.DataFrame, groups: dict[str, str], h: int = H_STACK):
    """Base predictors, per-horizon (drop, mech[, tax]) columns and stacked targets for m = 1..h."""
    from . import config as cfg
    from .backtest import design
    from .features import tax_feature

    X, g, pi, _ = design(frame, groups, h)
    base = [c for c in X.columns if c not in ("drop", "mech", "tax")]
    sa1, sa6 = frame["sa1"], frame["sa6"]
    pt = frame[f"tgt{h}"].shift(h)  # y/y with interpolated CPI months blanked (targets only)
    country = cfg.COUNTRIES[frame.attrs["country"]]
    has_tax = "tax" in X.columns
    hcols = ["drop", "mech"] + (["tax"] if has_tax else [])
    HF = np.empty((h, len(frame), len(hcols)))
    Z = np.empty((h, len(frame)))
    for m in range(1, h + 1):
        drop = sa1.rolling(m).mean().shift(12 - m) if m < 12 else pi
        mech = (m / 12.0) * (sa6 - drop) if m < 12 else sa6 - pi
        cols = [drop, mech]
        if has_tax:
            cols.append(X["tax"] if m == h else tax_feature(frame.index, country.tax_events, m, country.cpi[0].lag))
        HF[m - 1] = np.column_stack([np.asarray(c, float) for c in cols])
        Z[m - 1] = ((12.0 / m) * (pt.shift(-m) - pi) + drop - pi).to_numpy(float)
    return X[base], {c: g[c] for c in base}, hcols, HF, Z, pi.to_numpy(float)


def stacked_cells(k: int, t0: int, h: int, Z: np.ndarray, ok: np.ndarray, partial: bool):
    """(origin, horizon) cells published by origin ``k``."""
    J, M = [], []
    for m in range(1, h + 1):
        hi = k - m if partial else k - h
        js = np.arange(t0, hi + 1)
        js = js[np.isfinite(Z[m - 1, js]) & ok[js]]
        J.append(js)
        M.append(np.full(len(js), m))
    return np.concatenate(J), np.concatenate(M)


def stacked_matrix(Xb: np.ndarray, HF: np.ndarray, J: np.ndarray, M: np.ndarray, linear: bool) -> np.ndarray:
    B = np.column_stack([Xb[J], HF[M - 1, J]])
    hf = (M / H_STACK)[:, None]
    return np.column_stack([B, hf, hf * B]) if linear else np.column_stack([B, hf])


def _wmoments(D: np.ndarray, w: np.ndarray):
    ok = np.isfinite(D)
    W = np.where(ok, w[:, None], 0.0)
    tot = W.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.where(tot > 0, (W * np.where(ok, D, 0.0)).sum(axis=0) / tot, 0.0)
        var = np.where(tot > 0, (W * np.where(ok, D - mu, 0.0) ** 2).sum(axis=0) / tot, 1.0)
    sd = np.sqrt(var)
    return mu, np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)


def fit_stacked(model, D, y, w, J, M, names, groups, h: int, partial: bool, params: dict | None):
    """Weighted standardisation, date-blocked tuning (when ``params`` is None) and a weighted fit."""
    model.names, model.groups = list(names), list(groups)
    w = w / w.mean()
    mu, sd = _wmoments(D, w)
    Z = (D - mu) / sd
    model.mu, model.sd = mu, sd
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.zmin, model.zmax = np.nanmin(Z, axis=0), np.nanmax(Z, axis=0)
    if not model.nan_ok:
        Z = np.where(np.isfinite(Z), Z, 0.0)
    model.ybar = float(np.sum(w * y) / np.sum(w))
    ysd = float(np.sqrt(np.sum(w * (y - model.ybar) ** 2) / np.sum(w)))
    model.ysd = ysd if ysd > 1e-12 else 1.0
    yz = (y - model.ybar) / model.ysd
    if params is None:
        params = _tune(model, Z, yz, w, J, M, h, partial)
    model.params_ = model.tuned_ = dict(params)
    model.sample_weight_ = w
    model._fit(Z, yz, params)
    return model


def _tune(model, Z, yz, w, J, M, h, partial) -> dict:
    if len(model.grid) == 1:
        return dict(model.grid[0])
    dates = np.unique(J)
    edges = np.linspace(dates[len(dates) // 2], dates[-1] + 1, model.n_folds + 1).astype(int)
    sse = np.zeros(len(model.grid))
    used = 0
    for v0, v1 in zip(edges[:-1], edges[1:]):
        fit = (J + M <= v0) if partial else (J + h <= v0)
        val = (J >= v0) & (J < v1) & (M == h)
        if len(np.unique(J[fit])) < 60 or not val.any():
            continue
        for i, p in enumerate(model.grid):
            model.sample_weight_ = w[fit] / w[fit].mean()
            model._fit(Z[fit], yz[fit], p)
            sse[i] += np.sum((yz[val] - model._predict(Z[val])) ** 2)
        used += 1
    return dict(model.grid[int(np.argmin(sse))]) if used else dict(model.grid[0])


def run_stacked(frame: pd.DataFrame, groups: dict[str, str], h: int, spec, start: str = "2000-01",
                end: str | None = None, min_train: int = 120, min_hist: int = 60) -> pd.DataFrame:
    """Backtest of ``spec`` trained on stacked horizons (``spec.stack`` = "partial" or "complete")."""
    from .backtest import make_model

    if h != H_STACK:
        raise ValueError(f"stacked training is defined for h = {H_STACK}")
    if spec.model not in SUPPORTED:
        raise ValueError(f"stacked training supports {SUPPORTED}, not {spec.model}")
    if h in spec.anchor_horizons or spec.offset:
        raise ValueError("stacked training does not combine with a target offset")
    partial = spec.stack == "partial"
    linear = spec.model in LINEAR
    Xb_df, gb, hcols, HF, Z, piv = horizon_parts(frame, groups, h)
    if spec.features is not None:
        keep = [c for c in Xb_df.columns if c in spec.features]
        Xb_df, gb = Xb_df[keep], {c: gb[c] for c in keep}
        hcols_keep = [i for i, c in enumerate(hcols) if c in spec.features]
        HF, hcols = HF[:, :, hcols_keep], [hcols[i] for i in hcols_keep]
    Xb = Xb_df.to_numpy(float)
    bnames = list(Xb_df.columns)
    idx = frame.index
    tgt = frame[f"tgt{h}"]
    ok = np.isfinite(piv)
    first = idx.searchsorted(pd.Period(start, "M"))
    last = len(idx) - 1 if end is None else idx.searchsorted(pd.Period(end, "M"), side="right") - 1
    t0 = max(idx.searchsorted(pd.Period(spec.train_start, "M")), 0)
    rows = []
    model, cols, fitted_at, tuned_at, params, n_train = None, [], -10**9, -10**9, None, 0
    for k in range(first, last + 1):
        if not np.isfinite(piv[k]):
            continue
        due = model is None or k - fitted_at >= spec.refit_every or not np.all(np.isfinite(Xb[k, cols]))
        if due:
            J, M = stacked_cells(k, t0, h, Z, ok, partial)
            origins = np.unique(J)
            if len(np.unique(J[M == h])) < min_train:
                continue
            ok_now = np.isfinite(Xb[k])
            hist = np.isfinite(Xb[origins]).sum(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sdv = np.nanstd(Xb[origins], axis=0)
            new_cols = [i for i in range(Xb.shape[1]) if ok_now[i] and hist[i] >= min_hist and sdv[i] > 1e-12]
            if not new_cols:
                continue
            retune = k - tuned_at >= TUNE_EVERY or new_cols != cols
            cols = new_cols
            D = stacked_matrix(Xb[:, cols], HF, J, M, linear)
            names = [bnames[i] for i in cols] + hcols + ["hfrac"]
            grp = [gb[bnames[i]] for i in cols] + ["own"] * len(hcols) + ["calendar"]
            if linear:
                names += [f"hfrac*{n}" for n in names[:-1]]
                grp += grp[:-1]
            y = Z[M - 1, J]
            model = make_model(spec.model, h, **spec.kw)
            fit_stacked(model, D, y, M / H_STACK, J, M, names, grp, h, partial, None if retune else params)
            if retune:
                params, tuned_at = model.params_, k
            fitted_at, n_train = k, len(origins)
        row = stacked_matrix(Xb[:, cols], HF, np.array([k]), np.array([h]), linear)
        change = float(model.predict(row)[0])
        rows.append({"date": idx[k], "forecast": piv[k] + change, "rw": piv[k], "actual": tgt.iloc[k],
                     "n_train": n_train, "n_features": len(cols), "refit": fitted_at == k})
    out = pd.DataFrame(rows)
    return out.set_index("date") if not out.empty else out
