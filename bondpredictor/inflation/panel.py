"""Pooled ("cross-learning") models across the four economies.

Four economies share one model on the predictors they have in common
(``features.COMMON`` plus own-inflation, oil and global terms).  Each
economy's predictors are standardised with its own training moments, so the
model learns from relative movements, and country dummies enter linearly.
The target stays in percentage points.  Pooling gives the data-hungry
models four times the observations, including the high-inflation episodes
of every economy.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .backtest import Spec, make_model
from .features import common_columns


def _country_design(frame, groups, code, h):
    cmap = common_columns(code, frame, h)
    X = frame[list(cmap.values())].copy()
    X.columns = list(cmap.keys())
    g = {k: groups[v] for k, v in cmap.items() if v in groups}
    return X, g, frame["pi"], frame[f"tgt{h}"]


def run_panel_backtest(frames: dict, h: int, spec: Spec, start: str = "2000-01", end: str | None = None,
                       min_train: int = 120, min_hist: int = 60) -> dict[str, pd.DataFrame]:
    """``frames``: code -> (frame, groups).  Returns one result frame per economy.

    As in the single-economy models, an economy needs ``min_train`` realised
    targets to be forecast and a predictor enters an economy's block only
    with ``min_hist`` training observations there.  Shorter histories would
    give noisy standardisation moments and extreme z-scores.
    """
    codes = list(frames)
    D = {c: _country_design(*frames[c], c, h) for c in codes}
    names = sorted(set().union(*[set(D[c][0].columns) for c in codes]))
    group_of = {}
    for c in codes:
        group_of.update(D[c][1])
    dummies = [f"cty_{c}" for c in codes]
    group_of.update({d: "fixed" for d in dummies})
    Xc = {c: D[c][0].reindex(columns=names) for c in codes}
    Y = {c: D[c][3] - D[c][2] for c in codes}
    dates = pd.period_range(pd.Period(start, "M"), pd.Period(end, "M") if end else max(D[c][2].index.max() for c in codes), freq="M")
    t0 = pd.Period(spec.train_start, "M")
    rows = {c: [] for c in codes}
    model, frozen, fitted_at = None, None, None
    for d in dates:
        due = model is None or (d - fitted_at).n >= spec.refit_every
        if due:
            blocks, moments = [], {}
            for i, c in enumerate(codes):
                X, y = Xc[c], Y[c]
                lo = t0 if spec.window is None else max(t0, d - h - spec.window + 1)
                tr = (X.index >= lo) & (X.index <= d - h) & y.notna().to_numpy() & D[c][2].notna().to_numpy()
                if tr.sum() < min_train:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mu, sd = X[tr].mean(), X[tr].std()
                sd = sd.where(sd > 1e-12, 1.0)
                valid = X[tr].notna().sum() >= min_hist
                moments[c] = (mu, sd, valid)
                Z = (X[tr] - mu) / sd
                Z[list(valid.index[~valid])] = np.nan
                Z[dummies] = 0.0
                Z[f"cty_{c}"] = 1.0
                Z["_y"] = y[tr]
                blocks.append(Z)
            if not blocks:
                continue
            stacked = pd.concat(blocks).sort_index(kind="stable")
            if len(stacked) < min_train:
                continue
            hist = stacked[names].notna().sum()
            feats = [n for n in names if hist[n] >= min_hist] + dummies
            model = make_model(spec.model, h * len(blocks), **spec.kw)
            model.fit(stacked[feats].to_numpy(float), stacked["_y"].to_numpy(float), feats, [group_of[f] for f in feats])
            frozen, fitted_at = (feats, moments), d
        feats, moments = frozen
        for c in codes:
            if c not in moments or d not in Xc[c].index or not np.isfinite(D[c][2].get(d, np.nan)):
                continue
            mu, sd, valid = moments[c]
            z = (Xc[c].loc[[d]] - mu) / sd
            z[list(valid.index[~valid])] = np.nan
            z[dummies] = 0.0
            z[f"cty_{c}"] = 1.0
            change = float(model.predict(z[feats].to_numpy(float))[0])
            pi = float(D[c][2][d])
            row = {"date": d, "forecast": pi + change, "rw": pi, "actual": D[c][3].get(d, np.nan),
                   "n_train": int(len(stacked)) if fitted_at == d else np.nan, "n_features": len(feats),
                   "refit": fitted_at == d}
            if hasattr(model, "predict_quantiles"):
                q = model.predict_quantiles(z[feats].to_numpy(float), (0.1, 0.5, 0.9))[0]
                row.update({"q10": pi + q[0], "q50": pi + q[1], "q90": pi + q[2]})
            rows[c].append(row)
    return {c: pd.DataFrame(r).set_index("date") for c, r in rows.items() if r}


def panel_zoo() -> list[Spec]:
    return [
        Spec("panel:ridge", "ridge", refit_every=12),
        Spec("panel:lgbm", "lgbm", refit_every=12),
        Spec("panel:mrf", "mrf", refit_every=12),
        Spec("panel:bart", "bart", refit_every=12),
        Spec("panel:hnn", "hnn", refit_every=12),
    ]
