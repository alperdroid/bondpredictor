"""Re-estimation variants for the econometric ML models.

Each variant reruns one model through the same real-time backtest with one
thing changed:

* other random seeds (forests, boosting, MRF, BART, hemisphere net);
* each hyperparameter fixed instead of tuned, or set to an alternative value;
* a rolling 20-year window instead of an expanding one;
* training only from 1985, which drops the Great Inflation.  This asks
  whether the 1970s taught the models anything useful for 2021-22.

The report compares each variant with its baseline on the same months.
"""

from __future__ import annotations

from .backtest import Spec

SEED_MODELS = ("rf", "lgbm", "mrf", "bart", "hnn")
WINDOW_MODELS = ("ridge", "rf", "lgbm", "mrf", "bart", "hnn")


def robustness_specs() -> list[Spec]:
    specs = []
    for m in SEED_MODELS:
        for s in (1, 2):
            specs.append(Spec(f"{m} | seed {s}", m, kw={"seed": s}))
    fixed = {
        "rf": [{"min_samples_leaf": 15}],
        "lgbm": [{"n_estimators": n, "num_leaves": L} for n in (300, 800) for L in (4, 8)],
        "mrf": [{"min_leaf": 20, "lam": 0.1}, {"min_leaf": 40, "lam": 0.1}],
        "krr": [{"alpha": 1.0, "gamma": 0.3}],
        "hnn": [{"width": 16, "depth": 2, "dropout": 0.2, "weight_decay": 1e-3},
                {"width": 64, "depth": 2, "dropout": 0.2, "weight_decay": 1e-3},
                {"width": 32, "depth": 2, "dropout": 0.0, "weight_decay": 1e-3},
                {"width": 32, "depth": 2, "dropout": 0.4, "weight_decay": 1e-2}],
    }
    for m, grid in fixed.items():
        for p in grid:
            label = ",".join(f"{k}={v}" for k, v in p.items())
            specs.append(Spec(f"{m} | fixed {label}", m, kw={"params": p}))
    specs.append(Spec("mrf | no time smoothing", "mrf", kw={"zeta": 0.0}))
    specs.append(Spec("bart | k=3 (more shrinkage)", "bart", kw={"k": 3.0}))
    specs.append(Spec("bart | 100 trees", "bart", kw={"n_trees": 100}))
    for m in WINDOW_MODELS:
        specs.append(Spec(f"{m} | rolling 20y", m, window=240, refit_every=1 if m == "ridge" else 12))
        specs.append(Spec(f"{m} | train from 1985", m, train_start="1985-01", refit_every=1 if m == "ridge" else 12))
    return specs
