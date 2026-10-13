"""Machine-learning forecasters: random forest, gradient boosting, neural net.

These follow the bond-return ML literature, e.g. Bianchi, Büchner and Tamoni
(2021, RFS), who find neural nets on forward rates plus macro data improve
on linear models, and Gu, Kelly and Xiu (2020).

Small-sample safeguards, since there are only ~300–600 overlapping monthly
observations:

* inputs and target are standardised with training moments;
* hyperparameters come from small, conservative grids, tuned by the same
  expanding-window CV with a horizon gap as ridge/lasso (3 folds);
* neural nets are averaged over several random initialisations, which
  reduces seed noise, as in Bianchi et al.;
* inputs are clamped to the range seen in training before predicting
  (``clip=True``).  A ReLU net otherwise extrapolates linearly from
  never-seen inputs; in Germany it forecast +13% to +25% just before the
  2022 sell-off, when negative yields lay outside the training history;
* in the backtest the models are refitted once a year (``refit_every``),
  as in Gu–Kelly–Xiu.  The fitted factors are frozen with them, so there is
  no look-ahead between refits.

Requires scikit-learn (``pip install scikit-learn``).
"""

from __future__ import annotations

import warnings

import numpy as np

from .models import Estimator, cv_folds

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor
except ImportError as err:  # pragma: no cover
    raise ImportError("ML models need scikit-learn: pip install scikit-learn") from err


class _SklearnModel(Estimator):
    grid: list[dict] = [{}]
    default_seeds: tuple[int, ...] = (0,)
    tune_key = "params"

    def __init__(self, horizon: int = 12, params: dict | None = None, seeds=None, n_folds: int = 3,
                 cv_frac: float = 0.5, clip: bool = True):
        super().__init__(horizon)
        self.params = params
        self.clip = clip
        self.seeds = tuple(seeds) if seeds is not None else self.default_seeds
        self.n_folds = n_folds
        self.cv_frac = cv_frac

    def _make(self, params: dict, seed: int):
        raise NotImplementedError

    def _fit_models(self, Z, y, params, seeds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return [self._make(params, s).fit(Z, y) for s in seeds]

    def _cv(self, Z, y) -> dict:
        if len(self.grid) == 1:
            return dict(self.grid[0])
        sse = np.zeros(len(self.grid))
        used = 0
        for end_fit, v0, v1 in cv_folds(len(y), self.horizon, self.n_folds, self.cv_frac, min_fit=60):
            for i, params in enumerate(self.grid):
                models = self._fit_models(Z[:end_fit], y[:end_fit], params, self.seeds[:1])
                sse[i] += np.sum((y[v0:v1] - models[0].predict(Z[v0:v1])) ** 2)
            used += 1
        return dict(self.grid[int(np.argmin(sse))]) if used else dict(self.grid[0])

    def fit(self, X, y):
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        self.ybar, ysd = float(y.mean()), float(y.std())
        self.ysd = ysd if ysd > 0 else 1.0
        Z = (X - self.mu) / self.sd
        self.zmin, self.zmax = Z.min(axis=0), Z.max(axis=0)
        yz = (y - self.ybar) / self.ysd
        self.params_ = dict(self.params) if self.params is not None else self._cv(Z, yz)
        self.tuned_ = self.params_
        self.models = self._fit_models(Z, yz, self.params_, self.seeds)
        return self

    def predict(self, X):
        Z = (X - self.mu) / self.sd
        if self.clip:
            Z = np.clip(Z, self.zmin, self.zmax)
        pred = np.mean([m.predict(Z) for m in self.models], axis=0)
        return self.ybar + self.ysd * pred


class RandomForest(_SklearnModel):
    name = "rf"
    grid = [
        {"max_depth": 3, "min_samples_leaf": 20},
        {"max_depth": 6, "min_samples_leaf": 10},
    ]

    def _make(self, params, seed):
        return RandomForestRegressor(n_estimators=300, max_features=0.33, random_state=seed, n_jobs=1, **params)


class GradientBoosting(_SklearnModel):
    name = "gbm"
    grid = [
        {"max_depth": 1, "n_estimators": 150},
        {"max_depth": 2, "n_estimators": 150},
        {"max_depth": 2, "n_estimators": 400},
    ]

    def _make(self, params, seed):
        return GradientBoostingRegressor(learning_rate=0.05, subsample=0.7, min_samples_leaf=10, random_state=seed, **params)


class NeuralNet(_SklearnModel):
    """Feed-forward net (ReLU, L2 penalty), averaged over 5 initialisations."""

    name = "nn"
    default_seeds = (0, 1, 2, 3, 4)
    grid = [
        {"hidden_layer_sizes": (8,), "alpha": 1.0},
        {"hidden_layer_sizes": (8,), "alpha": 10.0},
        {"hidden_layer_sizes": (16, 8), "alpha": 1.0},
        {"hidden_layer_sizes": (16, 8), "alpha": 10.0},
    ]

    def _make(self, params, seed):
        return MLPRegressor(solver="lbfgs", max_iter=500, random_state=seed, **params)


ML_MODELS = {"rf": RandomForest, "gbm": GradientBoosting, "nn": NeuralNet}
