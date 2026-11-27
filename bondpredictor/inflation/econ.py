"""Shared estimator interface and the regularised, factor and tree models.

Every model forecasts the change in y/y inflation over the horizon (see
``features.py``) from a matrix that may contain missing values.  The base
class standardises predictors with training moments, fills missing values
with the training mean (zero after standardising) unless the model handles
them natively, and tunes hyperparameters on a small grid by expanding-window
cross-validation with a horizon-length gap (``bondpredictor.models.cv_folds``).

Models here follow the macro-forecasting ML literature:

* ridge, adaptive lasso (Zou 2006) and elastic net, the shrinkage
  benchmarks of Medeiros, Vasconcelos, Veiga and Zilberman (2021, JBES);
* the Stock-Watson (2002) diffusion index: principal components of the
  non-inflation predictors plus a few own-inflation terms;
* the three-pass regression filter (Kelly-Pruitt 2015), a target-aware
  factor model;
* complete subset regression (Elliott, Gargano and Timmermann 2013);
* kernel ridge regression, the non-linear workhorse of Goulet Coulombe,
  Leroux, Stevanovic and Surprenant (2022, JAE);
* random forest, the best inflation model in Medeiros et al. (2021), and
  gradient-boosted trees (LightGBM).
"""

from __future__ import annotations

import warnings

import numpy as np

from ..models import _lasso, _ridge, cv_folds

OWN_SMALL = ("pi", "sa6", "core", "mech", "drop", "tax")


class InflationModel:
    name = "model"
    grid: list[dict] = [{}]
    nan_ok = False  # True: missing values are passed through (trees)
    clip = False  # True: clamp inputs to the training range before predicting
    n_folds = 3

    def __init__(self, horizon: int = 12, params: dict | None = None, seed: int = 0, **kw):
        self.horizon = horizon
        self.params = params
        self.seed = seed
        self.kw = kw
        self.tuned_ = None

    # ---- preprocessing
    def _prep_fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # columns missing in every training row
            self.mu = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
        self.sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
        self.mu = np.where(np.isfinite(self.mu), self.mu, 0.0)
        Z = (X - self.mu) / self.sd
        self.zmin, self.zmax = np.nanmin(Z, axis=0), np.nanmax(Z, axis=0)
        if not self.nan_ok:
            Z = np.where(np.isfinite(Z), Z, 0.0)
        self.ybar = float(np.mean(y))
        ysd = float(np.std(y))
        self.ysd = ysd if ysd > 1e-12 else 1.0
        return Z, (y - self.ybar) / self.ysd

    def _prep(self, X):
        Z = (X - self.mu) / self.sd
        if self.clip:
            Z = np.clip(Z, self.zmin, self.zmax)
        if not self.nan_ok:
            Z = np.where(np.isfinite(Z), Z, 0.0)
        return Z

    # ---- interface
    def fit(self, X, y, names: list[str], groups: list[str]):
        self.names, self.groups = list(names), list(groups)
        Z, yz = self._prep_fit(np.asarray(X, float), np.asarray(y, float))
        self.params_ = dict(self.params) if self.params is not None else self._cv(Z, yz)
        self.tuned_ = self.params_
        self._fit(Z, yz, self.params_)
        return self

    def predict(self, X):
        return self.ybar + self.ysd * self._predict(self._prep(np.asarray(X, float)))

    def _cv(self, Z, y) -> dict:
        if len(self.grid) == 1:
            return dict(self.grid[0])
        sse = np.zeros(len(self.grid))
        used = 0
        for end_fit, v0, v1 in cv_folds(len(y), self.horizon, self.n_folds, 0.5, min_fit=60):
            for i, p in enumerate(self.grid):
                self._fit(Z[:end_fit], y[:end_fit], p)
                sse[i] += np.sum((y[v0:v1] - self._predict(Z[v0:v1])) ** 2)
            used += 1
        return dict(self.grid[int(np.argmin(sse))]) if used else dict(self.grid[0])

    def _fit(self, Z, y, params):
        raise NotImplementedError

    def _predict(self, Z):
        raise NotImplementedError

    def _cols(self, names) -> list[int]:
        return [i for i, n in enumerate(self.names) if n in names]


# ------------------------------------------------------------------ linear

class OLS(InflationModel):
    name = "ols"

    def _fit(self, Z, y, params):
        self.beta, *_ = np.linalg.lstsq(Z, y, rcond=None)

    def _predict(self, Z):
        return Z @ self.beta


class Ridge(InflationModel):
    name = "ridge"
    grid = [{"alpha": a} for a in np.logspace(-2, 2.5, 10)]
    n_folds = 5

    def _fit(self, Z, y, params):
        self.beta = _ridge(Z, y, params["alpha"])

    def _predict(self, Z):
        return Z @ self.beta


class AdaptiveLasso(InflationModel):
    """Lasso with penalty weights 1/|ridge coefficient| (Zou 2006); the ridge penalty is tuned first."""

    name = "adalasso"
    grid = [{"alpha": a} for a in np.logspace(-3.5, -0.5, 10)]
    n_folds = 5

    def _weights(self, Z, y):
        r = Ridge(self.horizon)
        r.n_folds = 3
        alpha = r._cv(Z, y)["alpha"]
        b = np.abs(_ridge(Z, y, alpha))
        return np.maximum(b, 1e-6)

    def fit(self, X, y, names, groups):
        self.names, self.groups = list(names), list(groups)
        Z, yz = self._prep_fit(np.asarray(X, float), np.asarray(y, float))
        self.w = self._weights(Z, yz)
        self.params_ = dict(self.params) if self.params is not None else self._cv(Z, yz)
        self.tuned_ = self.params_
        self._fit(Z, yz, self.params_)
        return self

    def _fit(self, Z, y, params):
        g = _lasso(Z * self.w, y, params["alpha"])
        self.beta = g * self.w

    def _predict(self, Z):
        return Z @ self.beta


class ElasticNet(InflationModel):
    name = "enet"
    grid = [{"alpha": a, "l1_ratio": r} for r in (0.2, 0.5, 0.8) for a in (0.003, 0.01, 0.03, 0.1, 0.3)]

    def _fit(self, Z, y, params):
        from sklearn.linear_model import ElasticNet as EN

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m = EN(max_iter=5000, **params).fit(Z, y)

    def _predict(self, Z):
        return self.m.predict(Z)


# ----------------------------------------------------------------- factors

def _pcs(Z, k):
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    return vt[:k].T


class DiffusionIndex(InflationModel):
    """OLS on a few own-inflation terms plus the first ``k`` principal components of everything else."""

    name = "di"
    grid = [{"k": k} for k in (2, 4, 8)]

    def _fit(self, Z, y, params):
        own = self._cols(OWN_SMALL)
        rest = [i for i in range(Z.shape[1]) if i not in own]
        self.own_, self.rest_ = own, rest
        k = min(params["k"], len(rest))
        self.V = _pcs(Z[:, rest], k) if k else np.zeros((len(rest), 0))
        R = np.column_stack([Z[:, own], Z[:, rest] @ self.V])
        self.beta, *_ = np.linalg.lstsq(R, y, rcond=None)

    def _predict(self, Z):
        return np.column_stack([Z[:, self.own_], Z[:, self.rest_] @ self.V]) @ self.beta


class ThreePassFilter(InflationModel):
    """Three-pass regression filter with ``L`` automatic proxies (Kelly-Pruitt 2015)."""

    name = "3prf"
    grid = [{"L": L} for L in (1, 2, 3)]

    @staticmethod
    def _passes(Z, y, proxies):
        # Pass 1: each predictor on the proxies (time series) -> loadings.
        P = np.column_stack([np.ones(len(Z)), proxies])
        phi = np.linalg.lstsq(P, Z, rcond=None)[0][1:].T  # (N, L)
        # Pass 2: each period's cross-section on the loadings -> factors.
        Q = np.column_stack([np.ones(len(phi)), phi])
        F = np.linalg.lstsq(Q, Z.T, rcond=None)[0][1:].T  # (T, L)
        return phi, F

    def _fit(self, Z, y, params):
        proxies = y[:, None]
        for _ in range(params["L"] - 1):
            phi, F = self._passes(Z, y, proxies)
            A = np.column_stack([np.ones(len(F)), F])
            resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
            proxies = np.column_stack([proxies, resid])
        self.phi, F = self._passes(Z, y, proxies)
        A = np.column_stack([np.ones(len(F)), F])
        self.beta = np.linalg.lstsq(A, y, rcond=None)[0]

    def _predict(self, Z):
        Q = np.column_stack([np.ones(len(self.phi)), self.phi])
        F = np.linalg.lstsq(Q, Z.T, rcond=None)[0][1:].T
        return np.column_stack([np.ones(len(F)), F]) @ self.beta


class SubsetRegression(InflationModel):
    """Complete subset regression: the average forecast of OLS models on random ``k``-subsets.

    Every subset also keeps the latest inflation rate and the mechanical
    base-effect term, so each member is a sensible inflation model.
    """

    name = "csr"
    grid = [{"k": k} for k in (2, 4, 8)]
    n_subsets = 500

    def _fit(self, Z, y, params):
        rng = np.random.default_rng(self.seed)
        base = self._cols(("pi", "mech"))
        pool = [i for i in range(Z.shape[1]) if i not in base]
        k = min(params["k"], len(pool))
        self.members = []
        for _ in range(self.n_subsets):
            cols = base + list(rng.choice(pool, size=k, replace=False)) if k else base
            b, *_ = np.linalg.lstsq(Z[:, cols], y, rcond=None)
            self.members.append((cols, b))

    def _predict(self, Z):
        return np.mean([Z[:, c] @ b for c, b in self.members], axis=0)


# --------------------------------------------------------------- non-linear

class KernelRidge(InflationModel):
    name = "krr"
    clip = True
    grid = [{"alpha": a, "gamma": g} for a in (0.3, 1.0, 3.0) for g in (0.1, 0.3, 1.0)]

    def _fit(self, Z, y, params):
        from sklearn.kernel_ridge import KernelRidge as KR

        # gamma is per standardised dimension: exp(-gamma * ||dz||^2 / p)
        self.m = KR(kernel="rbf", alpha=params["alpha"], gamma=params["gamma"] / Z.shape[1]).fit(Z, y)

    def _predict(self, Z):
        return self.m.predict(Z)


class RandomForest(InflationModel):
    """500 trees, a third of the predictors per split, leaves of at least 5 (Medeiros et al. 2021)."""

    name = "rf"
    nan_ok = True
    grid = [{"min_samples_leaf": 5}]

    def _fit(self, Z, y, params):
        from sklearn.ensemble import RandomForestRegressor

        self.m = RandomForestRegressor(n_estimators=self.kw.get("n_estimators", 500), max_features=1 / 3,
                                       random_state=self.seed, n_jobs=1, **params).fit(Z, y)

    def _predict(self, Z):
        return self.m.predict(Z)


class Boosting(InflationModel):
    name = "lgbm"
    nan_ok = True
    grid = [{"n_estimators": n, "num_leaves": L} for n in (300, 800) for L in (4, 8)]

    def _fit(self, Z, y, params):
        import lightgbm as lgb

        self.m = lgb.LGBMRegressor(learning_rate=0.02, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.5, reg_lambda=1.0, random_state=self.seed, n_jobs=1,
                                   verbose=-1, **params).fit(Z, y)

    def _predict(self, Z):
        return self.m.predict(Z)
