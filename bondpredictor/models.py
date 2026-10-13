"""Forecasting estimators.

All estimators standardise predictors with training moments and keep an
unpenalised intercept.  Tunable models (ridge, lasso and the machine-learning
models in :mod:`bondpredictor.ml`) choose hyperparameters by expanding-window
cross-validation over the second half of the training window.  Consecutive
validation blocks are each fitted only on earlier rows, with a gap of
``horizon`` months so overlapping returns cannot leak across.  A single
holdout block proved too noisy; the chosen penalty jumped across the whole
grid from one month to the next.
"""

from __future__ import annotations

import numpy as np


def cv_folds(n: int, horizon: int, n_folds: int = 5, cv_frac: float = 0.5, min_fit: int = 24):
    """Expanding-window CV splits: yields (end_fit, val_start, val_end).

    Validation blocks tile the last ``cv_frac`` of the sample; each is fitted
    on rows before ``val_start - (horizon - 1)``.
    """
    edges = np.linspace(int(n * (1 - cv_frac)), n, n_folds + 1).astype(int)
    gap = max(horizon - 1, 0)
    for v0, v1 in zip(edges[:-1], edges[1:]):
        if v1 > v0 and v0 - gap >= min_fit:
            yield v0 - gap, v0, v1


class Estimator:
    name = "estimator"
    tune_key: str | None = None  # constructor argument that fixes the tuned hyperparameter
    tuned_ = None

    def __init__(self, horizon: int = 12):
        self.horizon = horizon

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Estimator":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def coefficients(self) -> np.ndarray | None:
        """Slopes on standardised predictors, where meaningful."""
        return None


class _Standardized(Estimator):
    def _prep(self, X, y):
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        self.ybar = y.mean()
        return (X - self.mu) / self.sd, y - self.ybar

    def _z(self, X):
        return (X - self.mu) / self.sd

    def predict(self, X):
        return self.ybar + self._z(X) @ self.beta

    def coefficients(self):
        return self.beta


class HistoricalMean(Estimator):
    name = "mean"

    def fit(self, X, y):
        self.mu = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu)


class OLS(_Standardized):
    name = "ols"

    def fit(self, X, y):
        Z, yc = self._prep(X, y)
        self.beta, *_ = np.linalg.lstsq(Z, yc, rcond=None)
        return self


def _ridge(Z, yc, alpha):
    n, p = Z.shape
    return np.linalg.solve(Z.T @ Z / n + alpha * np.eye(p), Z.T @ yc / n)


def _lasso(Z, yc, alpha, beta0=None, max_iter=1000, tol=1e-6):
    """Coordinate descent for (1/2n)||y - Zb||^2 + alpha*||b||_1 on the Gram matrix."""
    n, p = Z.shape
    G = Z.T @ Z / n
    c = Z.T @ yc / n
    diag = np.diag(G).copy()
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    for _ in range(max_iter):
        delta = 0.0
        for j in range(p):
            if diag[j] <= 0:
                continue
            old = beta[j]
            rho = c[j] - G[j] @ beta + diag[j] * old
            new = np.sign(rho) * max(abs(rho) - alpha, 0.0) / diag[j]
            if new != old:
                beta[j] = new
                delta = max(delta, abs(new - old))
        if delta < tol * max(1.0, np.abs(beta).max()):
            break
    return beta


class _Penalized(_Standardized):
    alphas: np.ndarray
    tune_key = "alpha"

    def __init__(self, horizon=12, alpha: float | None = None, n_folds: int = 5, cv_frac: float = 0.5, alphas=None):
        super().__init__(horizon)
        self.alpha = alpha
        self.n_folds = n_folds
        self.cv_frac = cv_frac
        if alphas is not None:
            self.alphas = np.asarray(alphas, dtype=float)

    def _solve(self, Z, yc, alpha, warm=None):
        raise NotImplementedError

    def _cv_alpha(self, Z, y):
        n, p = Z.shape
        grid = np.sort(self.alphas)[::-1]  # strongest penalty first (warm starts)
        sse = np.zeros(len(grid))
        used = 0
        for end_fit, v0, v1 in cv_folds(n, self.horizon, self.n_folds, self.cv_frac, 3 * p + 12):
            Zf, yf = Z[:end_fit], y[:end_fit]
            mu = yf.mean()
            warm = None
            for i, a in enumerate(grid):
                warm = self._solve(Zf, yf - mu, a, warm)
                sse[i] += np.sum((y[v0:v1] - mu - Z[v0:v1] @ warm) ** 2)
            used += 1
        return float(grid[np.argmin(sse)]) if used else float(np.median(grid))

    def fit(self, X, y):
        Z, yc = self._prep(X, y)
        self.alpha_ = self.alpha if self.alpha is not None else self._cv_alpha(Z, y)
        self.tuned_ = self.alpha_
        self.beta = self._solve(Z, yc, self.alpha_)
        return self


class Ridge(_Penalized):
    name = "ridge"
    alphas = np.logspace(-3, 2, 16)

    def _solve(self, Z, yc, alpha, warm=None):
        return _ridge(Z, yc, alpha)


class Lasso(_Penalized):
    name = "lasso"
    alphas = np.logspace(-4, -1, 13)

    def _solve(self, Z, yc, alpha, warm=None):
        # Penalty scaled by the target's dispersion so the grid is unit-free.
        scale = max(np.std(yc), 1e-12)
        return _lasso(Z, yc, alpha * scale, beta0=warm)


class PCR(_Standardized):
    """Regression on the first ``k`` principal components of the predictors."""

    name = "pcr"

    def __init__(self, horizon=12, k: int = 3):
        super().__init__(horizon)
        self.k = k

    def fit(self, X, y):
        Z, yc = self._prep(X, y)
        k = min(self.k, Z.shape[1])
        _, _, vt = np.linalg.svd(Z, full_matrices=False)
        V = vt[:k].T
        F = Z @ V
        g, *_ = np.linalg.lstsq(F, yc, rcond=None)
        self.beta = V @ g
        return self


class Combination(_Standardized):
    """Equal-weighted average of univariate OLS forecasts.

    A simple, robust benchmark that tends to hold up out of sample
    (Rapach–Strauss–Zhou 2010).
    """

    name = "combo"

    def fit(self, X, y):
        Z, yc = self._prep(X, y)
        var = (Z**2).mean(axis=0)
        slopes = np.where(var > 0, (Z * yc[:, None]).mean(axis=0) / np.where(var > 0, var, 1), 0.0)
        self.beta = slopes / Z.shape[1]
        return self


ESTIMATORS = {
    "mean": HistoricalMean,
    "ols": OLS,
    "ridge": Ridge,
    "lasso": Lasso,
    "pcr": PCR,
    "combo": Combination,
}

ML_ESTIMATORS = ("rf", "gbm", "nn")


def make_estimator(name: str, horizon: int = 12, **kw) -> Estimator:
    if name in ML_ESTIMATORS:
        from . import ml  # optional dependency: scikit-learn

        return ml.ML_MODELS[name](horizon=horizon, **kw)
    return ESTIMATORS[name](horizon=horizon, **kw)
