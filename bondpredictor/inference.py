"""In-sample predictive regressions with inference suited to overlapping returns.

* Newey–West HAC standard errors.
* Hodrick (1992) "1B" standard errors.  These impose the null of no
  predictability and use one-period returns, which avoids the poor size of
  HAC estimators with heavily overlapping long-horizon returns.
* A small-sample bootstrap under the null in the spirit of Bauer–Hamilton
  (2018).  Predictors follow a VAR(1) fitted to the data; one-period returns
  are unpredictable with residuals drawn jointly with the VAR shocks; h-month
  returns are overlapping sums.  This shows how often persistent predictors
  produce the observed t-stats and R² by chance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def newey_west(X: np.ndarray, u: np.ndarray, lags: int) -> np.ndarray:
    """HAC covariance of OLS coefficients with Bartlett weights."""
    T = len(u)
    Xu = X * u[:, None]
    S = Xu.T @ Xu / T
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = Xu[L:].T @ Xu[:-L] / T
        S += w * (G + G.T)
    Qinv = np.linalg.inv(X.T @ X / T)
    return Qinv @ S @ Qinv / T


def hodrick_1b(X: np.ndarray, u1: np.ndarray, horizon: int) -> np.ndarray:
    """Hodrick (1992) 1B covariance for a regression of h-period returns on X.

    ``u1[t]`` is the demeaned one-period return from ``t`` to ``t+1``
    (the null residual).  ``X`` includes the constant.  Rows must be
    consecutive months.
    """
    T = len(u1)
    Z = np.zeros_like(X)
    for j in range(horizon):
        Z[j:] += X[: T - j] if j else X
    Z[: horizon - 1] = np.nan
    ok = np.isfinite(Z).all(axis=1) & np.isfinite(u1)
    Zu = Z[ok] * u1[ok, None]
    S = Zu.T @ Zu / ok.sum()
    Qinv = np.linalg.inv(X[ok].T @ X[ok] / ok.sum())
    return Qinv @ S @ Qinv / ok.sum()


@dataclass
class RegressionResult:
    table: pd.DataFrame
    r2: float
    adj_r2: float
    nobs: int
    horizon: int

    def __str__(self) -> str:
        head = f"N={self.nobs}  h={self.horizon}m  R2={self.r2:.3f}  adj.R2={self.adj_r2:.3f}"
        return head + "\n" + self.table.to_string(float_format=lambda v: f"{v: .3f}")


def predictive_regression(
    y: pd.Series,
    X: pd.DataFrame,
    horizon: int,
    one_period: pd.Series | None = None,
    nw_lags: int | None = None,
    standardize: bool = True,
) -> RegressionResult:
    """OLS of ``y`` (stored at t, realised at t+h) on predictors dated t.

    Coefficients are per one-standard-deviation move in each predictor when
    ``standardize`` is set.
    """
    df = pd.concat([y.rename("_y"), X], axis=1).dropna()
    yv = df.pop("_y").to_numpy()
    Xv = df.to_numpy(dtype=float)
    if standardize:
        Xv = (Xv - Xv.mean(axis=0)) / Xv.std(axis=0)
    A = np.column_stack([np.ones(len(yv)), Xv])
    beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
    resid = yv - A @ beta
    T, k = A.shape
    r2 = 1.0 - resid.var() / yv.var()
    adj = 1.0 - (1.0 - r2) * (T - 1) / (T - k)
    lags = nw_lags if nw_lags is not None else max(int(np.ceil(1.5 * horizon)), 1)
    se_nw = np.sqrt(np.diag(newey_west(A, resid, lags)))
    names = ["const"] + list(df.columns)
    table = pd.DataFrame({"coef": beta, "t_nw": beta / se_nw}, index=names)

    if one_period is not None:
        # Hodrick needs consecutive months, so work on the full monthly range.
        full = pd.period_range(df.index.min(), df.index.max(), freq="M")
        Xf = X.reindex(full)[list(df.columns)].to_numpy(dtype=float)
        if standardize:
            Xf = (Xf - np.nanmean(Xf, axis=0)) / np.nanstd(Xf, axis=0)
        Af = np.column_stack([np.ones(len(full)), Xf])
        r1 = one_period.reindex(full).to_numpy(dtype=float)
        u1 = r1 - np.nanmean(r1)
        V = hodrick_1b(Af, u1, horizon)
        table["t_hodrick"] = beta / np.sqrt(np.diag(V))
    return RegressionResult(table, float(r2), float(adj), int(T), horizon)


def null_bootstrap(
    X: pd.DataFrame,
    one_period: pd.Series,
    horizon: int,
    actual: RegressionResult,
    n_boot: int = 500,
    seed: int = 0,
) -> pd.Series:
    """Bootstrap p-values of the Newey–West t-stats and R² under no predictability."""
    rng = np.random.default_rng(seed)
    df = pd.concat([one_period.rename("_r"), X], axis=1).dropna()
    r = df.pop("_r").to_numpy()
    x = df.to_numpy(dtype=float)
    T, k = x.shape
    # VAR(1) for predictors.
    A = np.column_stack([np.ones(T - 1), x[:-1]])
    coef, *_ = np.linalg.lstsq(A, x[1:], rcond=None)
    v = x[1:] - A @ coef
    e = r[:-1] - r.mean()  # return t->t+1 paired with predictor shock at t+1
    shocks = np.column_stack([e, v])
    lags = max(int(np.ceil(1.5 * horizon)), 1)
    t_obs = actual.table["t_nw"].to_numpy()[1:]
    exceed = np.zeros(k)
    r2_exceed = 0
    for _ in range(n_boot):
        draw = shocks[rng.integers(0, len(shocks), size=T + 50)]
        xs = np.empty((T + 50, k))
        xs[0] = x[rng.integers(0, T)]
        for t in range(1, T + 50):
            xs[t] = coef[0] + xs[t - 1] @ coef[1:] + draw[t, 1:]
        rs = r.mean() + draw[1:, 0]  # rs[t]: return t -> t+1, paired with the x shock at t+1
        xs, rs = xs[50:], rs[50:]
        rh = np.convolve(rs, np.ones(horizon), mode="valid")[: T - horizon + 1]
        Xs = xs[: len(rh)]
        Xs = (Xs - Xs.mean(axis=0)) / Xs.std(axis=0)
        As = np.column_stack([np.ones(len(rh)), Xs])
        b, *_ = np.linalg.lstsq(As, rh, rcond=None)
        u = rh - As @ b
        se = np.sqrt(np.diag(newey_west(As, u, lags)))
        exceed += np.abs(b[1:] / se[1:]) >= np.abs(t_obs)
        r2_exceed += (1 - u.var() / rh.var()) >= actual.r2
    out = pd.Series(exceed / n_boot, index=list(X.columns), name="p_boot")
    out["R2"] = r2_exceed / n_boot
    return out
