"""Adrian–Crump–Moench (2013) term premium, estimable for any country.

Three linear regressions on monthly data:

1. VAR(1) for K pricing factors (principal components of yields).
2. Monthly excess returns on a cross-section of zero-coupon bonds regressed
   on a constant, lagged factors and the VAR innovations.
3. A cross-sectional regression that recovers the market prices of risk
   (lambda0, lambda1).

Affine recursions then give model-fitted yields, risk-neutral yields
(lambda = 0) and the term premium (their difference).  The NY Fed
publishes these estimates for the US.  This module produces comparable
estimates for Germany and Japan, where no official series exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .curves import ZeroCurve

log = logging.getLogger(__name__)


@dataclass
class ACMResult:
    fitted: pd.DataFrame  # annualised decimal yields, columns = maturity in years
    risk_neutral: pd.DataFrame
    term_premium: pd.DataFrame
    factors: pd.DataFrame
    lambda0: np.ndarray
    lambda1: np.ndarray
    n_factors: int
    fit_rmse_bp: float


def _recursions(n_max, mu, Phi, Sigma, sigma2, delta0, delta1, lambda0, lambda1):
    K = len(mu)
    A = np.zeros(n_max + 1)
    B = np.zeros((n_max + 1, K))
    for n in range(1, n_max + 1):
        b = B[n - 1]
        A[n] = A[n - 1] + b @ (mu - lambda0) + 0.5 * (b @ Sigma @ b + sigma2) - delta0
        B[n] = b @ (Phi - lambda1) - delta1
    return A, B


def _prepare(curve, short_rate, yield_months, return_months, report_months, start, end):
    months = np.arange(1, max(max(yield_months), max(return_months), max(report_months)) + 1)
    Y = curve.yields(months / 12.0)
    Y.columns = months
    if short_rate is not None:
        Y[1] = short_rate.reindex(Y.index)
    if start is not None:
        Y = Y.loc[pd.Period(start, "M") :]
    if end is not None:
        Y = Y.loc[: pd.Period(end, "M")]
    # Use only maturities the curve covers (e.g. Kim–Wright starts at 1y).
    covered = Y.notna().mean() > 0.9
    yield_months = [m for m in yield_months if covered[m]]
    return_months = [m for m in return_months if covered[m] and covered[m - 1]]
    if not covered[1]:
        raise ValueError("ACM needs a one-month rate: pass short_rate or use a curve with a short end")
    needed = sorted({1, *yield_months, *return_months, *(m - 1 for m in return_months), *report_months})
    Y = Y[needed].dropna()
    if len(Y) and not (np.diff(Y.index.asi8) == 1).all():
        breaks = np.flatnonzero(np.diff(Y.index.asi8) != 1)
        blocks = np.split(np.arange(len(Y)), breaks + 1)
        Y = Y.iloc[max(blocks, key=len)]  # longest run of complete months
    return Y, yield_months, return_months


def _estimate(Y, K, yield_months, return_months, report_months):
    ym = Y / 12.0  # monthly log yields
    lp = (-ym * np.asarray(Y.columns, dtype=float)[None, :]).to_numpy()
    col = {m: i for i, m in enumerate(Y.columns)}
    rf = ym[1].to_numpy()

    Yf = ym[list(yield_months)].to_numpy()
    Yc = Yf - Yf.mean(axis=0)
    _, _, vt = np.linalg.svd(Yc, full_matrices=False)
    X = Yc @ vt[:K].T
    T = len(X)

    # 1) VAR(1) for the factors.
    Z = np.column_stack([np.ones(T - 1), X[:-1]])
    coef, *_ = np.linalg.lstsq(Z, X[1:], rcond=None)
    mu, Phi = coef[0], coef[1:].T
    V = X[1:] - Z @ coef
    Sigma = V.T @ V / (T - 1)

    # 2) rx_{t+1}(n-1) = p_{t+1}(n-1) - p_t(n) - r_t on [1, v_{t+1}, X_t].
    ns = np.asarray(return_months)
    rx = np.column_stack([lp[1:, col[n - 1]] - lp[:-1, col[n]] - rf[:-1] for n in ns])
    R = np.column_stack([np.ones(T - 1), V, X[:-1]])
    g, *_ = np.linalg.lstsq(R, rx, rcond=None)
    a, beta, c = g[0], g[1 : 1 + K], g[1 + K :].T
    E = rx - R @ g
    N = len(ns)
    sigma2 = np.trace(E.T @ E) / (N * (T - 1))

    # 3) Prices of risk from the cross-section.
    Bstar = np.stack([np.outer(beta[:, i], beta[:, i]).ravel() for i in range(N)])
    inv = np.linalg.inv(beta @ beta.T)
    lambda0 = inv @ beta @ (a + 0.5 * (Bstar @ Sigma.ravel() + sigma2))
    lambda1 = inv @ beta @ c

    D = np.column_stack([np.ones(T), X])
    d, *_ = np.linalg.lstsq(D, rf, rcond=None)

    n_max = int(max(report_months))
    A, B = _recursions(n_max, mu, Phi, Sigma, sigma2, d[0], d[1:], lambda0, lambda1)
    Arn, Brn = _recursions(n_max, mu, Phi, Sigma, sigma2, d[0], d[1:], np.zeros(K), np.zeros((K, K)))
    explosive = np.abs(np.linalg.eigvals(Phi - lambda1)).max() >= 1.0
    return X, A, B, Arn, Brn, lambda0, lambda1, explosive


def acm(
    curve: ZeroCurve,
    short_rate: pd.Series | None = None,
    n_factors: int = 5,
    yield_months=range(3, 121, 3),
    return_months=range(6, 121, 6),
    report_years=(1, 2, 5, 10),
    start=None,
    end=None,
) -> ACMResult:
    """Estimate the ACM model on a zero curve.

    ``short_rate`` (annual cc decimal) is the one-month rate.  If omitted,
    the curve's one-month yield is used, which needs a curve with a short
    end (GSW, Bundesbank).  If the risk-neutral dynamics come out explosive
    (common when the curve is itself a low-dimensional model fit, e.g.
    Kim–Wright, so extra PCs are noise), the factor count is reduced.
    """
    report_months = [int(round(y * 12)) for y in report_years]
    Y, ym_, rm_ = _prepare(curve, short_rate, yield_months, return_months, report_months, start, end)
    if len(Y) < 60:
        raise ValueError("ACM: fewer than 60 complete months")
    for K in range(n_factors, 1, -1):
        if len(ym_) < K or len(rm_) <= K:
            continue
        X, A, B, Arn, Brn, l0, l1, explosive = _estimate(Y, K, ym_, rm_, report_months)
        if not explosive:
            break
        log.warning("ACM with %d factors has explosive Q-dynamics; trying %d", K, K - 1)
    cols = [float(y) for y in report_years]
    fitted = pd.DataFrame(np.column_stack([-(A[n] + X @ B[n]) / n * 12.0 for n in report_months]), index=Y.index, columns=cols)
    rn = pd.DataFrame(np.column_stack([-(Arn[n] + X @ Brn[n]) / n * 12.0 for n in report_months]), index=Y.index, columns=cols)
    obs = Y[report_months].set_axis(cols, axis=1)
    rmse = float(np.sqrt(np.nanmean((fitted - obs).to_numpy() ** 2)) * 1e4)
    return ACMResult(
        fitted=fitted,
        risk_neutral=rn,
        term_premium=fitted - rn,
        factors=pd.DataFrame(X, index=Y.index, columns=[f"pc{i + 1}" for i in range(X.shape[1])]),
        lambda0=l0,
        lambda1=l1,
        n_factors=X.shape[1],
        fit_rmse_bp=rmse,
    )
