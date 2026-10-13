"""Predictors that need no estimation on returns.

Every function here is one-sided: the value at month ``t`` uses only data
dated ``t`` or earlier, after publication lags.  Their output can therefore
be computed once for the whole sample without look-ahead.  Predictors that
are fitted to returns (Cochrane–Piazzesi, cycle, macro PCs) live in
:mod:`bondpredictor.learned` and are re-estimated inside the backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trend_inflation(inflation: pd.Series, v: float = 0.987, window: int = 120, min_periods: int = 36) -> pd.Series:
    """Discounted moving average of past inflation (Cieslak–Povala 2015).

    tau_t = sum_i v^i pi_{t-i} / sum_i v^i over the last ``window`` months.
    It proxies slow-moving long-run inflation expectations.
    """
    x = inflation.to_numpy(dtype=float)
    w = v ** np.arange(window)
    out = np.full(len(x), np.nan)
    for t in range(len(x)):
        lo = max(0, t - window + 1)
        seg = x[lo : t + 1][::-1]
        ok = np.isfinite(seg)
        if ok.sum() >= min_periods and np.isfinite(x[t]):
            ww = w[: len(seg)][ok]
            out[t] = np.dot(ww, seg[ok]) / ww.sum()
    return pd.Series(out, index=inflation.index, name="tau")


def one_sided_hp(y: pd.Series, lamb: float, burn_in: int = 0) -> pd.Series:
    """One-sided Hodrick–Prescott trend via the Kalman filter.

    The trend at ``t`` is the filtered estimate of the HP state-space model
    (tau_t = 2 tau_{t-1} - tau_{t-2} + eta, y = tau + eps,
    var(eps)/var(eta) = lambda) using data up to ``t`` only.
    """
    vals = y.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    valid = np.flatnonzero(np.isfinite(vals))
    if len(valid) == 0:
        return pd.Series(out, index=y.index)
    F = np.array([[2.0, -1.0], [1.0, 0.0]])
    Q = np.array([[1.0 / lamb, 0.0], [0.0, 0.0]])
    first = valid[0]
    x = np.array([vals[first], vals[first]])
    P = np.eye(2) * 1e4 * max(1.0, np.nanvar(vals))
    for t in range(first, len(vals)):
        if t > first:
            x = F @ x
            P = F @ P @ F.T + Q
        if np.isfinite(vals[t]):
            S = P[0, 0] + 1.0
            K = P[:, 0] / S
            x = x + K * (vals[t] - x[0])
            P = P - np.outer(K, P[0, :])
        out[t] = x[0]
    if burn_in:
        out[first : first + burn_in] = np.nan
    return pd.Series(out, index=y.index)


def output_gap(real_gdp: pd.Series, lamb: float = 1600.0, burn_in: int = 12) -> pd.Series:
    """Real-time output gap in percent: log GDP minus its one-sided HP trend.

    Apply to the raw quarterly series before moving it to monthly.
    """
    logy = 100.0 * np.log(real_gdp.astype(float))
    return (logy - one_sided_hp(logy, lamb, burn_in=burn_in)).rename("output_gap")


def inflation_uncertainty(inflation: pd.Series, window: int = 36, min_obs: int = 60) -> pd.Series:
    """Volatility of real-time inflation surprises.

    Each month an AR(1) for year-on-year inflation is re-estimated on data
    up to the previous month.  Uncertainty is the root mean squared
    one-step forecast error over the last ``window`` months.  It proxies
    the survey-dispersion measures in Wright (2011), which are paid
    (Consensus) or US-only (SPF).
    """
    x = inflation.to_numpy(dtype=float)
    n = len(x)
    err = np.full(n, np.nan)
    sx = sy = sxx = sxy = 0.0
    cnt = 0
    for t in range(1, n):
        prev, cur = x[t - 1], x[t]
        if cnt >= min_obs and np.isfinite(prev) and np.isfinite(cur):
            den = cnt * sxx - sx * sx
            if den > 0:
                b = (cnt * sxy - sx * sy) / den
                a = (sy - b * sx) / cnt
                err[t] = cur - (a + b * prev)
        if np.isfinite(prev) and np.isfinite(cur):
            sx += prev
            sy += cur
            sxx += prev * prev
            sxy += prev * cur
            cnt += 1
    e2 = pd.Series(err**2, index=inflation.index)
    return np.sqrt(e2.rolling(window, min_periods=window // 2).mean()).rename("infl_unc")


def realized_volatility(daily_yield: pd.Series, months: int = 1) -> pd.Series:
    """Annualised std of daily yield changes (decimal) over the last ``months`` months."""
    d = daily_yield.dropna().sort_index()
    if isinstance(d.index, pd.PeriodIndex):
        d.index = d.index.to_timestamp()
    chg = d.diff().dropna()
    per = chg.index.to_period("M")
    ss = (chg**2).groupby(per).sum()
    cnt = chg.groupby(per).count()
    full = pd.period_range(ss.index.min(), ss.index.max(), freq="M")
    ss, cnt = ss.reindex(full, fill_value=0.0), cnt.reindex(full, fill_value=0)
    var = ss.rolling(months).sum() / cnt.rolling(months).sum().replace(0, np.nan)
    return np.sqrt(var * 252.0).rename("rvol")


def realized_volatility_monthly(yield_m: pd.Series, window: int = 12) -> pd.Series:
    """Fallback when only monthly yields exist: rolling std of monthly changes."""
    return (yield_m.diff().rolling(window, min_periods=window).std() * np.sqrt(12.0)).rename("rvol")


def regime_dummy(index: pd.PeriodIndex, start: str, end: str | None) -> pd.Series:
    lo = pd.Period(start, "M")
    hi = pd.Period(end, "M") if end else index.max()
    return pd.Series(((index >= lo) & (index <= hi)).astype(float), index=index)
