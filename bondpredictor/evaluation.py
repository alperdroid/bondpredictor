"""Out-of-sample evaluation: R²_OS, Clark–West, and a duration-timing strategy."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .inference import newey_west


def _norm_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def oos_r2(actual, forecast, benchmark) -> float:
    """Campbell–Thompson out-of-sample R² against the prevailing mean."""
    a, f, b = (np.asarray(v, dtype=float) for v in (actual, forecast, benchmark))
    ok = np.isfinite(a) & np.isfinite(f) & np.isfinite(b)
    return 1.0 - np.sum((a[ok] - f[ok]) ** 2) / np.sum((a[ok] - b[ok]) ** 2)


def clark_west(actual, forecast, benchmark, horizon: int) -> tuple[float, float]:
    """Clark–West (2007) MSPE-adjusted test that the model beats the nested benchmark.

    Returns (t-stat, one-sided p-value) with Newey–West errors (h-1 lags).
    """
    a, f, b = (np.asarray(v, dtype=float) for v in (actual, forecast, benchmark))
    ok = np.isfinite(a) & np.isfinite(f) & np.isfinite(b)
    a, f, b = a[ok], f[ok], b[ok]
    adj = (a - b) ** 2 - ((a - f) ** 2 - (b - f) ** 2)
    if len(adj) < 10:
        return float("nan"), float("nan")
    X = np.ones((len(adj), 1))
    u = adj - adj.mean()
    se = math.sqrt(newey_west(X, u, max(horizon - 1, 0))[0, 0])
    t = adj.mean() / se if se > 0 else float("nan")
    return float(t), _norm_sf(t)


def summarize(result: pd.DataFrame, horizon: int, start=None, end=None) -> dict:
    r = result.dropna(subset=["actual"])
    if start is not None:
        r = r.loc[pd.Period(start, "M") :]
    if end is not None:
        r = r.loc[: pd.Period(end, "M")]
    if len(r) < 12:
        return {"N": len(r)}
    cw_t, cw_p = clark_west(r["actual"], r["forecast"], r["benchmark"], horizon)
    return {
        "N": len(r),
        "start": str(r.index.min()),
        "end": str(r.index.max()),
        "R2_OS": oos_r2(r["actual"], r["forecast"], r["benchmark"]),
        "CW_t": cw_t,
        "CW_p": cw_p,
        "corr": float(np.corrcoef(r["forecast"], r["actual"])[0, 1]),
    }


def timing_strategy(
    forecast: pd.Series,
    monthly_rx: pd.Series,
    horizon: int,
    gamma: float = 5.0,
    var_window: int = 60,
    bounds: tuple[float, float] = (-1.0, 2.0),
) -> pd.DataFrame:
    """Mean–variance duration timing, rebalanced monthly.

    The weight in the bond at month ``t`` is ``(forecast/h) / (gamma *
    var)``, clipped to ``bounds``.  ``var`` is the rolling variance of
    realised monthly excess returns.  The position earns ``monthly_rx[t]``
    (the return from ``t`` to ``t+1``).
    """
    rx = monthly_rx.reindex(forecast.index)
    var = monthly_rx.shift(1).rolling(var_window, min_periods=24).var().reindex(forecast.index)
    w = ((forecast / horizon) / (gamma * var)).clip(*bounds)
    return pd.DataFrame({"weight": w, "ret": w * rx}).dropna()


def strategy_stats(returns: pd.Series, gamma: float = 5.0) -> dict:
    r = returns.dropna()
    if len(r) < 12:
        return {}
    mu, sd = r.mean() * 12, r.std() * math.sqrt(12)
    return {
        "mean": mu,
        "vol": sd,
        "sharpe": mu / sd if sd > 0 else float("nan"),
        "cer": mu - 0.5 * gamma * sd**2,
    }


def evaluate_strategy(result: pd.DataFrame, monthly_rx: pd.Series, horizon: int, gamma: float = 5.0,
                      cost_bp: float = 0.0) -> dict:
    """Model vs prevailing-mean vs buy-and-hold, on the months with realised returns.

    ``cost_bp`` is charged per unit of turnover in the bond weight (|Δw|),
    in basis points of notional, to both timing strategies.
    """
    model = timing_strategy(result["forecast"], monthly_rx, horizon, gamma)
    bench = timing_strategy(result["benchmark"], monthly_rx, horizon, gamma)
    common = model.index.intersection(bench.index)
    m_ret, b_ret = model.loc[common, "ret"], bench.loc[common, "ret"]
    if cost_bp:
        m_ret = m_ret - cost_bp * 1e-4 * model.loc[common, "weight"].diff().abs().fillna(0.0)
        b_ret = b_ret - cost_bp * 1e-4 * bench.loc[common, "weight"].diff().abs().fillna(0.0)
    m, b = strategy_stats(m_ret, gamma), strategy_stats(b_ret, gamma)
    hold = strategy_stats(monthly_rx.reindex(common), gamma)
    if not m or not b:
        return {}
    return {
        "sharpe_model": m["sharpe"],
        "sharpe_mean": b["sharpe"],
        "sharpe_hold": hold.get("sharpe", float("nan")),
        "cer_gain_bp": (m["cer"] - b["cer"]) * 1e4,
        "avg_weight": float(model.loc[common, "weight"].mean()),
        "turnover": float(model.loc[common, "weight"].diff().abs().mean() * 12),
    }
