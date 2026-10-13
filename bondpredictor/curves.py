"""Zero-coupon yield curves.

A curve maps (month, maturity) to a continuously compounded zero yield in
decimals.  Two representations cover all three countries:

* :class:`SvenssonCurve` evaluates published Svensson parameters at any
  maturity: GSW for the US, the Bundesbank curve for Germany.
* :class:`TabulatedCurve` interpolates yields on a maturity grid: Kim–Wright,
  or a curve bootstrapped from MOF par yields.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def svensson(maturities, beta0, beta1, beta2, beta3, tau1, tau2):
    """Svensson (1994) zero yield, in the units of the betas.

    All arguments broadcast.  A missing ``beta3``/``tau2`` (Nelson–Siegel
    days in the GSW file) contributes nothing.
    """
    m = np.asarray(maturities, dtype=float)

    def loading(tau):
        x = m / tau
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            e = np.exp(-x)
            slope = np.where(np.abs(x) > 1e-10, (1.0 - e) / x, 1.0)
        return slope, slope - e

    s1, c1 = loading(np.asarray(tau1, dtype=float))
    beta3 = np.nan_to_num(np.asarray(beta3, dtype=float))
    tau2 = np.asarray(tau2, dtype=float)
    tau2 = np.where(np.isfinite(tau2) & (tau2 > 0), tau2, 1.0)
    _, c2 = loading(tau2)
    return beta0 + beta1 * s1 + beta2 * c1 + beta3 * c2


class ZeroCurve:
    """Monthly zero curve.  Subclasses implement :meth:`yields`."""

    index: pd.PeriodIndex
    name: str = "curve"

    def yields(self, maturities) -> pd.DataFrame:
        raise NotImplementedError

    def yield_at(self, maturity: float) -> pd.Series:
        return self.yields([maturity]).iloc[:, 0]

    def log_prices(self, maturities) -> pd.DataFrame:
        y = self.yields(maturities)
        return -y * np.asarray(maturities, dtype=float)[None, :]

    def forwards(self, maturities, length: float = 1.0) -> pd.DataFrame:
        """Forward rate from ``n - length`` to ``n`` for each maturity ``n``.

        With the default ``length = 1`` and integer ``n`` these are the
        one-year forwards used by Fama–Bliss and Cochrane–Piazzesi.
        """
        n = np.asarray(maturities, dtype=float)
        p_long = self.log_prices(n).to_numpy()
        short = n - length
        p_short = np.zeros_like(p_long)
        pos = short > 1e-12
        if pos.any():
            p_short[:, pos] = self.log_prices(short[pos]).to_numpy()
        f = (p_short - p_long) / length
        return pd.DataFrame(f, index=self.index, columns=list(maturities))


class SvenssonCurve(ZeroCurve):
    """Curve from Svensson parameters (betas in percent, taus in years).

    ``max_maturity`` (column or scalar) masks maturities beyond the longest
    bond outstanding on that date; ``min_maturity`` does the same at the
    short end.
    """

    def __init__(self, params: pd.DataFrame, max_maturity=None, min_maturity: float = 0.0, name="svensson"):
        self.params = params
        self.index = params.index
        if max_maturity is None and "max_maturity" in params:
            max_maturity = params["max_maturity"]
        if max_maturity is None:
            mx = np.full(len(params), np.inf)
        elif np.ndim(max_maturity) == 0:
            mx = np.full(len(params), float(max_maturity))
        else:
            mx = pd.Series(max_maturity).reindex(params.index).to_numpy(dtype=float)
        self.max_maturity = np.where(np.isfinite(mx), mx, np.inf)[:, None]
        self.min_maturity = min_maturity
        self.name = name

    def yields(self, maturities) -> pd.DataFrame:
        m = np.asarray(maturities, dtype=float)[None, :]
        p = {k: self.params[k].to_numpy(dtype=float)[:, None] if k in self.params else np.full((len(self.params), 1), np.nan)
             for k in ("beta0", "beta1", "beta2", "beta3", "tau1", "tau2")}
        y = svensson(m, p["beta0"], p["beta1"], p["beta2"], p["beta3"], p["tau1"], p["tau2"]) / 100.0
        y = np.where(m <= self.max_maturity + 1e-9, y, np.nan)
        y = np.where(m >= self.min_maturity - 1e-9, y, np.nan)
        return pd.DataFrame(y, index=self.index, columns=list(np.asarray(maturities, dtype=float)))


class TabulatedCurve(ZeroCurve):
    """Curve from zero yields (decimal) on a grid of maturities (columns, years).

    Linear interpolation in maturity.  Below the shortest available maturity
    the yield is held flat if ``extrapolate_short`` is set (only within
    ``short_limit`` years); beyond the longest it is missing.
    """

    def __init__(self, yields: pd.DataFrame, extrapolate_short: bool = False, short_limit: float = 0.5, name="tabulated"):
        cols = np.asarray([float(c) for c in yields.columns])
        order = np.argsort(cols)
        self.grid = cols[order]
        self.values = yields.to_numpy(dtype=float)[:, order]
        self.index = yields.index
        self.extrapolate_short = extrapolate_short
        self.short_limit = short_limit
        self.name = name

    def yields(self, maturities) -> pd.DataFrame:
        m = np.asarray(maturities, dtype=float)
        out = np.full((len(self.index), len(m)), np.nan)
        for i, row in enumerate(self.values):
            ok = np.isfinite(row)
            if ok.sum() == 0:
                continue
            g, v = self.grid[ok], row[ok]
            if ok.sum() == 1:
                vals = np.where(np.isclose(m, g[0]), v[0], np.nan)
            else:
                vals = np.interp(m, g, v, left=np.nan, right=np.nan)
            if self.extrapolate_short:
                low = (m < g[0]) & (m >= g[0] - self.short_limit)
                vals = np.where(low, v[0], vals)
            out[i] = vals
        return pd.DataFrame(out, index=self.index, columns=list(m))


def bootstrap_par_yields(par: pd.DataFrame, coupons_per_year: int = 2, max_maturity: float | None = None) -> pd.DataFrame:
    """Zero curve from par yields by bootstrapping.

    ``par`` holds par yields in percent with columns = maturity in years
    (MOF layout).  Par yields are interpolated linearly onto the coupon grid
    (flat below the first maturity), then discount factors are solved
    sequentially from ``1 = c/k * sum(P_i) + P_n``.  The result is
    continuously compounded zero yields (decimal) on the coupon grid.
    """
    k = coupons_per_year
    cols = np.asarray([float(c) for c in par.columns])
    order = np.argsort(cols)
    cols = cols[order]
    values = par.to_numpy(dtype=float)[:, order] / 100.0
    top = max_maturity or cols.max()
    grid = np.arange(1, int(round(top * k)) + 1) / k
    out = np.full((len(par), len(grid)), np.nan)
    for i, row in enumerate(values):
        ok = np.isfinite(row)
        if ok.sum() < 2:
            continue
        g, v = cols[ok], row[ok]
        usable = grid <= g[-1] + 1e-9
        c = np.interp(grid[usable], g, v)  # flat below g[0]
        disc = np.empty(usable.sum())
        running = 0.0
        for j, cj in enumerate(c):
            disc[j] = (1.0 - cj / k * running) / (1.0 + cj / k)
            running += disc[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i, : usable.sum()] = -np.log(disc) / grid[usable]
    return pd.DataFrame(out, index=par.index, columns=list(grid))


def monthly_params(daily: pd.DataFrame) -> pd.DataFrame:
    """End-of-month rows of a daily parameter/yield frame, on a PeriodIndex."""
    daily = daily.sort_index().dropna(how="all")
    # Whole last row of each month (not column-wise last values, which could
    # mix parameters from different days).
    out = daily.groupby(daily.index.to_period("M")).tail(1)
    out.index = out.index.to_period("M")
    full = pd.period_range(out.index.min(), out.index.max(), freq="M")
    return out.reindex(full)
