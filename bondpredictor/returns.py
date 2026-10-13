"""Bond excess returns.

Zero-coupon bonds (from a curve)
    rx_{t->t+h}(n) = p_{t+h}(n-h) - p_t(n) - h * r_t(h),   p(n) = -n y(n)
    stored at month t.  With h = 1 year and the curve's 1-year yield as the
    funding rate this is the Fama–Bliss / Cochrane–Piazzesi return.

Constant-maturity par bonds (when only a 10-year yield series exists)
    One-month return of a par bond approximated by carry, duration and
    convexity (as in Swinkels 2019), funded at the short rate.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .curves import ZeroCurve
from .timeseries import forward_sum


_TARGET_RE = re.compile(r"(?:^|\.)rx(\d+)_")


def target_horizon(column: str) -> int | None:
    """Holding period in months encoded in a return column name (``US.rx12_n10`` -> 12)."""
    m = _TARGET_RE.search(column)
    return int(m.group(1)) if m else None


def maturity_label(n: float) -> str:
    return f"{n:g}".replace(".", "p")


def rx_column(horizon_months: int, n: float) -> str:
    return f"rx{horizon_months}_n{maturity_label(n)}"


def zero_excess_returns(
    curve: ZeroCurve,
    maturities,
    horizon_months: int,
    funding: pd.Series | None = None,
) -> pd.DataFrame:
    """Log excess returns on zero-coupon bonds over ``horizon_months``.

    ``funding`` is an annualised continuously compounded rate (decimal) for
    the holding period, known at ``t``.  By default the curve's own
    ``h``-maturity yield is used.
    """
    h = horizon_months / 12.0
    mats = np.asarray(maturities, dtype=float)
    y_now = curve.yields(mats).to_numpy()
    y_aged = curve.yields(mats - h).shift(-horizon_months).to_numpy()
    if funding is None:
        r = curve.yield_at(h).to_numpy()
    else:
        r = funding.reindex(curve.index).to_numpy(dtype=float)
    rx = mats[None, :] * y_now - (mats[None, :] - h) * y_aged - h * r[:, None]
    cols = [rx_column(horizon_months, n) for n in mats]
    return pd.DataFrame(rx, index=curve.index, columns=cols)


def par_bond_duration_convexity(y, maturity: float, freq: int = 2):
    """Modified duration and convexity of a par bond with yield ``y`` (decimal).

    Closed forms for a bond priced at par with ``freq`` coupons per year.
    Valid for negative yields as long as ``1 + y/freq > 0``.
    """
    y = np.asarray(y, dtype=float)
    y = np.where(np.abs(y) < 1e-6, 1e-6, y)
    k, N = freq, maturity
    z = 1.0 + y / k
    duration = (1.0 - z ** (-k * N)) / y
    convexity = (2.0 / y**2) * (1.0 - z ** (-k * N)) - (2.0 * N) / (y * z ** (k * N + 1))
    return duration, convexity


def par_bond_excess_returns(
    par_yield: pd.Series,
    maturity: float,
    funding: pd.Series,
    horizon_months: int = 1,
) -> pd.Series:
    """Log excess return of a constant-maturity par bond, stored at ``t``.

    ``par_yield`` is in decimals (bond-equivalent), ``funding`` an annualised
    continuously compounded rate.  The one-month return is
    ``y/12 - D*dy + 0.5*C*dy^2``.  Longer horizons sum monthly log returns,
    i.e. a position rolled into the new on-the-run bond each month.
    """
    y0 = par_yield
    dy = par_yield.shift(-1) - y0
    D, C = par_bond_duration_convexity(y0.to_numpy(), maturity)
    gross = y0 / 12.0 - D * dy + 0.5 * C * dy**2
    rx1 = np.log1p(gross) - funding.reindex(par_yield.index) / 12.0
    rx1.name = rx_column(1, maturity)
    if horizon_months == 1:
        return rx1
    out = forward_sum(rx1, horizon_months)
    out.name = rx_column(horizon_months, maturity)
    return out
