"""Yield-curve trades: duration-neutral steepeners and butterflies.

A steepener is long the short leg and short the long leg in equal duration
(weights 1/a and -1/b on zero-coupon bonds of maturities a < b).  Its return
is close to the change in the a-b yield spread plus its carry, so it does
not depend on the level of rates.  A butterfly is long the belly and short
equal halves of the wings (weights 1/m, -0.5/a, -0.5/b), which bets on
curvature with neither duration nor slope exposure.  Each trade is a
separate bet, so three trades in three markets give nine bets instead of
the one global term-premium bet that duration timing makes.

Curves are month-end zero curves: GSW (US, 1961-), Bundesbank (Germany,
1972-) and Bank of England (UK, 1970-).

Fitting noise.  A yield that is mis-fitted at month-end t enters both the
signal and the return measured from t, so a strategy can profit from noise
that mean-reverts.  The main specification therefore trades with a
one-month delay (signal at t, held from t+1 to t+2), which no month-t
fitting error can reach.  The undelayed results are shown alongside.  In
the US they replicate on returns from the Treasury's own constant-maturity
curve, an independent fit (``us_independent_check``), so the truth lies
between the two.  Funding and the policy signal use measured money-market
rates, not the curve's fitted short end.

Signals (positions clipped to [-2, 2]; nothing is fitted):

* ``carry``     carry plus roll-down of the trade if the curve stays put,
                over its own 10-year standard deviation
* ``value``     mean reversion: minus the z-score of the slope (steepeners)
                or plus the z-score of curvature (butterflies; a cheap belly
                is bought), over the last 10 years
* ``momentum``  the trade's 12-month return over its volatility
* ``policy``    minus the z-score of the 6-month change in the short rate:
                hiking cycles flatten curves (steepeners only)
* ``combo``     average of the available signals

Each trade's position is divided by its 36-month return volatility, so every
trade carries the same risk; the portfolio averages across trades.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .curves import TabulatedCurve, ZeroCurve, monthly_params
from .data import DataUnavailable, Downloader
from .data.uk import fetch_boe_curve
from .dataset import load_curve, load_series, short_rate_cc
from .relval import performance
from .returns import zero_excess_returns
from .robustness import SUBPERIODS, _fmt, spa_test

SIGNALS = ("carry", "value", "momentum", "policy", "combo")
WINDOW = 120  # months of history for z-scores
MIN_HISTORY = 60
VOL_WINDOW = 36
COSTS_BP = (0.5, 1.0)  # per unit of spread DV01 traded (half-spread of 0.25 / 0.5 bp on each of two legs)


@dataclass(frozen=True)
class Trade:
    country: str
    legs: tuple[float, ...]  # (short, long) or (short, belly, long)

    @property
    def kind(self) -> str:
        return "steepener" if len(self.legs) == 2 else "butterfly"

    @property
    def name(self) -> str:
        legs = "s".join(f"{x:g}" for x in self.legs) + "s"
        return f"{self.country} {legs} {'steepener' if self.kind == 'steepener' else 'fly'}"

    def weights(self) -> dict[float, float]:
        if self.kind == "steepener":
            a, b = self.legs
            return {a: 1.0 / a, b: -1.0 / b}
        a, m, b = self.legs
        return {m: 1.0 / m, a: -0.5 / a, b: -0.5 / b}


TRADES = (
    Trade("US", (2, 10)), Trade("US", (5, 30)), Trade("US", (2, 5, 10)),
    Trade("DE", (2, 10)), Trade("DE", (5, 20)), Trade("DE", (2, 5, 10)),
    Trade("GB", (2, 10)), Trade("GB", (5, 20)), Trade("GB", (2, 5, 10)),
)

# Measured money-market rates for funding and the policy signal (the fitted
# 1-month point of a curve is unreliable, especially GSW's).
_MARKET_SHORT = {
    "US": (config.US.short_rate, config.US.short_rate_convention),
    "DE": (config.DE.short_rate, config.DE.short_rate_convention),
    "GB": (config.fred("IR3TIB01GBM156N"), "simple3m"),
}


# ---------------------------------------------------------------------- data


def load_curves(dl: Downloader, countries=("US", "DE", "GB")) -> dict[str, tuple[ZeroCurve, pd.Series]]:
    """Month-end zero curve and funding rate (cc decimal) per country."""
    out = {}
    for c in countries:
        notes: list[str] = []
        if c == "GB":
            curve = TabulatedCurve(monthly_params(fetch_boe_curve(dl)) / 100.0, extrapolate_short=True,
                                   short_limit=1.0, name="BoE")
        else:
            curve, src, _ = load_curve(dl, config.COUNTRIES[c], notes)
            if curve is None:
                raise DataUnavailable(f"{c}: no zero curve ({'; '.join(notes)})")
        fitted = curve.yield_at(1 / 12)
        spec, conv = _MARKET_SHORT[c]
        try:
            market = short_rate_cc(load_series(dl, spec), conv).reindex(fitted.index)
            funding = market.combine_first(fitted)  # measured rate first, fitted short end only to fill gaps
        except DataUnavailable:
            funding = fitted
        out[c] = (curve, funding)
    return out


# ------------------------------------------------------------------- signals


def _z(x: pd.Series, demean: bool = True) -> pd.Series:
    roll = x.rolling(WINDOW, min_periods=MIN_HISTORY)
    centre = roll.mean() if demean else 0.0
    return (x - centre) / roll.std()


def trade_frame(curve: ZeroCurve, funding: pd.Series, trade: Trade) -> pd.DataFrame:
    """Return (t -> t+1), state variables and signal positions of one trade, at t."""
    mats = list(trade.legs)
    w = trade.weights()
    rx = zero_excess_returns(curve, mats, 1, funding=funding)
    ret = sum(w[n] * rx.iloc[:, i] for i, n in enumerate(mats))
    y = curve.yields(mats)
    y_aged = curve.yields([n - 1 / 12 for n in mats])
    r = funding.reindex(y.index)
    carry_leg = {n: n * y.iloc[:, i] - (n - 1 / 12) * y_aged.iloc[:, i] - r / 12 for i, n in enumerate(mats)}
    carry = 12 * sum(w[n] * carry_leg[n] for n in mats)
    if trade.kind == "steepener":
        a, b = mats
        level = y.iloc[:, 1] - y.iloc[:, 0]
        value = -_z(level)
        d6 = r - r.shift(6)
        policy = -d6 / d6.rolling(WINDOW, min_periods=MIN_HISTORY).std()
    else:
        level = y.iloc[:, 1] - 0.5 * (y.iloc[:, 0] + y.iloc[:, 2])
        value = _z(level)
        policy = pd.Series(np.nan, index=y.index)
    vol = ret.shift(1).rolling(VOL_WINDOW, min_periods=24).std()  # returns realised by t
    momentum = ret.shift(1).rolling(12).sum() / (vol * math.sqrt(12))
    pos = pd.DataFrame({
        "carry": carry / carry.rolling(WINDOW, min_periods=MIN_HISTORY).std(),
        "value": value,
        "momentum": momentum,
        "policy": policy,
    }).clip(-2, 2)
    pos["combo"] = pos.mean(axis=1).where(pos.notna().sum(axis=1) >= 2)
    out = pd.DataFrame({"ret": ret, "vol": vol, "level": level, "carry_ann": carry, "short": r})
    return pd.concat([out, pos.add_prefix("pos_")], axis=1)


def build_trades(curves: dict, trades=TRADES) -> dict[str, pd.DataFrame]:
    return {t.name: trade_frame(*curves[t.country], t) for t in trades if t.country in curves}


# ----------------------------------------------------------------- portfolio


def portfolio(frames: dict[str, pd.DataFrame], signal: str, start=None, end=None, sign_only: bool = False,
              risk_scale: bool = True, flat_below: float | None = None, lag: int = 0) -> pd.DataFrame:
    """Equal-risk average across trades of position x return, stored at the position date.

    ``turnover`` is in spread-DV01 units (what costs scale with); ``pos_turnover``
    is the change in the clipped signal position, for display.  ``lag = 1``
    holds a position formed at t from t+1 to t+2, so fitting noise in the
    month-t curve cannot reach the return.
    """
    pnl, turn, pturn = {}, {}, {}
    for name, f in frames.items():
        p = f[f"pos_{signal}"]
        if sign_only:
            p = np.sign(p)
        if flat_below is not None:
            p = p.where(~(f["short"] < flat_below), 0.0)
        scaled = p / f["vol"] if risk_scale else p / f["vol"].median()
        pnl[name] = scaled * f["ret"].shift(-lag)
        turn[name] = scaled.fillna(0.0).diff().abs().where(scaled.notna())
        pturn[name] = p.fillna(0.0).diff().abs().where(p.notna())
    pnl, turn, pturn = pd.DataFrame(pnl), pd.DataFrame(turn), pd.DataFrame(pturn)
    out = pd.DataFrame({"ret": pnl.mean(axis=1), "turnover": turn.mean(axis=1),
                        "pos_turnover": pturn.mean(axis=1), "n": pnl.notna().sum(axis=1)})
    if start:
        out = out.loc[pd.Period(start, "M") :]
    if end:
        out = out.loc[: pd.Period(end, "M")]
    return out.dropna(subset=["ret"])


def _sr(res: pd.DataFrame) -> float:
    r = res["ret"].dropna()
    return float(r.mean() / r.std() * math.sqrt(12)) if len(r) > 24 else float("nan")


def shift_placebo(frames: dict[str, pd.DataFrame], signal: str, actual_sr: float, start: str, end: str,
                  n_perm: int = 1000, seed: int = 0, min_shift: int = 24, lag: int = 0) -> float:
    """Share of random circular time shifts of each trade's positions with an equal or higher Sharpe ratio.

    Shifting keeps the persistence of positions and returns but breaks their
    alignment, so the null keeps the features that make time-series
    strategies look good by chance.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M")
    P = np.column_stack([(f[f"pos_{signal}"] / f["vol"]).reindex(idx).to_numpy() for f in frames.values()])
    R = np.column_stack([f["ret"].shift(-lag).reindex(idx).to_numpy() for f in frames.values()])
    T = len(idx)
    hits = 0
    for _ in range(n_perm):
        Ps = np.empty_like(P)
        for j in range(P.shape[1]):
            Ps[:, j] = np.roll(P[:, j], rng.integers(min_shift, T - min_shift))
        with np.errstate(invalid="ignore"):
            pnl = np.nanmean(np.where(np.isfinite(Ps * R), Ps * R, np.nan), axis=1)
        pnl = pnl[np.isfinite(pnl)]
        hits += pnl.mean() / pnl.std() * math.sqrt(12) >= actual_sr
    return hits / n_perm


def us_independent_check(dl: Downloader, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """US trade frames with returns re-measured on the Treasury constant-maturity curve.

    Signals stay GSW-based.  Each leg is a par bond repriced a month later at
    its aged maturity on FRED's month-end CMT curve (linear interpolation), so
    roll-down is included but no GSW fitting error enters the return.
    """
    from .data.fred import fetch_fred
    from .returns import par_bond_duration_convexity
    from .timeseries import to_monthly

    grid = [1, 2, 3, 5, 7, 10, 20, 30]
    cmt = pd.DataFrame({n: to_monthly(fetch_fred(dl, f"DGS{n}") / 100.0) for n in grid})
    vals, g = cmt.to_numpy(), np.array(grid, dtype=float)

    def at(n: float) -> pd.Series:
        out = np.full(len(cmt), np.nan)
        for i, row in enumerate(vals):
            ok = np.isfinite(row)
            if ok.sum() >= 2:
                out[i] = np.interp(n, g[ok], row[ok], left=np.nan, right=np.nan)
        return pd.Series(out, index=cmt.index)

    any_us = next(f for k, f in frames.items() if k.startswith("US "))
    funding = any_us["short"].reindex(cmt.index)
    rx, dur = {}, {}
    for n in (2, 5, 10, 30):
        y0, y1 = cmt[n], at(n - 1 / 12).shift(-1)
        D, C = par_bond_duration_convexity(y0.to_numpy(), n)
        dy = y1 - y0
        rx[n] = np.log1p(y0 / 12 - D * dy + 0.5 * C * dy**2) - funding / 12
        dur[n] = pd.Series(D, index=cmt.index)
    alt = {(2, 10): rx[2] / dur[2] - rx[10] / dur[10], (5, 30): rx[5] / dur[5] - rx[30] / dur[30],
           (2, 5, 10): rx[5] / dur[5] - 0.5 * (rx[2] / dur[2] + rx[10] / dur[10])}
    out = {}
    for t in TRADES:
        if t.country == "US" and t.name in frames and t.legs in alt:
            f = frames[t.name].copy()
            f["ret"] = alt[t.legs].reindex(f.index)
            out[t.name] = f
    return out


# --------------------------------------------------------------- the study


@dataclass
class CurveStudy:
    main: pd.DataFrame
    returns: dict[str, pd.DataFrame]
    report: str


def study(dl: Downloader, start: str = "1980-01", end: str | None = None, n_perm: int = 1000, lag: int = 1,
          timing_returns: pd.Series | None = None, relval_returns: pd.DataFrame | None = None) -> CurveStudy:
    """Full evaluation.  ``lag = 1`` (default) is the conservative, noise-proof specification."""
    curves = load_curves(dl)
    frames = build_trades(curves)
    last = max(f["ret"].last_valid_index() for f in frames.values()) - lag
    end = end or str(last)

    def pf(fr, sig, a=start, **kw):
        return portfolio(fr, sig, a, end, lag=kw.pop("lag", lag), **kw)

    rets = {s: pf(frames, s) for s in SIGNALS}
    main = pd.DataFrame({s: performance(r, costs_bp=COSTS_BP) for s, r in rets.items()}).T
    main["position change/yr"] = [float(r["pos_turnover"].mean() * 12) for r in rets.values()]
    main["SR, no delay"] = [_sr(pf(frames, s, lag=0)) for s in SIGNALS]
    main.index.name = "signal"
    main["placebo p"] = [shift_placebo(frames, s, main.loc[s, "SR"], start, end, n_perm=n_perm, lag=lag)
                         for s in SIGNALS]

    R = pd.DataFrame({s: r["ret"] for s, r in rets.items()}).dropna()
    spa = {}
    for label, lo in ((f"from {start[:4]}", start), ("from 1990", "1990-01")):
        Rw = R.loc[pd.Period(lo, "M") :]
        spa[label] = spa_test(pd.concat([pd.Series(0.0, index=Rw.index, name="zero"), -Rw], axis=1), "zero",
                              n_boot=2000)

    per_trade = pd.DataFrame({name: {s: _sr(pf({name: f}, s)) for s in SIGNALS} for name, f in frames.items()}).T
    per_trade["combo, no delay"] = [_sr(pf({name: f}, "combo", lag=0)) for name, f in frames.items()]
    per_trade.insert(0, "first signal", pd.Series({n: str(f["pos_combo"].first_valid_index()) for n, f in frames.items()}))
    per_trade.index.name = "Sharpe ratio by trade"

    per_country = pd.DataFrame({c: {s: _sr(pf({k: f for k, f in frames.items() if k.startswith(c + " ")}, s))
                                    for s in SIGNALS} for c in curves}).T
    per_country.index.name = "Sharpe ratio by market"

    check = None
    try:
        us_alt = us_independent_check(dl, frames)
        us = {k: f for k, f in frames.items() if k in us_alt}
        check = pd.DataFrame({
            "GSW returns, no delay": {s: _sr(pf(us, s, lag=0)) for s in SIGNALS},
            "Treasury CMT returns, no delay": {s: _sr(pf(us_alt, s, lag=0)) for s in SIGNALS},
            "GSW returns, 1-month delay": {s: _sr(pf(us, s, lag=1)) for s in SIGNALS},
            "Treasury CMT returns, 1-month delay": {s: _sr(pf(us_alt, s, lag=1)) for s in SIGNALS},
        })
        check.index.name = "US trades, Sharpe ratio"
    except DataUnavailable:
        pass

    variants = {
        "sign of the signal only": dict(sign_only=True),
        "equal DV01 (no risk scaling)": dict(risk_scale=False),
        "flat when short rate < 0.5%": dict(flat_below=0.005),
    }
    vrows = {label: {s: _sr(pf(frames, s, **kw)) for s in SIGNALS} for label, kw in variants.items()}
    vrows["start 1990"] = {s: _sr(pf(frames, s, "1990-01")) for s in SIGNALS}
    vrows["start 2000"] = {s: _sr(pf(frames, s, "2000-01")) for s in SIGNALS}
    vrows["steepeners only"] = {s: _sr(pf({k: f for k, f in frames.items() if "steepener" in k}, s)) for s in SIGNALS}
    vrows["butterflies only"] = {s: _sr(pf({k: f for k, f in frames.items() if "fly" in k}, s)) for s in SIGNALS}
    vrows["no delay (exposed to fitting noise)"] = {s: _sr(pf(frames, s, lag=0)) for s in SIGNALS}
    variants = pd.DataFrame(vrows).T
    variants.index.name = "Sharpe ratio, variant"

    sub = pd.DataFrame({label: {s: _sr(r.loc[pd.Period(a, "M") : pd.Period(b, "M")]) for s, r in rets.items()}
                        for label, (a, b) in SUBPERIODS.items()})
    sub.index.name = "Sharpe ratio by sub-period"

    text = _report(frames, main, spa, per_trade, per_country, check, variants, sub, rets, start, end, n_perm, lag,
                   timing_returns, relval_returns)
    return CurveStudy(main, rets, text)


def _combination(rets, timing_returns, relval_returns, lag) -> pd.DataFrame | None:
    """Correlations and equal-risk combinations with duration timing and cross-country value.

    A curve return at t is earned from t+lag to t+lag+1; the timing return at
    t from t to t+1; a relative-value return at t from mid t+1 to mid t+2.
    """
    parts = {"curve combo": rets["combo"]["ret"], "curve policy": rets["policy"]["ret"]}
    other = {}
    if timing_returns is not None:
        other["duration timing"] = timing_returns.shift(-lag)
    if relval_returns is not None and "value" in relval_returns:
        other["relative value"] = relval_returns["value"].shift(1 - lag)
    if not other:
        return None
    df = pd.concat({**parts, **other}, axis=1).dropna()
    if len(df) < 36:
        return None
    z = df / df.std()
    rows = {}
    for name in df.columns:
        rows[name] = {"SR alone": df[name].mean() / df[name].std() * math.sqrt(12)}
        for o in other:
            if name != o:
                rows[name][f"corr with {o}"] = df[name].corr(df[o])
    allz = z[["curve combo", *other]].mean(axis=1)
    rows["equal risk: curve combo + " + " + ".join(other)] = {"SR alone": allz.mean() / allz.std() * math.sqrt(12)}
    no_curve = z[list(other)].mean(axis=1)
    rows["equal risk: " + " + ".join(other) + " (no curve)"] = {"SR alone": no_curve.mean() / no_curve.std() * math.sqrt(12)}
    t = pd.DataFrame(rows).T
    t.index.name = f"{df.index.min()} – {df.index.max()}"
    return t


def _report(frames, main, spa, per_trade, per_country, check, variants, sub, rets, start, end, n_perm, lag,
            timing_returns, relval_returns) -> str:
    cols = ["N", "start", "SR", "t (NW)", "placebo p", "SR net 0.5bp", "SR net 1bp", "SR, no delay",
            "position change/yr", "hit rate", "max DD at 10% vol"]
    timing = ("signals formed at month-end t are traded with a one-month delay (held from t+1 to t+2), so curve-fitting "
              "noise cannot reach the return" if lag else "positions formed at month-end earn the next month's return")
    lines = [
        "# Yield-curve trades",
        "",
        f"Trades: {', '.join(frames)}.  Month-end GSW, Bundesbank and Bank of England zero curves; {timing}.  "
        f"Evaluation {start} – {end} (signal dates).  Each trade risk-scaled by its 36-month volatility, equal-risk "
        "average across trades.",
        "",
        "## Main results (all nine trades)",
        "",
        _fmt(main[[c for c in cols if c in main]]),
        "",
        f"Placebo p: share of {n_perm:,} random circular time shifts of each trade's positions with an equal or higher "
        "Sharpe ratio.  Costs: 0.5 / 1 bp per unit of spread DV01 traded (a 0.25 / 0.5 bp half-spread on each of two "
        "legs).  \"SR, no delay\" trades at the signal month-end and is exposed to curve-fitting noise.",
        "",
        "## Multiple-testing correction across the five signals",
        "",
        "Hansen SPA test of the null that no signal has a positive mean return (stationary bootstrap, mean block "
        "12 months, 2,000 draws):",
        "",
    ]
    for label, res in spa.items():
        lines.append(f"- {label}: consistent p = {res['p_consistent']:.3f}, Reality Check p = {res['p_upper']:.3f} "
                     f"(best signal: {res['best']})")
    lines += ["", "## By trade and market", "", _fmt(per_trade), "", _fmt(per_country), ""]
    if check is not None:
        lines += ["## Is it fitting noise? US check on an independent curve", "",
                  "Signals from GSW; returns re-measured on the Treasury's constant-maturity curve (a separate fit), "
                  "with roll-down.  If GSW fitting noise drove the results, they would vanish on CMT returns.", "",
                  _fmt(check), ""]
    comb = _combination(rets, timing_returns, relval_returns, lag)
    if comb is not None:
        lines += ["## Combination with the other strategies", "", _fmt(comb), ""]
    lines += ["## Robustness", "", _fmt(variants), "", _fmt(sub), ""]
    return "\n".join(lines)
