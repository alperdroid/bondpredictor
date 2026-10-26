"""Cross-country government bond relative value: value, carry and momentum.

Instead of timing one market's duration, each month the sovereign bond
markets are ranked on a signal.  The attractive ones are held long and the
unattractive ones short, each side carrying the same duration risk.  Returns
are local excess returns (bond return minus the local short rate).  Under
covered interest parity these equal currency-hedged returns, as in
Asness–Moskowitz–Pedersen (2013) and Koijen–Moskowitz–Pedersen–Vrugt (2018).
Spreading the bet over 10–17 markets is what gives it breadth; timing three
correlated markets does not.

Timing convention for monthly-average yields
--------------------------------------------
The free OECD long-term rates (FRED ``IRLTLT01..M156N``) are monthly
*averages*.  Suppose a signal uses the month-t average and the return runs
from the month-t average to the month-t+1 average.  Those two overlap
month t, and part of that month is already known when the signal is formed,
which is a look-ahead.  So a signal formed at the end of month t is traded
at the average price of month t+1 and closed at the average of month t+2,
which amounts to executing gradually over the month (TWAP).  The return
then shares no month with the data in the signal.  ``lag=0`` gives the
biased naive version for comparison.

Signals (all known at the end of month t; nothing is fitted):

* ``value``      real yield: 10y yield minus year-on-year CPI inflation
* ``value_5y``   5-year yield reversal: average 10y yield 4.5–5.5 years ago minus today's
* ``carry``      10y yield minus the 3-month rate
* ``momentum``   sum of the last 12 monthly excess returns
* ``infl_trend`` minus the 12-month change in year-on-year inflation
* ``vcm``        equal-weight composite of value, carry and momentum z-scores
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import JP_DEFLATOR, Series, fred
from .data import DataUnavailable, Downloader
from .dataset import _try, inflation, load_series, short_rate_cc
from .returns import par_bond_excess_returns
from .robustness import SUBPERIODS, _fmt, spa_test

G10 = ("US", "CA", "GB", "DE", "JP", "AU", "NZ", "SE", "CH", "NO")
EURO_EXTRA = ("FR", "IT", "ES", "NL", "BE", "AT", "DK")
_HICP = {"DE", "FR", "IT", "ES", "NL", "BE", "AT", "DK", "SE", "NO", "CH", "GB"}

SIGNALS = ("value", "value_5y", "carry", "momentum", "infl_trend", "vcm")
COMPOSITE = ("value", "carry", "momentum")
STALE_INFLATION_MONTHS = 12  # inflation held this long after a source stops


# ---------------------------------------------------------------------- data


def market_specs(code: str) -> dict:
    """FRED series for one market: OECD 10y yield, short rates and CPI with fallbacks."""
    if code == "US":
        cpi = (fred("CPIAUCSL", lag=1), None)
    elif code == "JP":
        cpi = (fred("JPNCPIALLMINMEI", lag=1), JP_DEFLATOR)
    elif code == "AU":
        cpi = (fred("AUSCPIALLQINMEI", lag=2), None)  # quarterly
    elif code == "NZ":
        cpi = (fred("NZLCPIALLQINMEI", lag=2), None)  # quarterly
    else:
        cpi = (fred(f"CPALTT01{code}M659N", lag=1, kind="yoy_pct"),
               fred(f"CP0000{code}M086NEST", lag=1) if code in _HICP else None)
    return {
        "long": fred(f"IRLTLT01{code}M156N"),
        "short": [(fred(f"IR3TIB01{code}M156N"), "simple3m"), (fred(f"IRSTCI01{code}M156N"), "overnight")],
        "cpi": cpi,
    }


@dataclass
class Market:
    code: str
    yld: pd.Series  # 10y yield, decimal (monthly average)
    short: pd.Series  # short rate, continuously compounded decimal
    infl: pd.Series | None  # year-on-year inflation, decimal, publication-lagged
    notes: list[str] = field(default_factory=list)


def load_market(dl: Downloader, code: str) -> Market:
    spec = market_specs(code)
    notes: list[str] = []
    y = load_series(dl, spec["long"]) / 100.0
    short = None
    for s, conv in spec["short"]:
        raw = _try(dl, s, notes, f"{code} short rate")
        if raw is None:
            continue
        cc = short_rate_cc(raw, conv)
        short = cc if short is None else short.combine_first(cc)
    if short is None:
        raise DataUnavailable(f"{code}: no short rate")
    infl = inflation(dl, *spec["cpi"], notes, f"{code} CPI")
    if infl is not None:
        idx = pd.period_range(infl.index.min(), infl.index.max() + STALE_INFLATION_MONTHS, freq="M")
        infl = infl.reindex(idx).ffill(limit=STALE_INFLATION_MONTHS)
    return Market(code, y.rename(code), short.rename(code), None if infl is None else infl.rename(code), notes)


def load_universe(dl: Downloader, codes=G10) -> dict[str, Market]:
    return {c: load_market(dl, c) for c in codes}


# ------------------------------------------------------------------- signals


@dataclass
class Panel:
    """Dates x markets frames.  ``fwd`` at t is the return the position formed at t earns."""

    ret: pd.DataFrame  # monthly excess return from the month-t average to the month-t+1 average, at t
    fwd: pd.DataFrame
    vol: pd.DataFrame
    signals: dict[str, pd.DataFrame]
    lag: int


def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1), axis=0)


def build_panel(markets: dict[str, Market], lag: int = 1, maturity: float = 10.0, vol_window: int = 36) -> Panel:
    y = pd.DataFrame({c: m.yld for c, m in markets.items()}).sort_index()
    idx = pd.period_range(y.index.min(), y.index.max(), freq="M")
    y = y.reindex(idx)
    short = pd.DataFrame({c: m.short for c, m in markets.items()}).reindex(idx)
    infl = pd.DataFrame({c: (m.infl.reindex(idx) if m.infl is not None else pd.Series(np.nan, index=idx))
                         for c, m in markets.items()})

    ret = pd.DataFrame({c: par_bond_excess_returns(y[c], maturity, short[c], 1) for c in y.columns})
    fwd = ret.shift(-lag)  # lag=1: average of t+1 to average of t+2
    vol = ret.shift(1).rolling(vol_window, min_periods=24).std()  # returns ending at the month-t average

    sig = {
        "value": y - infl,
        "value_5y": -(y - y.rolling(13).mean().shift(54)),
        "carry": y - short,
        "momentum": ret.shift(1).rolling(12).sum(),
        "infl_trend": -(infl - infl.shift(12)),
    }
    z = [_zscore(sig[s]) for s in COMPOSITE]
    stacked = np.stack([f.to_numpy() for f in z])
    count = np.isfinite(stacked).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells -> NaN
        comp = np.where(count >= 2, np.nanmean(stacked, axis=0), np.nan)
    sig["vcm"] = pd.DataFrame(comp, index=idx, columns=y.columns)
    return Panel(ret, fwd, vol, sig, lag)


# ----------------------------------------------------------------- portfolio


def weights(signal: pd.DataFrame, vol: pd.DataFrame, min_n: int = 5, scheme: str = "rank",
            vol_scale: bool = True) -> pd.DataFrame:
    """Long-short notional weights from a cross-sectional signal, using only data at t.

    Rank (or z-score) deviations from the cross-sectional mean; the long side
    sums to +1 and the short side to -1 in risk units.  With ``vol_scale`` each
    position is then scaled by median vol / own vol, so a low-volatility market
    takes a larger notional.  Months with fewer than ``min_n`` markets hold
    nothing.
    """
    avail = signal.notna() & vol.notna() & (vol > 0)
    s = signal.where(avail)
    if scheme == "rank":
        d = s.rank(axis=1)
        d = d.sub(d.mean(axis=1), axis=0)
    elif scheme == "zscore":
        d = _zscore(s).clip(-2.5, 2.5)
    else:
        raise ValueError(scheme)
    pos, neg = d.clip(lower=0), (-d).clip(lower=0)
    w = pos.div(pos.sum(axis=1), axis=0) - neg.div(neg.sum(axis=1), axis=0)
    if vol_scale:
        v = vol.where(avail)
        w = w.mul(v.median(axis=1), axis=0) / v
    w[avail.sum(axis=1) < min_n] = np.nan
    return w


def portfolio_returns(w: pd.DataFrame, fwd: pd.DataFrame, hold: int = 1) -> pd.DataFrame:
    """Monthly portfolio returns at the signal date, with turnover.

    ``hold > 1`` averages the weights of the last ``hold`` signal dates
    (overlapping portfolios, Jegadeesh–Titman).
    """
    active = w.notna().any(axis=1)
    w = w.fillna(0.0)
    if hold > 1:
        w = w.rolling(hold, min_periods=1).mean()
    w = w.where(active, np.nan)  # months with no position at all
    r = (w * fwd).sum(axis=1, min_count=1)
    r[fwd.reindex(columns=w.columns).isna().all(axis=1) | ~active] = np.nan
    turnover = w.fillna(0.0).diff().abs().sum(axis=1)
    return pd.DataFrame({"ret": r, "turnover": turnover.where(r.notna())})


def run_strategy(panel: Panel, signal: str, start: str | None = None, end: str | None = None, **kw) -> pd.DataFrame:
    hold = kw.pop("hold", 1)
    w = weights(panel.signals[signal], panel.vol, **kw)
    out = portfolio_returns(w, panel.fwd, hold=hold)
    if start:
        out = out.loc[pd.Period(start, "M") :]
    if end:
        out = out.loc[: pd.Period(end, "M")]
    return out.dropna(subset=["ret"])


# --------------------------------------------------------------- statistics


def _nw_t(x: np.ndarray, lags: int = 3) -> float:
    u = x - x.mean()
    T = len(u)
    s = u @ u / T
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (u[L:] @ u[:-L]) / T
    return float(x.mean() / math.sqrt(s / T)) if s > 0 else float("nan")


def performance(res: pd.DataFrame, costs_bp=(5.0, 10.0)) -> dict:
    """Annualised statistics of a monthly long-short return series."""
    r = res["ret"].dropna()
    if len(r) < 24:
        return {"N": len(r)}
    mu, sd = r.mean() * 12, r.std() * math.sqrt(12)
    scaled = r / r.std() * (0.10 / math.sqrt(12))  # at 10% annual volatility
    wealth = scaled.cumsum()
    out = {
        "N": len(r),
        "start": str(r.index.min()),
        "SR": mu / sd,
        "t (NW)": _nw_t(r.to_numpy()),
        "hit rate": float((r > 0).mean()),
        "max DD at 10% vol": float((wealth - wealth.cummax()).min()),
        "turnover/yr": float(res["turnover"].loc[r.index].mean() * 12),
    }
    for c in costs_bp:
        net = r - c * 1e-4 * res["turnover"].loc[r.index]
        out[f"SR net {c:g}bp"] = net.mean() / net.std() * math.sqrt(12)
    return out


def placebo_pvalue(panel: Panel, signal: str, actual_sr: float, start=None, end=None, n_perm: int = 1000,
                   seed: int = 0, **kw) -> float:
    """Share of random cross-country rankings with a Sharpe ratio at least the actual one.

    Each month, the markets with a valid signal are ranked at random.  This
    keeps the universe, the volatility scaling and the return path, and
    destroys only the signal's information.
    """
    rng = np.random.default_rng(seed)
    base = panel.signals[signal]
    mask = base.notna()
    hits = 0
    for _ in range(n_perm):
        rnd = pd.DataFrame(rng.random(base.shape), index=base.index, columns=base.columns).where(mask)
        res = portfolio_returns(weights(rnd, panel.vol, **kw), panel.fwd)
        if start:
            res = res.loc[pd.Period(start, "M") :]
        if end:
            res = res.loc[: pd.Period(end, "M")]
        r = res["ret"].dropna()
        hits += (r.mean() / r.std() * math.sqrt(12)) >= actual_sr
    return hits / n_perm


def timing_portfolio(frame: pd.DataFrame, forecasts: dict, model: str = "slope", vol_window: int = 60) -> pd.Series:
    """Equal-risk portfolio of the single-market duration-timing strategies.

    Uses the backtest forecasts of ``model`` and the month-end 1-month returns
    of the main panel.  The value at t is the return from t to t+1.
    """
    from .evaluation import timing_strategy

    parts = {}
    for (c, name), res in forecasts.items():
        if name != model:
            continue
        rx = frame[f"{c}.rx1_n10"]
        vol = rx.shift(1).rolling(vol_window, min_periods=24).std()
        strat = timing_strategy(res["forecast"], rx, 12)["ret"]
        parts[c] = strat / vol.reindex(strat.index)
    return pd.DataFrame(parts).mean(axis=1).dropna().rename("timing")


# --------------------------------------------------------------- the study


@dataclass
class RelValStudy:
    main: pd.DataFrame  # performance per signal, main specification
    returns: dict[str, pd.DataFrame]
    report: str


def _sr(res: pd.DataFrame) -> float:
    r = res["ret"].dropna()
    return float(r.mean() / r.std() * math.sqrt(12)) if len(r) > 24 else float("nan")


def study(dl: Downloader, universe=G10, extended=EURO_EXTRA, start: str = "1980-01", end: str | None = None,
          n_perm: int = 1000, timing_returns: pd.Series | None = None) -> RelValStudy:
    markets = load_universe(dl, tuple(universe) + tuple(extended))
    core = {c: markets[c] for c in universe}
    panel = build_panel(core)
    full = build_panel(markets)
    naive = build_panel(core, lag=0)
    # Last month with a realised return for (nearly) every market.
    last = panel.fwd.dropna(thresh=max(len(core) - 2, 1)).index.max()
    end = end or str(last)

    rets = {s: run_strategy(panel, s, start, end) for s in SIGNALS}
    main = pd.DataFrame({s: performance(r) for s, r in rets.items()}).T
    main.index.name = "signal"
    main["placebo p"] = [placebo_pvalue(panel, s, main.loc[s, "SR"], start, end, n_perm=n_perm) for s in SIGNALS]

    # Data-snooping-robust test across all signals: does any have a positive mean?
    R = pd.DataFrame({s: r["ret"] for s, r in rets.items()}).dropna()
    spa = {}
    for label, lo in ((f"from {start[:4]}", start), ("from 1990", "1990-01")):
        Rw = R.loc[pd.Period(lo, "M") :]
        losses = pd.concat([pd.Series(0.0, index=Rw.index, name="zero"), -Rw], axis=1)
        spa[label] = spa_test(losses, "zero", n_boot=2000)

    # Variants of the main specification.
    variant_specs = {
        "z-score weights": dict(scheme="zscore"),
        "no volatility scaling": dict(vol_scale=False),
        "3-month holding (overlapping)": dict(hold=3),
        "at least 8 markets": dict(min_n=8),
    }
    vrows = {}
    for label, kw in variant_specs.items():
        vrows[label] = {s: _sr(run_strategy(panel, s, start, end, **dict(kw))) for s in SIGNALS}
    if extended:
        vrows[f"+{len(extended)} euro/EEA markets"] = {s: _sr(run_strategy(full, s, start, end)) for s in SIGNALS}
    vrows["start 1990"] = {s: _sr(run_strategy(panel, s, "1990-01", end)) for s in SIGNALS}
    vrows["start 2000"] = {s: _sr(run_strategy(panel, s, "2000-01", end)) for s in SIGNALS}
    vrows["BIASED: no implementation lag"] = {s: _sr(run_strategy(naive, s, start, end)) for s in SIGNALS}
    variants = pd.DataFrame(vrows).T
    variants.index.name = "Sharpe ratio, variant"

    sub = {}
    for label, (a, b) in SUBPERIODS.items():
        sub[label] = {s: _sr(r.loc[pd.Period(a, "M") : pd.Period(b, "M")]) for s, r in rets.items()}
    sub = pd.DataFrame(sub)
    sub.index.name = "Sharpe ratio by sub-period"

    loo = {}
    for c in universe:
        sub_panel = build_panel({k: v for k, v in core.items() if k != c})
        loo[f"without {c}"] = {s: _sr(run_strategy(sub_panel, s, start, end, min_n=4)) for s in SIGNALS}
    loo = pd.DataFrame(loo).T
    loo.index.name = "Sharpe ratio, leave one market out"

    text = _report(core, panel, main, spa, variants, sub, loo, rets, start, end, timing_returns, n_perm)
    return RelValStudy(main, rets, text)


def _report(core, panel, main, spa, variants, sub, loo, rets, start, end, timing_returns, n_perm) -> str:
    cols = ["N", "start", "SR", "t (NW)", "placebo p", "SR net 5bp", "SR net 10bp", "turnover/yr", "hit rate",
            "max DD at 10% vol"]
    lines = [
        "# Cross-country bond relative value",
        "",
        f"Universe: {', '.join(core)} ({len(core)} markets).  Signals formed at month-end, traded at the next "
        f"month's average price, held one month.  Evaluation {start} – {end}.  Local excess returns of 10-year "
        "par bonds (currency-hedged under covered interest parity), long-short, risk-scaled.",
        "",
        "## Main results",
        "",
        _fmt(main[[c for c in cols if c in main]]),
        "",
        f"Placebo p: share of {n_perm:,} random cross-country rankings with an equal or higher Sharpe ratio.",
        "",
        "## Multiple-testing correction across the six signals",
        "",
        "Hansen SPA test of the null that no signal has a positive mean return (stationary bootstrap, "
        "mean block 12 months, 2,000 draws):",
        "",
    ]
    for label, res in spa.items():
        lines.append(f"- {label}: consistent p = {res['p_consistent']:.3f}, Reality Check p = {res['p_upper']:.3f} "
                     f"(best signal: {res['best']})")
    lines.append("")
    if timing_returns is not None:
        R = pd.DataFrame({s: r["ret"] for s, r in rets.items()})
        # A relative-value return at t is earned from mid t+1 to mid t+2; pair it
        # with the timing return over month t+2 (stored at t+1).
        both = pd.concat([timing_returns.shift(-1).rename("timing"), R], axis=1).dropna()
        if len(both) > 36:
            corr = both.corr()["timing"].drop("timing")
            z = both.div(both.std())
            combo = {s: (z["timing"] + z[s]) / 2 for s in R.columns}
            t = pd.DataFrame({
                "corr with timing": corr,
                "SR alone": {s: both[s].mean() / both[s].std() * math.sqrt(12) for s in R.columns},
                "SR timing alone": both["timing"].mean() / both["timing"].std() * math.sqrt(12),
                "SR 50/50 risk": {s: c.mean() / c.std() * math.sqrt(12) for s, c in combo.items()},
            })
            t.index.name = f"with duration timing ({both.index.min()} – {both.index.max()})"
            lines += ["## Combination with duration timing", "", _fmt(t), ""]
    lines += ["## Robustness", "", _fmt(variants), "",
              "The BIASED row uses the month-t average both in the signal and as the entry price. Comparing it "
              "with the main results shows how much that look-ahead would flatter each signal (most of all "
              "carry and the composite).", "",
              _fmt(sub), "", _fmt(loo), ""]
    return "\n".join(lines)
