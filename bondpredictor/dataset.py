"""Assemble the monthly modelling frame for each country.

``build_dataset`` returns one wide DataFrame on a monthly PeriodIndex.
Columns are prefixed by country (``US.slope``, ``DE.rx12_n10``) or ``G.``
for global series.  Targets are the ``*.rx{h}_n{n}`` columns.  Everything
else is a predictor known at the end of the row's month.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfgmod
from .config import CountryConfig, Series
from .curves import SvenssonCurve, TabulatedCurve, ZeroCurve, bootstrap_par_yields, monthly_params
from .data import DataUnavailable, Downloader, fetch_raw
from .data.bundesbank import max_maturity_from_published, svensson_parameters
from .data.fredmd import fetch_fredmd, transform, transform_panel
from .data.japan import fetch_mof_par_yields
from .data.us import fetch_gsw, fetch_kim_wright
from .features import (
    inflation_uncertainty,
    output_gap,
    realized_volatility,
    realized_volatility_monthly,
    regime_dummy,
    trend_inflation,
)
from .returns import maturity_label, par_bond_excess_returns, rx_column, target_horizon, zero_excess_returns
from .timeseries import to_monthly

log = logging.getLogger(__name__)

TARGET_MATURITY = 10.0  # bond used for the timing strategy and the main target


@dataclass
class CountryData:
    code: str
    frame: pd.DataFrame
    curve: ZeroCurve | None
    curve_source: str
    notes: list[str] = field(default_factory=list)


@dataclass
class Dataset:
    frame: pd.DataFrame
    countries: dict[str, CountryData]

    @property
    def sources(self) -> dict[str, str]:
        return {c: d.curve_source for c, d in self.countries.items()}


# ----------------------------------------------------------------- raw series


def load_series(dl: Downloader, spec: Series) -> pd.Series:
    raw = fetch_raw(dl, spec.source, spec.key)
    s = to_monthly(raw, how=spec.how, lag=spec.lag)
    if spec.minus is not None:
        other = load_series(dl, spec.minus)
        idx = s.index.union(other.index)
        s = s.reindex(idx) - other.reindex(idx)
    if spec.divide is not None:
        den = load_series(dl, spec.divide)
        idx = s.index.union(den.index)
        s = s.reindex(idx) / den.reindex(idx)
    return (s * spec.scale).rename(spec.key)


def _try(dl: Downloader, spec: Series | None, notes: list[str], label: str) -> pd.Series | None:
    if spec is None:
        return None
    try:
        return load_series(dl, spec)
    except (DataUnavailable, ValueError, KeyError) as err:
        notes.append(f"{label}: unavailable ({err})")
        return None


def yoy(series: pd.Series, spec: Series) -> pd.Series:
    """Year-on-year inflation in decimals (log change, or converted % change)."""
    if spec.kind == "yoy_pct":
        return np.log1p(series / 100.0)
    return np.log(series.where(series > 0)).diff(12)


def inflation(dl: Downloader, primary: Series | None, extension: Series | None, notes, label) -> pd.Series | None:
    parts = []
    for spec in (primary, extension):
        s = _try(dl, spec, notes, label)
        if s is not None:
            parts.append(yoy(s, spec))
    if not parts:
        return None
    out = parts[0]
    for p in parts[1:]:
        idx = out.index.union(p.index)
        out = out.reindex(idx).combine_first(p.reindex(idx))
    return out


def short_rate_cc(rate_pct: pd.Series, convention: str) -> pd.Series:
    """Convert a quoted money-market/bill rate (percent) to a cc annual decimal."""
    r = rate_pct / 100.0
    if convention == "cc":
        return r
    if convention == "discount3m":
        return -np.log(1.0 - r * 91.0 / 360.0) / (91.0 / 365.0)
    if convention == "simple3m":
        return np.log1p(r * 90.0 / 360.0) / (90.0 / 360.0)
    if convention == "overnight":
        return np.log1p(r / 365.0) * 365.0
    raise ValueError(convention)


# ---------------------------------------------------------------------- curves


def load_curve(dl: Downloader, cfg: CountryConfig, notes: list[str]):
    """Try the configured curve sources in order.

    Returns (curve or None, source name, daily 10y yield or None).
    """
    for src in cfg.curve_sources:
        try:
            if src == "gsw":
                return SvenssonCurve(monthly_params(fetch_gsw(dl)), name="GSW"), "gsw", None
            if src == "kim_wright":
                kw = monthly_params(fetch_kim_wright(dl)) / 100.0
                return TabulatedCurve(kw, name="Kim-Wright"), "kim_wright", None
            if src == "bundesbank":
                params = monthly_params(svensson_parameters(dl, "M"))
                max_mat = to_monthly(max_maturity_from_published(dl, "M"))
                return SvenssonCurve(params, max_maturity=max_mat, name="Bundesbank"), "bundesbank", None
            if src == "mof":
                par = fetch_mof_par_yields(dl)
                zeros = bootstrap_par_yields(monthly_params(par))
                curve = TabulatedCurve(zeros, extrapolate_short=True, short_limit=0.5, name="MOF")
                daily10 = par[10.0] / 100.0 if 10.0 in par else None
                return curve, "mof", daily10
            if src == "par_fred":
                return None, "par_fred", None
            raise ValueError(f"unknown curve source {src}")
        except (DataUnavailable, ValueError, KeyError) as err:
            notes.append(f"curve source {src} unavailable ({err})")
            log.warning("%s: curve source %s unavailable: %s", cfg.code, src, err)
    return None, "none", None


# --------------------------------------------------------------------- country


def build_country(dl: Downloader, cfg: CountryConfig) -> CountryData:
    notes: list[str] = []
    curve, source, daily10 = load_curve(dl, cfg, notes)
    cols: dict[str, pd.Series] = {}

    # Funding rate (cc decimal) for one-month returns and carry.
    money = _try(dl, cfg.short_rate, notes, "short rate")
    short = short_rate_cc(money, cfg.short_rate_convention) if money is not None else None
    if curve is not None and cfg.short_rate_from_curve:
        from_curve = curve.yield_at(cfg.short_rate_from_curve)
        short = from_curve if short is None else from_curve.combine_first(short.reindex(from_curve.index))

    n_star = TARGET_MATURITY
    if curve is not None:
        index = curve.index
        mats = list(cfg.maturities)
        ylds = curve.yields(mats)
        for n in mats:
            cols[f"y_{maturity_label(n)}"] = ylds[float(n)]
        fwd = curve.forwards(range(1, 6))
        for k in range(1, 6):
            cols[f"fwd_{k}"] = fwd[k]
        if short is None:
            short = curve.yield_at(0.25)
        short = short.reindex(index)
        cols["slope"] = cols["y_10"] - cols["y_1"]
        cols["carry"] = curve.forwards([n_star], length=1 / 12).iloc[:, 0] - short
        rx12 = zero_excess_returns(curve, [n for n in mats if n >= 2], 12)
        cols.update({c: rx12[c] for c in rx12.columns})
        rx1 = zero_excess_returns(curve, [n_star], 1, funding=short).iloc[:, 0]
        cols[rx_column(1, n_star)] = rx1
        long_level = cols["y_10"]
    else:
        par = _try(dl, cfg.par_long, notes, "par 10y yield")
        if par is None or short is None:
            raise DataUnavailable(f"{cfg.code}: no curve and no par-yield fallback")
        notes.append("using constant-maturity par-bond approximation (no zero curve)")
        par = par / 100.0
        index = par.index.union(short.index)
        par, short = par.reindex(index), short.reindex(index)
        cols["y_10"] = par
        cols["slope"] = par - short
        cols["carry"] = par - short
        rx1 = par_bond_excess_returns(par, n_star, short, 1)
        cols[rx_column(1, n_star)] = rx1
        cols[rx_column(12, n_star)] = par_bond_excess_returns(par, n_star, short, 12)
        long_level = par

    cols["short"] = short
    cols["policy_chg"] = short - short.shift(3)
    cols["zlb"] = (short < 0.0025).astype(float).where(short.notna())
    cols["mom"] = rx1.shift(1).rolling(12, min_periods=12).sum()

    # Realised rate volatility from daily yields when available.
    daily = daily10
    if daily is None and cfg.long_yield_daily is not None:
        try:
            daily = fetch_raw(dl, cfg.long_yield_daily.source, cfg.long_yield_daily.key) / 100.0
        except DataUnavailable as err:
            notes.append(f"daily yields unavailable ({err})")
    if daily is not None:
        cols["rvol"] = realized_volatility(daily, months=1)
    else:
        cols["rvol"] = realized_volatility_monthly(long_level)

    # Inflation, trend inflation (cycle factor input), uncertainty, value.
    headline = inflation(dl, cfg.cpi, cfg.cpi_extension, notes, "CPI")
    core = inflation(dl, cfg.core_cpi, cfg.core_cpi_extension, notes, "core CPI")
    if core is None:
        core = headline
    if headline is not None:
        cols["infl"] = headline
        cols["infl_unc"] = inflation_uncertainty(headline)
    if core is not None:
        cols["core_infl"] = core
        tau = trend_inflation(core)
        cols["tau"] = tau
        cols["real_yield"] = long_level - tau.reindex(long_level.index)

    # Output gap from quarterly real GDP (computed before moving to monthly).
    if cfg.real_gdp is not None:
        try:
            gdp = fetch_raw(dl, cfg.real_gdp.source, cfg.real_gdp.key)
            cols["output_gap"] = to_monthly(output_gap(gdp), lag=cfg.real_gdp.lag)
        except DataUnavailable as err:
            notes.append(f"real GDP unavailable ({err})")

    for name, spec in (("cb_gdp", cfg.cb_assets_gdp), ("debt_gdp", cfg.debt_gdp)):
        s = _try(dl, spec, notes, name)
        if s is not None:
            cols[name] = s
            cols[f"{name}_chg"] = s - s.shift(12)

    for name, spec in cfg.extras.items():
        s = _try(dl, spec, notes, name)
        if s is not None:
            cols[name] = s

    # Macro panel for Ludvigson–Ng factors.
    panel = None
    if cfg.use_fredmd:
        try:
            levels, tcodes = fetch_fredmd(dl)
            panel = transform_panel(levels, tcodes)
            panel.index = panel.index + 1  # FRED-MD values become available with ~1 month delay
            notes.append(f"macro panel: FRED-MD ({panel.shape[1]} series)")
        except DataUnavailable as err:
            notes.append(f"FRED-MD unavailable ({err}); using curated FRED panel")
    if panel is None:
        series = {}
        for name, spec in cfg.macro_panel.items():
            s = _try(dl, spec, notes, f"panel {name}")
            if s is not None:
                series[name] = transform(s, spec.tcode)
        panel = pd.DataFrame(series)
    for name in panel.columns:
        cols[f"m_{name}"] = panel[name]

    frame = pd.concat({k: v for k, v in cols.items() if v is not None}, axis=1)
    frame = frame.sort_index()
    full = pd.period_range(frame.index.min(), frame.index.max(), freq="M")
    frame = frame.reindex(full)
    for name, (start, end) in cfg.regimes.items():
        frame[f"reg_{name}"] = regime_dummy(frame.index, start, end)
    frame = frame.loc[frame.first_valid_index() :]
    return CountryData(cfg.code, frame, curve, source, notes)


def drop_unrealized(frame: pd.DataFrame, cutoff: pd.Period) -> pd.DataFrame:
    """Blank returns that would need data after ``cutoff``.

    Returns are computed before the sample is cut, so without this a
    12-month return could be completed with a partial-month yield.
    """
    frame = frame.copy()
    for col in frame.columns:
        h = target_horizon(col)
        if h:
            frame.loc[frame.index > cutoff - h, col] = np.nan
    return frame


def build_dataset(
    countries=("US", "DE", "JP"),
    dl: Downloader | None = None,
    start: str = "1960-01",
    end: str | None = None,
) -> Dataset:
    """Download, transform and align everything.

    ``end`` defaults to the last *completed* month, so a partially observed
    current month never enters as if it were a month-end.
    """
    dl = dl or Downloader()
    built = {}
    for code in countries:
        built[code] = build_country(dl, cfgmod.COUNTRIES[code])
        log.info("%s: curve=%s, %d rows", code, built[code].curve_source, len(built[code].frame))
    parts = [d.frame.add_prefix(f"{code}.") for code, d in built.items()]
    for name, spec in cfgmod.GLOBAL.items():
        try:
            s = load_series(dl, spec)
            if name == "vix":
                s = np.log(s)
            parts.append(s.rename(f"G.{name}").to_frame())
        except DataUnavailable as err:
            log.warning("global %s unavailable: %s", name, err)
    frame = pd.concat(parts, axis=1).sort_index()
    full = pd.period_range(max(frame.index.min(), pd.Period(start, "M")), frame.index.max(), freq="M")
    frame = frame.reindex(full)
    # Drop the trailing months where nothing but slow macro data exists.
    last = max(frame[f"{c}.y_10"].last_valid_index() for c in built)
    cutoff = min(last, pd.Period(end, "M") if end else pd.Period(pd.Timestamp.today(), "M") - 1)
    frame = drop_unrealized(frame, cutoff).loc[:cutoff]
    return Dataset(frame=frame, countries=built)
