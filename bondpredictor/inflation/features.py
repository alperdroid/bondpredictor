"""Real-time features and targets for the inflation forecasts.

Timing.  Row ``k`` is a forecast origin at the end of month ``k`` and holds
only what was published by then.  With CPI published one month after the
reference month, the latest print at origin ``k`` is for month ``k - 1``.
The target for horizon ``h`` is year-on-year inflation ``h`` months after
that print, ``pi[k + h]`` on this grid, which is realised at origin
``k + h``.

Models forecast the change ``pi[k + h] - pi[k]``, so a zero forecast is
the random walk.  This keeps tree models, which cannot extrapolate, from
being stuck at the inflation levels seen in training (2021-22 was far
above anything in the training data since the early 1990s).

Seasonality.  Year-on-year rates need no seasonal adjustment, but recent
month-on-month rates do.  :func:`sa_onesided` removes a stable seasonal
pattern estimated only from past data: the median, over the previous six
years, of each calendar month's deviation from a centred 2x12 moving
average.  The latest deviation it uses is centred six months before the
month being adjusted, so no future data enters.  Official seasonally
adjusted series are revised with hindsight and are not used.

Base effects.  For ``h < 12`` part of the target is already known:
``pi[k+h] - pi[k]`` equals the inflation of the next ``h`` months minus that
of the ``h`` months dropping out of the 12-month window.  ``drop`` is the
annualised seasonally adjusted rate of the months dropping out and
``mech`` the change implied if the next ``h`` months ran at the latest
six-month rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .data import CountryData, InflationData, yoy
from .ucsv import ucsv_filter

HORIZONS = (3, 6, 12, 24)


# ------------------------------------------------------------- transforms

def _gap(x: pd.Series, window: int = 120, min_periods: int = 60) -> pd.Series:
    return x - x.rolling(window, min_periods=min_periods).mean()


def _log(x: pd.Series) -> pd.Series:
    return np.log(x.where(x > 0))


TRANSFORMS = {
    "lvl": lambda x: x,
    "d3": lambda x: x - x.shift(3),
    "d12": lambda x: x - x.shift(12),
    "yoy": lambda x: 100.0 * _log(x).diff(12),
    "c3": lambda x: 100.0 * _log(x).diff(3),
    "c1": lambda x: 100.0 * _log(x).diff(1),
    "gap": _gap,  # level minus its trailing 10-year mean
    "gap_log": lambda x: 100.0 * _gap(_log(x)),
}


def sa_onesided(m: pd.Series, years: int = 6, min_years: int = 3) -> pd.Series:
    """Remove a stable seasonal pattern from monthly changes using past data only."""
    centred = (m.rolling(12).mean().shift(-5) + m.rolling(12).mean().shift(-6)) / 2.0
    dev = m - centred  # uses m up to six months ahead of each date
    past = pd.concat([dev.shift(12 * j) for j in range(1, years + 1)], axis=1)
    seasonal = past.median(axis=1).where(past.notna().sum(axis=1) >= min_years)
    return m - seasonal


def _ann(sa: pd.Series, months: int) -> pd.Series:
    """Annualised % rate over the last ``months`` months of seasonally adjusted changes."""
    return sa.rolling(months).sum() * (12.0 / months)


# -------------------------------------------------------------- own block

def own_features(cd: CountryData) -> pd.DataFrame:
    P = cd.cpi
    pi = yoy(P)
    sa = sa_onesided(100.0 * _log(P).diff())
    f = {
        "pi": pi,
        "pi_d1": pi.diff(1),
        "pi_d3": pi.diff(3),
        "pi_d12": pi.diff(12),
        "sa1": 12.0 * sa,
        "sa3": _ann(sa, 3),
        "sa6": _ann(sa, 6),
        "pi_avg36": pi.rolling(36).mean(),
        "pi_avg120": pi.rolling(120, min_periods=60).mean(),
        "pi_vol": (12.0 * sa).rolling(24).std(),
    }
    if cd.core is not None:
        core = yoy(cd.core)
        csa = sa_onesided(100.0 * _log(cd.core).diff())
        f.update({"core": core, "core_sa3": _ann(csa, 3), "core_sa6": _ann(csa, 6), "core_d12": core.diff(12),
                  "headline_minus_core": pi - core})
    for name, s in cd.components.items():
        c = yoy(s)
        f[f"{name}_yoy"] = c
        f[f"{name}_d3"] = c.diff(3)
    if cd.oil is not None:
        for t in ("yoy", "c3", "c1"):
            f[f"oil_{t}"] = TRANSFORMS[t](cd.oil)
    out = pd.DataFrame(f)
    out.attrs["sa"] = sa
    out.attrs["ucsv"] = ucsv_filter(12.0 * sa)
    return out


def base_effects(pi: pd.Series, sa: pd.Series, h: int) -> pd.DataFrame:
    """``drop`` and ``mech`` features for horizon ``h`` (see module docstring)."""
    sa6 = _ann(sa, 6)
    if h < 12:
        drop = sa.rolling(h).sum().shift(12 - h) * (12.0 / h)
        mech = (h / 12.0) * (sa6 - drop)
        return pd.DataFrame({f"drop_h{h}": drop, f"mech_h{h}": mech})
    return pd.DataFrame({f"mech_h{h}": sa6 - pi})


def tax_feature(index: pd.PeriodIndex, events, h: int, cpi_lag: int = 1) -> pd.Series:
    """Known change in the tax contribution to y/y inflation between the latest print and ``h`` months later."""
    T = np.array([(p - cpi_lag).ordinal for p in index])  # latest CPI reference month
    k = np.array([p.ordinal for p in index])
    out = np.zeros(len(index))
    for e in events:
        E = pd.Period(e.effective, "M").ordinal
        A = pd.Period(e.announced, "M").ordinal
        future = (E > T + h - 12) & (E <= T + h) & (A <= k)
        now = (E > T - 12) & (E <= T)
        out += e.change * (future.astype(float) - now.astype(float))
    return pd.Series(out, index=index, name=f"tax_h{h}")


# ------------------------------------------------------------ full frame

# Predictors that exist, under different names, in every economy; used by the pooled models.
COMMON = {
    "unemp": {"US": "unemp", "UK": "unemp", "JP": "unemp", "EA": "unemp"},
    "ip": {"US": "ip", "UK": "ip", "JP": "ip", "EA": "ip"},
    "wages": {"US": "wages", "UK": "earnings", "JP": "earnings"},
    "ppi": {"US": "ppi", "UK": "ppi", "JP": "ppi", "EA": "ppi"},
    "import_prices": {"US": "import_prices", "JP": "import_prices", "EA": "import_prices"},
    "fx": {"US": "dollar", "UK": "reer", "JP": "reer", "EA": "neer"},
    "policy_rate": {"US": "fedfunds", "UK": "policy_rate", "JP": "call_rate", "EA": "euribor3m"},
    "long_rate": {"US": "gs10", "UK": "gilt10", "JP": "jgb10", "EA": "bond10"},
    "money": {"US": "m2", "UK": "money", "JP": "money", "EA": "m3"},
    "stocks": {"US": "stocks", "UK": "stocks", "JP": "stocks"},
    "consumer_conf": {"US": "sentiment", "UK": "consumer_conf", "JP": "consumer_conf", "EA": "consumer_conf"},
    "price_exp": {"US": "mich", "JP": "tankan_output_prices", "EA": "consumer_price_exp"},
    "commodities": {c: "commodities" for c in cfg.COUNTRIES},
    "food_commodities": {c: "food_commodities" for c in cfg.COUNTRIES},
}

GROUP_OF_PREFIX = {"pi": "own", "sa": "own", "core": "own", "headline": "own", "drop": "own", "mech": "own",
                   "oil": "cost", "g_infl": "global", "x_": "global", "tax": "calendar"}


def build_frame(data: InflationData, code: str, horizons=HORIZONS) -> tuple[pd.DataFrame, dict[str, str]]:
    """All features and targets for one economy, and the group of each feature.

    Rows run from the first CPI observation to the current month (the last
    rows have unrealised targets).  Targets are ``tgt{h}``: y/y inflation
    ``h`` months after the latest print.
    """
    c = cfg.COUNTRIES[code]
    cd = data.countries[code]
    own = own_features(cd)
    sa = own.attrs["sa"]
    end = max(data.last_month + 1, cd.cpi.index.max())
    idx = pd.period_range(cd.cpi.dropna().index.min(), end, freq="M")
    parts = [own.reindex(idx)]
    groups = {col: "own" for col in own.columns}
    for col in own.columns:
        if col.startswith("oil_"):
            groups[col] = "cost"
        elif any(col.startswith(n) for n in cd.components):
            groups[col] = "own"
    for name, s in cd.predictors.items():
        spec = c.predictors.get(name) or cfg.GLOBAL_PREDICTORS[name]
        for t in spec.transforms:
            col = f"{name}_{t}"
            parts.append(TRANSFORMS[t](s).rename(col).reindex(idx))
            groups[col] = spec.group
    g = data.global_block
    gf = pd.DataFrame({"g_infl": g["g_infl"], "g_infl_d6": g["g_infl"].diff(6), "g_infl_iqr": g["g_infl_iqr"]})
    parts.append(gf.reindex(idx))
    groups.update({k: "global" for k in gf.columns})
    for other, od in data.countries.items():
        if other != code:
            parts.append(yoy(od.cpi).rename(f"x_{other}").reindex(idx))
            groups[f"x_{other}"] = "global"
    pi = own["pi"].reindex(idx)
    sa_i = sa.reindex(idx)
    # Interpolated CPI months are fine as inputs but are never forecast targets.
    pi_target = pi.copy()
    for p in getattr(cd, "filled", {}).get("CPI", []):
        pi_target[pi_target.index.isin([p, p + 12])] = np.nan
    # Benchmark input, not a predictor (not listed in ``groups``).
    parts.append(own.attrs["ucsv"].reindex(idx).rename("ucsv_tau"))
    cpi_lag = c.cpi[0].lag
    for h in horizons:
        be = base_effects(pi, sa_i, h)
        parts.append(be)
        groups.update({k: "own" for k in be.columns})
        if c.tax_events:
            parts.append(tax_feature(idx, c.tax_events, h, cpi_lag))
            groups[f"tax_h{h}"] = "calendar"
        parts.append(pi_target.shift(-h).rename(f"tgt{h}"))
    frame = pd.concat(parts, axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame.attrs["country"] = code
    return frame, groups


def horizon_columns(frame: pd.DataFrame, groups: dict[str, str], h: int) -> list[str]:
    """Features for horizon ``h``: the base features plus that horizon's base-effect and tax columns."""
    hcols = {f"drop_h{h}", f"mech_h{h}", f"tax_h{h}"}
    out = []
    for col in frame.columns:
        if col.startswith("tgt"):
            continue
        if col.startswith(("drop_h", "mech_h", "tax_h")) and col not in hcols:
            continue
        if col in groups:
            out.append(col)
    return out


def generic_name(col: str, h: int) -> str:
    """Horizon-free name (``mech_h12`` -> ``mech``), used to pool across horizons and economies."""
    for p in ("drop", "mech", "tax"):
        if col == f"{p}_h{h}":
            return p
    return col


def common_columns(code: str, frame: pd.DataFrame, h: int) -> dict[str, str]:
    """Map pooled-model feature names to this economy's columns."""
    out = {}
    for col in frame.columns:
        g = generic_name(col, h)
        if col.startswith(("pi", "sa", "core", "headline", "oil_", "g_infl", "energy", "food")) or g in ("drop", "mech", "tax"):
            if not col.startswith(("drop_h", "mech_h", "tax_h")) or g != col:
                out[g] = col
    for generic, by_country in COMMON.items():
        name = by_country.get(code)
        if name is None:
            continue
        for col in frame.columns:
            if col.startswith(name + "_") and col[len(name) + 1 :] in TRANSFORMS:
                out[f"{generic}_{col[len(name) + 1:]}"] = col
    return out
