"""Pilot extensions, tested before they join the main pipeline.

Three ideas from the forecasting literature, each built from data known at the
forecast origin:

* **Nowcast** (Knotek and Zaman 2017): the month whose CPI is not yet
  published is largely known from prices observed during it.  ``nowcast``
  predicts the change in y/y inflation at the next print from this month's
  moves in retail gasoline and crude oil, by recursive least squares.
* **Survey path** (Faust and Wright 2013): professional forecasters' expected
  path of y/y inflation, interpolated to each horizon's target month from the
  latest survey published by the origin and the latest print.  Used as a
  predictor (``svy_gap_h``: path minus current inflation) or as the anchor the
  models forecast around.
* **Bottom-up**: energy, food, core goods and core services inflation are
  forecast separately, each from its own drivers, and added up with weights
  estimated on the last five years of published data.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from ..config import Series, fred
from ..data import Downloader
from ..dataset import load_series
from .data import CountryData, fill_price_gaps, yoy
from .features import HORIZONS, base_effects, own_features
from .recursive import recursive_ols
from .surveys import ECB_SPF, SPF_CPI_URL, parse_spf_cpi, us_spf_yoy

TRAIN_START = "1960-01"

# Extra US series: name -> (source, group).  Prices observed within the month have no publication lag.
EXTRA = {
    "US": {
        "gasoline": (fred("GASREGW", how="mean"), "cost"),
        "natgas": (fred("DHHNGSP", how="mean"), "cost"),
        "core_ppi": (fred("WPSFD4131", lag=1), "cost"),
        "import_ex_fuel": (fred("IREXPET", lag=1), "cost"),
        "rent": (fred("CUUR0000SEHA", lag=1), "own"),
        "oer": (fred("CUUR0000SEHC", lag=1), "own"),
        "zori": (Series("zillow", "zori", lag=1), "own"),
    },
}
# Components that add up to the all-items index: name -> (price index, drivers from the main frame and EXTRA).
COMPONENTS = {
    "US": {
        "energy": (fred("CPIENGNS", lag=1),
                   ["oil_yoy", "oil_c3", "oil_c1", "gasoline_yoy", "gasoline_c3", "gasoline_c1", "natgas_yoy",
                    "natgas_c3", "commodities_yoy", "dollar_yoy"]),
        "food": (fred("CPIUFDNS", lag=1),
                 ["food_commodities_yoy", "commodities_yoy", "ppi_yoy", "ppi_c3", "oil_yoy", "wages_yoy", "dollar_yoy",
                  "import_prices_yoy", "g_infl"]),
        "core_goods": (fred("CUUR0000SACL1E", lag=1),
                       ["core_ppi_yoy", "core_ppi_c3", "ppi_yoy", "import_ex_fuel_yoy", "import_prices_yoy",
                        "dollar_yoy", "gscpi_lvl", "gscpi_d3", "commodities_yoy", "retail_yoy", "ip_yoy", "capu_lvl",
                        "unemp_gap", "wages_yoy"]),
        "core_services": (fred("CUUR0000SASLE", lag=1),
                          ["shelter_yoy", "shelter_d3", "rent_yoy", "rent_c3", "oer_yoy", "zori_yoy", "zori_c3",
                           "house_prices_yoy", "house_prices_yoy_l12", "wages_yoy", "wage_tracker_lvl", "unemp_gap",
                           "unemp_lvl", "payrolls_yoy", "sticky_core_lvl", "sticky_core_m_ma3", "median_cpi_lvl",
                           "mich_lvl", "cleveland1y_lvl", "fedfunds_lvl"]),
    },
}


def _monthly(s: pd.Series, index: pd.PeriodIndex) -> pd.Series:
    return s.dropna().reindex(index)


def extra_columns(dl: Downloader, code: str, index: pd.PeriodIndex) -> tuple[pd.DataFrame, dict]:
    """Growth rates of the extra series on the known-at grid, and their groups."""
    cols, groups = {}, {}
    for name, (spec, group) in EXTRA.get(code, {}).items():
        try:
            s = _monthly(load_series(dl, spec), index)
        except Exception as err:  # a missing source should not stop the others
            warnings.warn(f"{code} {name}: unavailable ({str(err)[:100]})")
            continue
        L = 100.0 * np.log(s.where(s > 0))
        for t, k in (("yoy", 12), ("c3", 3), ("c1", 1)):
            cols[f"{name}_{t}"] = L.diff(k)
            groups[f"{name}_{t}"] = group
    return pd.DataFrame(cols, index=index), groups


# ------------------------------------------------------------------ nowcast

def nowcast(frame: pd.DataFrame, gasoline_yoy: pd.Series | None = None, min_train: int = 120) -> pd.Series:
    """Expected change in y/y inflation at the next print (the month that is not yet published).

    Recursive regression of that change on this month's change in the y/y rates of crude oil and,
    where available, retail gasoline, and on the last change in inflation.  Year-on-year rates
    difference out the seasonal pattern on both sides.  Rows up to the previous origin train the
    regression: the change they predict is published by now.
    """
    pi = frame["pi"]
    n = len(frame)
    y = (pi.shift(-1) - pi).to_numpy(float)
    base = [np.ones(n), frame["oil_yoy"].diff().to_numpy(float), frame["pi_d1"].to_numpy(float)]
    t0 = frame.index.searchsorted(pd.Period(TRAIN_START, "M"))
    origins = np.arange(n)
    out = recursive_ols(np.column_stack(base), y, origins, 1, t0, min_train)
    if gasoline_yoy is not None:
        g = gasoline_yoy.reindex(frame.index).diff().to_numpy(float)
        full = recursive_ols(np.column_stack([*base, g]), y, origins, 1, t0, min_train)
        out = np.where(np.isfinite(full), full, out)
    return pd.Series(out, index=frame.index, name="now_d1")


# ------------------------------------------------------------------ survey path

def _us_points(dl: Downloader) -> list[tuple[pd.Period, list[tuple[pd.Period, float]]]]:
    """Per survey: the month it is known from, and (target month, expected y/y inflation) points."""
    spf = parse_spf_cpi(dl.get(SPF_CPI_URL, name="phil_spf_median_cpi_level.xlsx"))
    cpi = load_series(dl, fred("CPIAUCSL")).dropna()
    path = us_spf_yoy(spf, cpi.set_axis(cpi.index.to_timestamp()))
    out = []
    for Q, g in path.groupby("survey"):
        pts = [((Q + int(j)).asfreq("M", "start") + 1, float(f)) for j, f in zip(g["j"], g["forecast"]) if np.isfinite(f)]
        last = max(m for m, _ in pts)
        for years, col in ((1, "CPIB"), (2, "CPIC")):  # Q4-over-Q4 inflation in the following years
            v = spf.loc[Q].get(col, np.nan)
            month = pd.Period(f"{Q.year + years}-11", "M")
            if np.isfinite(v) and month > last:
                pts.append((month, float(v)))
        out.append((Q.asfreq("M", "start") + 1, sorted(pts)))  # returned in the quarter's second month
    return out


def _ea_points(dl: Downloader) -> list[tuple[pd.Period, list[tuple[pd.Period, float]]]]:
    from ..data import fetch_raw

    s12 = fetch_raw(dl, "ecb", ECB_SPF[12]).dropna()
    s24 = fetch_raw(dl, "ecb", ECB_SPF[24]).dropna()
    s12.index, s24.index = pd.PeriodIndex(s12.index, freq="M"), pd.PeriodIndex(s24.index, freq="M")
    out = []
    for T, f in s12.items():
        pts = [(T, float(f))]
        if T + 12 in s24.index:
            pts.append((T + 12, float(s24[T + 12])))
        out.append((T - 12 + 2, pts))  # latest print at the survey is T - 12; results appear the month after the survey
    return out


SURVEY_POINTS = {"US": _us_points, "EA": _ea_points}


def survey_gap(dl: Downloader, code: str, frame: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """``svy_gap_h{h}``: the survey path at each horizon's target month minus current inflation.

    At origin ``k`` the path runs from the latest print through the points of the latest survey
    published by ``k`` that lie beyond it, flat after the last point.  Missing once every point
    of that survey lies behind the latest print.
    """
    surveys = sorted(SURVEY_POINTS[code](dl), key=lambda x: x[0])
    known = [a for a, _ in surveys]
    pi = frame["pi"]
    cols = {h: np.full(len(frame), np.nan) for h in horizons}
    for i, k in enumerate(frame.index):
        j = np.searchsorted(known, k, side="right") - 1
        if j < 0 or not np.isfinite(pi.iloc[i]):
            continue
        pts = [(k - 1, float(pi.iloc[i]))] + [(m, f) for m, f in surveys[j][1] if m > k - 1]
        if len(pts) < 2:
            continue
        xs, ys = [(m - k).n for m, _ in pts], [f for _, f in pts]
        for h in horizons:
            cols[h][i] = np.interp(h - 1, xs, ys) - pi.iloc[i]
    return pd.DataFrame({f"svy_gap_h{h}": v for h, v in cols.items()}, index=frame.index)


# ------------------------------------------------------------------ bottom-up

def component_frames(dl: Downloader, code: str, frame: pd.DataFrame, groups: dict, extras: pd.DataFrame,
                     extra_groups: dict, horizons=HORIZONS) -> dict[str, tuple[pd.DataFrame, dict]]:
    """One feature frame per component, shaped like the main frame so the backtest engine can run on it."""
    wide = pd.concat([frame, extras], axis=1)
    if "house_prices_yoy" in wide:
        wide["house_prices_yoy_l12"] = wide["house_prices_yoy"].shift(12)
    all_groups = {**groups, **extra_groups, "house_prices_yoy_l12": "money"}
    out = {}
    for name, (spec, drivers) in COMPONENTS[code].items():
        P, filled = fill_price_gaps(load_series(dl, spec))
        own = own_features(CountryData(code, P, None, {}, None, {}))
        sa = own.attrs["sa"].reindex(frame.index)
        own = own.reindex(frame.index)
        parts = [own]
        g = {c: "own" for c in own.columns}
        use = [d for d in drivers if d in wide]
        parts.append(wide[use])
        g.update({d: all_groups.get(d, "cost") for d in use})
        parts.append(pd.DataFrame({"hl_pi": frame["pi"], "hl_core": frame["core"]}))
        g.update({"hl_pi": "own", "hl_core": "own"})
        target = own["pi"].copy()
        for p in filled:  # interpolated months are inputs, never targets
            target[target.index.isin([p + 1, p + 13])] = np.nan
        for h in horizons:
            be = base_effects(own["pi"], sa, h)
            parts.append(be)
            g.update({c: "own" for c in be.columns})
            parts.append(target.shift(-h).rename(f"tgt{h}"))
        f = pd.concat(parts, axis=1)
        out[name] = (f.loc[:, ~f.columns.duplicated()], g)
    return out


def aggregation_weights(headline: pd.Series, comps: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Non-negative weights that reproduce headline y/y inflation from the components', per origin.

    Fitted on the ``window`` months published up to each origin; they track the components'
    relative importance without needing the statistical office's weight tables.
    """
    from scipy.optimize import nnls

    X, y = comps.to_numpy(float), headline.to_numpy(float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    W = np.full(X.shape, np.nan)
    for k in range(len(y)):
        rows = np.flatnonzero(ok[max(0, k - window + 1):k + 1]) + max(0, k - window + 1)
        if len(rows) >= window // 2:
            W[k] = nnls(X[rows], y[rows])[0]
    return pd.DataFrame(W, index=comps.index, columns=comps.columns)


def bottom_up(frame: pd.DataFrame, h: int, comp_frames: dict, comp_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Headline forecast from component forecasts: current inflation plus the weighted component changes."""
    names = list(comp_results)
    pis = pd.DataFrame({n: comp_frames[n][0]["pi"] for n in names})
    W = aggregation_weights(frame["pi"], pis)
    idx = None
    for r in comp_results.values():
        idx = r.index if idx is None else idx.intersection(r.index)
    change = sum(W.loc[idx, n] * (comp_results[n].loc[idx, "forecast"] - pis.loc[idx, n]) for n in names)
    out = pd.DataFrame({"forecast": frame["pi"].loc[idx] + change, "rw": frame["pi"].loc[idx],
                        "actual": frame[f"tgt{h}"].loc[idx]})
    return out.dropna(subset=["forecast"])


# ------------------------------------------------------------------ experiments

PILOT_MODELS = ("ridge", "adalasso", "enet", "krr", "rf", "lgbm")


def pilot_sets(dl: Downloader, code: str, frame: pd.DataFrame, groups: dict, h: int) -> dict:
    """Feature sets for the direct models at horizon ``h``: name -> (frame, groups, offset or None)."""
    out = {"current": (frame, groups, None)}
    extras, eg = extra_columns(dl, code, frame.index)
    gas = extras["gasoline_yoy"] if "gasoline_yoy" in extras else None
    now = nowcast(frame, gas)
    ncols = pd.DataFrame({"now_d1": now})
    ng = {"now_d1": "own"}
    for c in ("gasoline_yoy", "gasoline_c3", "gasoline_c1"):
        if c in extras:
            ncols[c], ng[c] = extras[c], "cost"
    out["nowcast"] = (pd.concat([frame, ncols], axis=1), {**groups, **ng}, None)
    if code in SURVEY_POINTS:
        sv = survey_gap(dl, code, frame, (h,))
        col = f"svy_gap_h{h}"
        out["survey"] = (pd.concat([frame, sv.rename(columns={col: "svy_gap"})], axis=1), {**groups, "svy_gap": "expect"}, None)
        # Forecast around the survey path.  Before the first survey the offset is zero (no anchor).
        anchor = sv[col].where(sv[col].notna() | (frame.index >= sv[col].first_valid_index()), 0.0)
        out["survey_anchor"] = (frame.assign(_svy=anchor), groups, "_svy")
        both = pd.concat([frame, ncols], axis=1).assign(_svy=anchor)
        out["nowcast+survey_anchor"] = (both, {**groups, **ng}, "_svy")
    return out


def run_pilot(store: dict, dl: Downloader, codes, horizons=HORIZONS, models=PILOT_MODELS, start: str = "2000-01",
              n_jobs: int = 1, progress=None, bottomup: bool = True) -> dict:
    """Backtest the pilot sets and the bottom-up model: {(economy, h, set, model): results}."""
    from .backtest import model_zoo, run_jobs

    specs = {s.name: s for s in model_zoo(1)}
    frames, jobs, comps = {}, [], {}
    for c in codes:
        frame, groups = store["frames"][c]
        for h in horizons:
            for name, (f, g, offset) in pilot_sets(dl, c, frame, groups, h).items():
                key = f"{c}|{name}|{h}"
                frames[key] = (f, g)
                for m in models:
                    spec = specs[m] if offset is None else replace(specs[m], offset=offset)
                    jobs.append(((key, h, m), spec, dict(start=start)))
        if bottomup and c in COMPONENTS:
            extras, eg = extra_columns(dl, c, frame.index)
            comps[c] = component_frames(dl, c, frame, groups, extras, eg, horizons)
            for name, (f, g) in comps[c].items():
                key = f"{c}|comp:{name}|0"
                frames[key] = (f, g)
                for h in horizons:
                    for m in models:  # components are forecast without the headline's core anchor
                        jobs.append(((key, h, m), replace(specs[m], anchor_horizons=()), dict(start=start)))
    raw = run_jobs(frames, jobs, n_jobs, progress)
    out, parts = {}, {}
    for (key, h, m), r in raw.items():
        c, name, _ = key.split("|")
        if name.startswith("comp:"):
            parts.setdefault((c, h, m), {})[name[5:]] = r
        else:
            out[(c, h, name, m)] = r
    for (c, h, m), rs in parts.items():
        if all(isinstance(r, pd.DataFrame) and not r.empty for r in rs.values()) and len(rs) == len(comps[c]):
            out[(c, h, "bottom_up", m)] = bottom_up(store["frames"][c][0], h, comps[c], rs)
            for name, r in rs.items():
                out[(c, h, f"comp:{name}", m)] = r
    return out
