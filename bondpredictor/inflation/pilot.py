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
from .config import HICP, ecb, estat, oecd_cpi, ons
from .data import CountryData, fill_price_gaps, load_sources, yoy
from .features import HORIZONS, base_effects, own_features
from .recursive import recursive_ols
from .surveys import ECB_SPF, SPF_CPI_URL, parse_spf_cpi, us_spf_yoy

TRAIN_START = "1960-01"
FUEL_MIN_TRAIN = 60  # months before the pump-price nowcast replaces the crude-oil one (UK pump prices start in 2003)
NOWCAST_FUELS = ("gasoline", "diesel")  # road fuels whose average y/y change drives the nowcast

ONS_CPI = "/economy/inflationandpriceindices/timeseries/{}/mm23"
ECB_HICP = "ICP/M.U2.N.{}.4.INX"

# Extra series: name -> (source, group).  Prices observed within the month have no publication lag.
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
    "UK": {
        "gasoline": (Series("ukfuel", "petrol", how="mean"), "cost"),
        "diesel": (Series("ukfuel", "diesel", how="mean"), "cost"),
        "natgas": (fred("PNGASEUUSDM", lag=1), "cost"),  # European gas, US$ (IMF)
    },
    "EA": {
        "gasoline": (Series("ecoil", "euro95", how="mean"), "cost"),
        "diesel": (Series("ecoil", "diesel", how="mean"), "cost"),
        "heating_oil": (Series("ecoil", "heating_oil", how="mean"), "cost"),
        "natgas": (fred("PNGASEUUSDM", lag=1), "cost"),
    },
    "JP": {  # the pump prices need hand-saved files (see data/fuel.py); without them the nowcast uses crude oil in yen
        "gasoline": (Series("meti", "regular", how="mean"), "cost"),
        "kerosene": (Series("meti", "kerosene", how="mean"), "cost"),
        "natgas": (fred("PNGASJPUSDM", lag=1), "cost"),  # LNG in Asia, US$ (IMF)
    },
}
# Components that add up to the all-items index: name -> (price index sources, drivers from the main frame and EXTRA).
COMPONENTS = {
    "US": {
        "energy": ((fred("CPIENGNS", lag=1),),
                   ["oil_yoy", "oil_c3", "oil_c1", "gasoline_yoy", "gasoline_c3", "gasoline_c1", "natgas_yoy",
                    "natgas_c3", "commodities_yoy", "dollar_yoy"]),
        "food": ((fred("CPIUFDNS", lag=1),),
                 ["food_commodities_yoy", "commodities_yoy", "ppi_yoy", "ppi_c3", "oil_yoy", "wages_yoy", "dollar_yoy",
                  "import_prices_yoy", "g_infl"]),
        "core_goods": ((fred("CUUR0000SACL1E", lag=1),),
                       ["core_ppi_yoy", "core_ppi_c3", "ppi_yoy", "import_ex_fuel_yoy", "import_prices_yoy",
                        "dollar_yoy", "gscpi_lvl", "gscpi_d3", "commodities_yoy", "retail_yoy", "ip_yoy", "capu_lvl",
                        "unemp_gap", "wages_yoy"]),
        "core_services": ((fred("CUUR0000SASLE", lag=1),),
                          ["shelter_yoy", "shelter_d3", "rent_yoy", "rent_c3", "oer_yoy", "zori_yoy", "zori_c3",
                           "house_prices_yoy", "house_prices_yoy_l12", "wages_yoy", "wage_tracker_lvl", "unemp_gap",
                           "unemp_lvl", "payrolls_yoy", "sticky_core_lvl", "sticky_core_m_ma3", "median_cpi_lvl",
                           "mich_lvl", "cleveland1y_lvl", "fedfunds_lvl"]),
    },
    # ONS special aggregates of CPIH, the index behind the OECD's UK headline (CPI plus owner occupiers' housing
    # costs and council tax): energy; food, alcohol and tobacco; non-energy industrial goods; services.
    "UK": {
        "energy": ((ons(ONS_CPI.format("l5jj"), lag=1),),
                   ["oil_yoy", "oil_c3", "oil_c1", "gasoline_yoy", "gasoline_c3", "gasoline_c1", "diesel_yoy",
                    "diesel_c3", "natgas_yoy", "natgas_c3", "commodities_yoy", "gbpusd_yoy"]),
        "food": ((ons(ONS_CPI.format("l5je"), lag=1),),
                 ["food_commodities_yoy", "commodities_yoy", "ppi_yoy", "oil_yoy", "earnings_yoy", "gbpusd_yoy",
                  "reer_yoy", "g_infl", "x_EA"]),
        "core_goods": ((ons(ONS_CPI.format("l5jm"), lag=1),),
                       ["ppi_yoy", "reer_yoy", "gbpusd_yoy", "gscpi_lvl", "gscpi_d3", "commodities_yoy", "retail_yoy",
                        "ip_yoy", "unemp_gap", "earnings_yoy", "energy_yoy", "g_infl", "x_EA"]),
        "services": ((ons(ONS_CPI.format("l52g"), lag=1),),
                     ["earnings_yoy", "unemp_gap", "unemp_lvl", "breakeven10_lvl", "policy_rate_lvl",
                      "consumer_conf_lvl", "business_conf_lvl", "money_yoy", "energy_yoy", "food_yoy", "g_infl"]),
    },
    # HICP special aggregates (1996-): energy; food, alcohol and tobacco; non-energy industrial goods; services.
    "EA": {
        "energy": ((estat(HICP.format(item="NRG"), lag=1),),
                   ["oil_yoy", "oil_c3", "oil_c1", "gasoline_yoy", "gasoline_c3", "gasoline_c1", "diesel_yoy",
                    "diesel_c3", "heating_oil_yoy", "heating_oil_c3", "natgas_yoy", "natgas_c3", "commodities_yoy",
                    "eurusd_yoy"]),
        "food": ((estat(HICP.format(item="FOOD"), lag=1),),
                 ["food_commodities_yoy", "commodities_yoy", "ppi_yoy", "import_prices_yoy", "oil_yoy", "eurusd_yoy",
                  "neer_yoy", "retail_price_exp_lvl", "consumer_price_exp_lvl", "g_infl"]),
        # Eurostat's non-energy industrial goods index starts in 2000; the ECB's seasonally adjusted one covers 1996-99.
        "core_goods": ((estat(HICP.format(item="IGD_NNRG"), lag=1), ecb("ICP/M.U2.Y.IGXE00.3.INX", lag=1)),
                       ["ppi_yoy", "import_prices_yoy", "neer_yoy", "eurusd_yoy", "gscpi_lvl", "gscpi_d3",
                        "commodities_yoy", "industry_price_exp_lvl", "industry_price_exp_d12", "retail_price_exp_lvl",
                        "ip_yoy", "ip_gap_log", "unemp_gap", "energy_yoy"]),
        "services": ((estat(HICP.format(item="SERV"), lag=1), ecb(ECB_HICP.format("SERV00"), lag=1)),
                     ["unemp_gap", "unemp_lvl", "consumer_price_exp_lvl", "consumer_price_past_lvl",
                      "retail_price_exp_lvl", "esi_lvl", "consumer_conf_lvl", "euribor3m_lvl", "m3_yoy", "energy_yoy",
                      "food_yoy", "g_infl"]),
    },
    # OECD: energy; food and non-alcoholic beverages; everything else.  No current goods/services split is published there.
    "JP": {
        "energy": ((oecd_cpi("JPN", "CP045_0722", coicop2018=True),),
                   ["oil_yoy", "oil_c3", "oil_c1", "gasoline_yoy", "gasoline_c3", "gasoline_c1", "kerosene_yoy",
                    "kerosene_c3", "natgas_yoy", "natgas_c3", "import_prices_yoy", "import_prices_c3", "usdjpy_yoy",
                    "commodities_yoy"]),
        "food": ((oecd_cpi("JPN", "CP01", coicop2018=True), oecd_cpi("JPN", "CP01")),
                 ["food_commodities_yoy", "commodities_yoy", "import_prices_yoy", "import_prices_c3", "ppi_yoy",
                  "ppi_c3", "usdjpy_yoy", "reer_yoy", "earnings_yoy", "oil_yoy", "g_infl"]),
        "core": ((oecd_cpi("JPN", "_TXCP01_NRG", coicop2018=True),),
                 ["sppi_yoy", "earnings_yoy", "unemp_gap", "unemp_lvl", "tankan_output_prices_lvl",
                  "tankan_output_prices_fcst_lvl", "tankan_infl_outlook_1y_lvl", "import_prices_yoy", "ppi_yoy",
                  "usdjpy_yoy", "reer_yoy", "gscpi_lvl", "consumer_conf_lvl", "call_rate_lvl", "energy_yoy",
                  "food_yoy", "g_infl"]),
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

def fuel_yoy(extras: pd.DataFrame) -> pd.Series | None:
    """Average y/y change of the road fuels that have pump prices (None without any)."""
    cols = [f"{n}_yoy" for n in NOWCAST_FUELS if f"{n}_yoy" in extras]
    return extras[cols].mean(axis=1, skipna=False) if cols else None


def nowcast(frame: pd.DataFrame, gasoline_yoy: pd.Series | None = None, min_train: int = 120,
            fuel_min_train: int = FUEL_MIN_TRAIN) -> pd.Series:
    """Expected change in y/y inflation at the next print (the month that is not yet published).

    Recursive regression of that change on this month's change in the y/y rates of crude oil and,
    where available, retail fuel, and on the last change in inflation.  Year-on-year rates
    difference out the seasonal pattern on both sides.  Rows up to the previous origin train the
    regression: the change they predict is published by now.  The regression with retail fuel takes
    over once it has ``fuel_min_train`` months.
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
        full = recursive_ols(np.column_stack([*base, g]), y, origins, 1, t0, fuel_min_train)
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
    out, notes = {}, []
    for name, (sources, drivers) in COMPONENTS[code].items():
        P = load_sources(dl, sources, "ratio", notes, f"{code} {name}")
        if P is None:
            raise ValueError(f"{code} {name}: no price index available ({'; '.join(notes)})")
        P, filled = fill_price_gaps(P)
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
            if f"tax_h{h}" in frame:  # legislated tax changes, scaled for the all-items index
                parts.append(frame[f"tax_h{h}"])
                g[f"tax_h{h}"] = "calendar"
            parts.append(target.shift(-h).rename(f"tgt{h}"))
        f = pd.concat(parts, axis=1)
        out[name] = (f.loc[:, ~f.columns.duplicated()], g)
    return out


def aggregation_weights(headline: pd.Series, comps: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Non-negative weights, summing to one, that reproduce headline y/y inflation from the components', per origin.

    Fitted on the ``window`` months published up to each origin; they track the components'
    relative importance without needing the statistical office's weight tables.
    """
    from scipy.optimize import nnls

    X, y = comps.to_numpy(float), headline.to_numpy(float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    W = np.full(X.shape, np.nan)
    unit = 1e3  # weight of the adding-up constraint, far above any inflation rate
    for k in range(len(y)):
        rows = np.flatnonzero(ok[max(0, k - window + 1):k + 1]) + max(0, k - window + 1)
        if len(rows) >= window // 2:
            W[k] = nnls(np.vstack([X[rows], np.full(X.shape[1], unit)]), np.append(y[rows], unit))[0]
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
    now = nowcast(frame, fuel_yoy(extras))
    ncols = pd.DataFrame({"now_d1": now})
    ng = {"now_d1": "own"}
    for c in (f"{n}_{t}" for n in NOWCAST_FUELS for t in ("yoy", "c3", "c1")):
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
              n_jobs: int = 1, progress=None, bottomup: bool = True, sets=None, end: str | None = None) -> dict:
    """Backtest the pilot sets and the bottom-up model: {(economy, h, set, model): results}.

    ``sets`` restricts the direct-model feature sets that are run (default: all); ``start`` and ``end``
    bound the forecast origins.
    """
    from .backtest import model_zoo, run_jobs

    specs = {s.name: s for s in model_zoo(1)}
    frames, jobs, comps = {}, [], {}
    span = dict(start=start) if end is None else dict(start=start, end=end)
    for c in codes:
        frame, groups = store["frames"][c]
        for h in horizons:
            for name, (f, g, offset) in pilot_sets(dl, c, frame, groups, h).items():
                if sets is not None and name not in sets:
                    continue
                key = f"{c}|{name}|{h}"
                frames[key] = (f, g)
                for m in models:
                    spec = specs[m] if offset is None else replace(specs[m], offset=offset)
                    jobs.append(((key, h, m), spec, span))
        if bottomup and c in COMPONENTS:
            extras, eg = extra_columns(dl, c, frame.index)
            comps[c] = component_frames(dl, c, frame, groups, extras, eg, horizons)
            for name, (f, g) in comps[c].items():
                key = f"{c}|comp:{name}|0"
                frames[key] = (f, g)
                for h in horizons:
                    for m in models:  # components are forecast without the headline's core anchor
                        jobs.append(((key, h, m), replace(specs[m], anchor_horizons=()), span))
    raw = run_jobs(frames, jobs, n_jobs, progress)
    out = {}
    for (key, h, m), r in raw.items():
        c, name, _ = key.split("|")
        out[(c, h, name, m)] = r
    for c, cf in comps.items():
        out.update(bottom_up_results(store["frames"][c][0], cf, out, c))
    return out


def bottom_up_results(frame: pd.DataFrame, comp_frames: dict, results: dict, code: str) -> dict:
    """``bottom_up`` forecasts for every horizon and model whose component forecasts are all in ``results``."""
    parts = {}
    for (c, h, name, m), r in results.items():
        if c == code and name.startswith("comp:") and isinstance(r, pd.DataFrame) and not r.empty:
            parts.setdefault((h, m), {})[name[5:]] = r
    return {(code, h, "bottom_up", m): bottom_up(frame, h, comp_frames, rs)
            for (h, m), rs in parts.items() if set(rs) == set(comp_frames)}


def rebuild_bottom_up(results: dict, store: dict, dl: Downloader) -> dict:
    """Recompute every economy's bottom-up forecasts from its stored component forecasts."""
    out = {k: v for k, v in results.items() if k[2] != "bottom_up"}
    for c in sorted({k[0] for k in out if k[2].startswith("comp:")}):
        frame, groups = store["frames"][c]
        extras, eg = extra_columns(dl, c, frame.index)
        out.update(bottom_up_results(frame, component_frames(dl, c, frame, groups, extras, eg), out, c))
    return out


# ------------------------------------------------------------------ report

SET_LABELS = {"current": "Current", "nowcast": "+ nowcast", "survey": "+ survey path as predictor",
              "survey_anchor": "Anchored on survey path", "nowcast+survey_anchor": "Nowcast, anchored on survey path",
              "bottom_up": "Bottom-up", "combined": "Combined (three-way average)"}
COMBINED = ("nowcast", "bottom_up", "nowcast+survey_anchor")


def medians(results: dict, code: str, h: int) -> pd.DataFrame:
    """Median across models of each set's forecasts, plus ``combined`` where its three parts exist."""
    sets = sorted({s for (c, hh, s, _) in results if (c, hh) == (code, h) and not s.startswith("comp:")})
    F = pd.DataFrame({s: pd.DataFrame({m: r["forecast"] for (c, hh, ss, m), r in results.items()
                                       if (c, hh, ss) == (code, h, s)}).median(axis=1) for s in sets})
    if set(COMBINED) <= set(F.columns):
        F["combined"] = F[list(COMBINED)].mean(axis=1, skipna=False)
    return F


def score_sets(results: dict, code: str, split: str = "2013-01") -> pd.DataFrame:
    """Relative RMSE of each set's median by horizon and sub-period (by target month), with DM p-values vs current."""
    import math

    from .evaluation import target_month
    from .transform_search import _nw_se

    rows = []
    for h in sorted({hh for (c, hh, _, _) in results if c == code}):
        F = medians(results, code, h)
        base = next(r for (c, hh, s, _), r in results.items() if (c, hh, s) == (code, h, "current"))
        d = pd.concat([F, base[["actual", "rw"]]], axis=1).dropna()
        tm = target_month(d.index, h)
        cut = pd.Period(split, "M")
        for period, mask in (("all", np.ones(len(d), bool)), (f"to {cut.year - 1}", tm < cut), (f"{cut.year} on", tm >= cut)):
            x = d[mask]
            if len(x) < 24:
                continue
            den = math.sqrt(((x["actual"] - x["rw"]) ** 2).mean())
            row = {"economy": code, "horizon": h, "period": period, "n": len(x)}
            for s in F.columns:
                row[s] = math.sqrt(((x["actual"] - x[s]) ** 2).mean()) / den
                if period == "all" and s != "current":
                    dd = ((x["actual"] - x[s]) ** 2 - (x["actual"] - x["current"]) ** 2).to_numpy()
                    row[f"p:{s}"] = math.erfc(abs(dd.mean() / _nw_se(dd, h - 1)) / math.sqrt(2.0))
            rows.append(row)
    return pd.DataFrame(rows)


def report(results: dict, store: dict, surveys: dict | None = None) -> str:
    """Markdown summary of the pilot backtest."""
    from .report import NAMES, md_table
    from .surveys import us_comparison

    L = ["<!-- Generated by `python -m bondpredictor.inflation pilot --report`. -->", "",
         "# Pilot: nowcast, survey path and bottom-up", "",
         "Median of six fast models (ridge, adaptive lasso, elastic net, kernel ridge, random forest, gradient "
         "boosting), monthly origins from 2000, RMSE relative to no change (below 1 beats it).  Sub-periods are by the "
         "month being forecast.  `*` marks a Diebold-Mariano p-value below 0.1 against the current features "
         "(full sample, better or worse).", ""]
    for code in sorted({c for (c, _, _, _) in results}):
        t = score_sets(results, code)
        cols = [s for s in SET_LABELS if s in t.columns]
        show = t[["horizon", "period", "n"]].copy()
        for s in cols:
            p = t.get(f"p:{s}")
            show[SET_LABELS[s]] = [f"{v:.3f}" + ("*" if p is not None and pd.notna(p.iloc[i]) and p.iloc[i] < 0.1 else "")
                                   for i, v in enumerate(t[s])]
        show.index = [f"{h} months, {per}" for h, per in zip(show.pop("horizon"), show.pop("period"))]
        L += [f"## {NAMES.get(code, code)}", "", md_table(show, index_name="horizon, sample"), ""]
    comps = sorted({s[5:] for (c, _, s, _) in results if c == "US" and s.startswith("comp:")})
    if comps:
        rows = {}
        for comp in comps:
            row = {}
            for h in HORIZONS:
                rs = {m: r for (c, hh, s, m), r in results.items() if (c, hh, s) == ("US", h, f"comp:{comp}")}
                if rs:
                    f = pd.DataFrame({m: r["forecast"] for m, r in rs.items()}).median(axis=1)
                    d = pd.concat([f.rename("f"), next(iter(rs.values()))[["actual", "rw"]]], axis=1).dropna()
                    row[f"{h} months"] = float(np.sqrt(((d["actual"] - d["f"]) ** 2).mean() / ((d["actual"] - d["rw"]) ** 2).mean()))
            rows[comp.replace("_", " ")] = row
        L += ["## US components", "",
              "How well each component's own y/y inflation is forecast, relative to no change in that component:", "",
              md_table(pd.DataFrame(rows).T.round(3), index_name="component"), ""]
    if surveys is not None and any(c == "US" for (c, _, _, _) in results):
        res = {}
        for h in HORIZONS:
            F = medians(results, "US", h)
            for s in ("current", "combined"):
                if s in F:
                    res[(h, s)] = pd.DataFrame({"forecast": F[s].dropna()})
        t = us_comparison(store["frames"]["US"][0], res, surveys["us_spf"], surveys["us_cpi_sa"], models=("current", "combined"))
        t = t[t["basis"] == "monthly"].set_index("horizon")
        show = pd.DataFrame({"Professional forecasters": t["rel_survey"], "Current": t["rel_current"],
                             "Combined": t["rel_combined"], "Combined + survey, averaged": t["rel_mix:combined"],
                             "p: combined vs survey": t["dm_p_combined"], "p: average vs survey": t["dm_p_mix:combined"]})
        show.index = [f"{h} months" for h in show.index]
        L += ["## Against the Survey of Professional Forecasters (US)", "",
              "Quarterly survey dates from 2000, y/y inflation in the last month of the target quarter:", "",
              md_table(show.round(3), index_name="horizon"), ""]
    last = {}
    for code in sorted({c for (c, _, _, _) in results}):
        for h in HORIZONS:
            F = medians(results, code, h)
            if not F.empty:
                last[(NAMES.get(code, code), f"{h} months")] = F.iloc[-1]
    if last:
        t = pd.DataFrame(last).T
        t = t[[s for s in SET_LABELS if s in t.columns]].rename(columns=SET_LABELS)
        t.index = [f"{c}, {h}" for c, h in t.index]
        L += ["## Latest forecasts under each variant", "", md_table(t.round(2), index_name="economy, horizon"), ""]
    return "\n".join(L)
