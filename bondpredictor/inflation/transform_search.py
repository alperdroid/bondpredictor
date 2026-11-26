"""Which transformation of each predictor forecasts inflation best?

Every raw predictor of an economy (its own CPI and core index, the CPI
components, oil, each macro series, the global inflation factor and the other
economies' CPI) gets a library of candidate transformations:

* price-like series (positive indexes and prices): log growth over 1, 2, 3, 6,
  9, 12, 18 and 24 months (moving averages of monthly growth, as in the MARX
  transformation of Goulet Coulombe et al. 2021, up to scale), seasonally
  adjusted growth over 1, 3 and 6 months, the change in 12-month growth over
  3, 6 and 12 months, the gap of the log level from its trailing 12-, 36-,
  60- and 120-month mean, 12-month growth as a z-score against its trailing
  36 and 120 months, and its positive and negative parts;
* level-like series (rates, spreads, surveys, unemployment): the level, its
  change over 1, 3, 6, 12 and 24 months, gaps from trailing means, z-scores of
  the level, and the positive and negative parts of the 12-month change;
* each of these delayed by 0, 1, 3, 6 or 12 further months.

All raw series are on the known-at grid (publication lags applied), and every
transformation looks only backwards, so a candidate's value at an origin was
known at that origin.

Each candidate is scored by adding it to the own-inflation benchmark (the
AR-type model of ``backtest.AR_FEATURES``) and forecasting the change in y/y
inflation ``h`` months ahead by OLS from 1960, refitted at every origin on the
rows whose target had been published by then.  The benchmark is refitted on
exactly the candidate's rows, so the two nested models differ only by the
candidate (otherwise a late-starting series would score on its shorter
training window rather than on its content).  Scores use only the origins
where the candidate has a forecast, and need at least 60 of them: the ratio
of mean squared errors to the benchmark, and the Clark-West statistic for
nested models, separately on the selection sample (origins 2000-01 to
2010-12, whose targets are all published by 2012-12 even at 24 months) and on
the holdout (origins from 2013-01).  A transformation chosen on the selection
sample is therefore judged on data it never saw.  Candidates that are linear
combinations of the benchmark terms (for example 12-month CPI growth) are
dropped as aliases.

Which candidates exist is decided on the full sample (at least 60
observations, price or level), which changes no candidate's values.  The
benchmark uses complete rows only; ``run_backtest``'s AR model mean-imputes
missing early values instead, so their forecasts coincide for the US, UK and
Japan but not for the euro area, whose data start in 1991.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config as cfg
from .backtest import AR_FEATURES, design
from .features import sa_onesided

SCREEN_VERSION = 2  # bump when the scoring changes, so a stale screen file is not reused
FIRST_ORIGIN = "2000-01"
TRAIN_START = "1960-01"  # as Spec.train_start
MIN_COVERED = 60
SELECT_END = "2010-12"
HOLDOUT_START = "2013-01"
LAGS = (0, 1, 3, 6, 12)
PRICE_GROWTH = (1, 2, 3, 6, 9, 12, 18, 24)
LEVEL_DIFFS = (1, 3, 6, 12, 24)
GAP_WINDOWS = (12, 36, 60, 120)
Z_WINDOWS = (36, 120)
SAMPLES = {"select": (FIRST_ORIGIN, SELECT_END), "holdout": (HOLDOUT_START, "2100-12"), "full": (FIRST_ORIGIN, "2100-12")}


# ------------------------------------------------------------------ inputs

def raw_inputs(data, code: str) -> dict[str, tuple[pd.Series, str, str]]:
    """Raw known-at series of one economy: name -> (series, kind, group); kind is "price" or "level"."""
    cd, c = data.countries[code], cfg.COUNTRIES[code]
    out = {"cpi": (cd.cpi, "price", "own")}
    if cd.core is not None:
        out["core"] = (cd.core, "price", "own")
    for name, s in cd.components.items():
        out[name] = (s, "price", "own")
    if cd.oil is not None:
        out["oil"] = (cd.oil, "price", "cost")
    for name, s in cd.predictors.items():
        spec = c.predictors.get(name) or cfg.GLOBAL_PREDICTORS[name]
        priced = bool(set(spec.transforms) & {"yoy", "c3", "c1", "gap_log"}) and bool((s.dropna() > 0).all())
        out[name] = (s, "price" if priced else "level", spec.group)
    out["g_infl"] = (data.global_block["g_infl"], "level", "global")
    for other, od in data.countries.items():
        if other != code:
            out[f"cpi_{other}"] = (od.cpi, "price", "global")
    return out


def _monthly(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    return s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))


def _z(x: pd.Series, w: int) -> pd.Series:
    r = x.rolling(w, min_periods=w // 2)
    return (x - r.mean()) / r.std()


def base_transforms(s: pd.Series, kind: str) -> dict[str, pd.Series]:
    """Undelayed candidate transformations of one series (monthly PeriodIndex, no gaps)."""
    out = {}
    if kind == "price":
        L = 100.0 * np.log(s.where(s > 0))
        for k in PRICE_GROWTH:
            out[f"g{k}"] = L.diff(k)
        sa = sa_onesided(L.diff())
        out["sa1"], out["sa3"], out["sa6"] = sa, sa.rolling(3).sum(), sa.rolling(6).sum()
        g12 = out["g12"]
        for k in (3, 6, 12):
            out[f"acc{k}"] = g12.diff(k)
        for w in GAP_WINDOWS:
            out[f"gap{w}"] = L - L.rolling(w, min_periods=w // 2).mean()
        for w in Z_WINDOWS:
            out[f"z{w}"] = _z(g12, w)
        out["pos12"], out["neg12"] = g12.clip(lower=0), g12.clip(upper=0)
    else:
        out["lvl"] = s
        for k in LEVEL_DIFFS:
            out[f"d{k}"] = s.diff(k)
        for w in GAP_WINDOWS:
            out[f"gap{w}"] = s - s.rolling(w, min_periods=w // 2).mean()
        for w in Z_WINDOWS:
            out[f"z{w}"] = _z(s, w)
        out["pos12"], out["neg12"] = out["d12"].clip(lower=0), out["d12"].clip(upper=0)
    return out


def candidates(raw: dict, index: pd.PeriodIndex, lags=LAGS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All candidates on ``index`` (columns ``var:transform@lag``) and their metadata."""
    cols, meta = {}, []
    for var, (s, kind, group) in raw.items():
        s = _monthly(s)
        if len(s) < 36:
            continue
        for base, x in base_transforms(s, kind).items():
            for lag in lags:
                name = f"{var}:{base}@{lag}"
                cols[name] = x.shift(lag).reindex(index)
                meta.append({"name": name, "var": var, "transform": base, "lag": lag, "kind": kind, "group": group})
    X = pd.DataFrame(cols, index=index)
    keep = X.notna().sum() >= 60
    meta = pd.DataFrame(meta).set_index("name").loc[keep[keep].index]
    return X.loc[:, keep], meta


# ------------------------------------------------------------------ scoring

def recursive_ols(Z: np.ndarray, y: np.ndarray, origins: np.ndarray, h: int, t0: int = 0, min_train: int = 120,
                  ridge: float = 1e-9) -> np.ndarray:
    """OLS forecasts at each origin ``k`` from the rows ``t0 <= j <= k - h`` with complete data.

    Uses running sums of the normal equations, so each origin costs one small solve.  The tiny
    ridge is proportional to each diagonal element, so forecasts do not depend on column scaling;
    the rescaling by standard deviations only conditions the arithmetic.
    """
    n, p = Z.shape
    sd = np.nanstd(Z, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    Zs = Z / sd
    ok = np.isfinite(Zs).all(axis=1) & np.isfinite(y)
    ok[:t0] = False
    Zw = np.where(ok[:, None], Zs, 0.0)
    yw = np.where(ok, y, 0.0)
    A = np.cumsum(Zw[:, :, None] * Zw[:, None, :], axis=0)
    b = np.cumsum(Zw * yw[:, None], axis=0)
    cnt = np.cumsum(ok)
    j = origins - h
    good = (j >= 0) & np.isfinite(Zs[origins]).all(axis=1)
    good[good] &= cnt[j[good]] >= min_train
    out = np.full(len(origins), np.nan)
    if good.any():
        Aj, bj = A[j[good]], b[j[good]]
        diag = np.einsum("kii->ki", Aj)
        # A floor keeps an all-zero column (e.g. a tax dummy before its first event) solvable.
        diag = diag + 1e-12 * diag.mean(axis=1, keepdims=True)
        reg = Aj + ridge * diag[:, :, None] * np.eye(p)
        beta = np.linalg.solve(reg, bj[..., None])[..., 0]
        out[good] = (Zs[origins[good]] * beta).sum(axis=1)
    return out


def _nw_se(x: np.ndarray, lags: int) -> float:
    x = x - x.mean()
    n = len(x)
    v = x @ x / n
    for k in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    return math.sqrt(max(v, 1e-18) / n)


def _scores(eb, ec, fb, fc, mask, h) -> dict:
    eb, ec, fb, fc = eb[mask], ec[mask], fb[mask], fc[mask]
    if len(eb) < MIN_COVERED:
        return {"n": len(eb), "ratio": np.nan, "cw_t": np.nan}
    f = eb**2 - ec**2 + (fb - fc) ** 2
    return {"n": len(eb), "ratio": float(np.mean(ec**2) / np.mean(eb**2)), "cw_t": float(f.mean() / _nw_se(f, max(h - 1, 0)))}


def screen(frame: pd.DataFrame, groups: dict, raw: dict, h: int, min_train: int = 120) -> tuple[pd.DataFrame, dict]:
    """Score every candidate of one economy at horizon ``h``.

    Returns a table (one row per candidate, scores per sample) and, for real-time selection, the
    per-origin loss gains ``e_bench**2 - e_cand**2`` with their origins.
    """
    X, _, pi, tgt = design(frame, groups, h)
    y = (tgt - pi).to_numpy(float)
    idx = frame.index
    bench = X[[c for c in AR_FEATURES if c in X]].to_numpy(float)
    B = np.column_stack([np.ones(len(idx)), bench])
    t0 = idx.searchsorted(pd.Period(TRAIN_START, "M"))
    first = idx.searchsorted(pd.Period(FIRST_ORIGIN, "M"))
    origins = np.arange(first, len(idx))
    fb = recursive_ols(B, y, origins, h, t0, min_train)
    use = np.isfinite(fb) & np.isfinite(y[origins])
    origins = origins[use]
    yo = y[origins]
    when = idx[origins]
    masks = {k: (when >= pd.Period(a, "M")) & (when <= pd.Period(b, "M")) for k, (a, b) in SAMPLES.items()}
    C, meta = candidates(raw, idx)
    rows, gains = [], {}
    for name in C.columns:
        x = C[name].to_numpy(float)
        # The benchmark refitted on the candidate's rows: the nested models differ only by the candidate.
        fbc = recursive_ols(np.where(np.isfinite(x)[:, None], B, np.nan), y, origins, h, t0, min_train)
        fc = recursive_ols(np.column_stack([B, x]), y, origins, h, t0, min_train)
        cov = np.isfinite(fc) & np.isfinite(fbc)
        alias = bool(cov.any()) and float(np.max(np.abs(fc[cov] - fbc[cov]))) < 1e-6
        eb, ec = yo - fbc, yo - fc
        row = {"name": name, **meta.loc[name].to_dict(), "alias": alias}
        for k, m in masks.items():
            mk = m & cov
            row[f"{k}_coverage"] = float(mk.sum() / max(m.sum(), 1))
            sc = _scores(eb, ec, fbc, fc, mk, h) if not alias else {"n": int(mk.sum()), "ratio": np.nan, "cw_t": np.nan}
            row.update({f"{k}_{s}": v for s, v in sc.items()})
        rows.append(row)
        gains[name] = np.where(cov & ~alias, eb**2 - ec**2, 0.0).astype(np.float32)
    table = pd.DataFrame(rows)
    return table, {"origins": when, "gains": gains}


def best_per_variable(table: pd.DataFrame, k: int = 1, sample: str = "select") -> pd.DataFrame:
    """The ``k`` best candidates of each variable by the given sample's MSE ratio."""
    t = table.dropna(subset=[f"{sample}_ratio"]).sort_values(f"{sample}_ratio")
    return t.groupby("var", sort=False).head(k)


def run_screen(data, store: dict, codes=("US", "UK", "JP", "EA"), horizons=(3, 6, 12, 24), progress=None) -> dict:
    """Screen every economy and horizon: {"table": all candidates' scores, "gains": {(c, h): {...}}}."""
    tables, gains = [], {}
    for c in codes:
        frame, groups = store["frames"][c]
        raw = raw_inputs(data, c)
        for h in horizons:
            t, extra = screen(frame, groups, raw, h)
            t.insert(0, "horizon", h)
            t.insert(0, "economy", c)
            tables.append(t)
            gains[(c, h)] = extra
            if progress:
                progress((c, h), f"{len(t)} candidates")
    return {"table": pd.concat(tables, ignore_index=True), "gains": gains, "version": SCREEN_VERSION}


# ------------------------------------------------------------------ multivariate comparison

SETS = ("current", "core_anchor", "top1", "top3", "library", "lags")
COMPARE_MODELS = ("ridge", "adalasso", "enet", "krr", "rf", "lgbm")
FEATURE_LAGS = (1, 3, 6, 12)


def feature_sets(frame: pd.DataFrame, groups: dict, raw: dict, table: pd.DataFrame, h: int) -> dict:
    """The feature sets compared at horizon ``h``: name -> (frame, groups, target offset).

    * current: the frame as used by the backtest;
    * core_anchor: the same features, but the models forecast headline inflation relative to
      today's core rate instead of today's headline rate (a stickiness prior);
    * top1 / top3: current plus each variable's best one or three candidates on the selection
      sample (``table`` holds this economy's and horizon's screening scores);
    * library: current plus every undelayed candidate (growth over 1-24 months, accelerations,
      gaps, z-scores, asymmetric parts), letting the models choose jointly;
    * lags: current plus the current features delayed by 1, 3, 6 and 12 months.
    """
    C, meta = candidates(raw, frame.index)
    X, g, _, _ = design(frame, groups, h)

    def plus(names):
        names = [n for n in names if n in C]
        return pd.concat([frame, C[names]], axis=1), {**groups, **{n: meta.loc[n, "group"] for n in names}}

    out = {"current": (frame, groups, None)}
    if "headline_minus_core" in X:
        out["core_anchor"] = (frame, groups, "-headline_minus_core")
    for k in (1, 3):
        out[f"top{k}"] = (*plus(best_per_variable(table, k)["name"].tolist()), None)
    out["library"] = (*plus(meta.index[meta["lag"] == 0].tolist()), None)
    base = [c for c in X.columns if c not in ("drop", "mech", "tax")]
    lagged = {f"L{lag}:{c}": X[c].shift(lag) for lag in FEATURE_LAGS for c in base}
    out["lags"] = (pd.concat([frame, pd.DataFrame(lagged, index=frame.index)], axis=1),
                   {**groups, **{n: g[n.split(":", 1)[1]] for n in lagged}}, None)
    return out


def run_compare(data, store: dict, screen_table: pd.DataFrame, codes, horizons=(3, 6, 12, 24), sets=SETS,
                models=COMPARE_MODELS, start: str = HOLDOUT_START, n_jobs: int = 1, progress=None) -> dict:
    """Backtest each model on each feature set over the holdout: {(economy, h, set, model): results}."""
    from dataclasses import replace

    from .backtest import model_zoo, run_jobs

    specs = {s.name: s for s in model_zoo(1)}
    frames, jobs = {}, []
    for c in codes:
        frame, groups = store["frames"][c]
        raw = raw_inputs(data, c)
        for h in horizons:
            table = screen_table[(screen_table["economy"] == c) & (screen_table["horizon"] == h)]
            for name, (f, g, offset) in feature_sets(frame, groups, raw, table, h).items():
                if name not in sets:
                    continue
                key = f"{c}|{name}|{h}"
                frames[key] = (f, g)
                for m in models:
                    jobs.append(((key, h, m), replace(specs[m], offset=offset), dict(start=start)))
    raw_out = run_jobs(frames, jobs, n_jobs, progress)
    out = {}
    for (key, h, m), r in raw_out.items():
        c, name, _ = key.split("|")
        out[(c, h, name, m)] = r
    return out


def compare_table(results: dict) -> pd.DataFrame:
    """RMSE relative to the random walk per economy, horizon, feature set and model, plus each set's median ensemble."""
    rows = []
    keys = sorted({(c, h) for (c, h, _, _) in results})
    for c, h in keys:
        sub = {(s, m): r for (cc, hh, s, m), r in results.items() if (cc, hh) == (c, h) and isinstance(r, pd.DataFrame) and not r.empty}
        common = None
        for r in sub.values():
            d = r.dropna(subset=["forecast", "actual"]).index
            common = d if common is None else common.intersection(d)
        for s in sorted({s for s, _ in sub}):
            F = pd.DataFrame({m: r.loc[common, "forecast"] for (ss, m), r in sub.items() if ss == s})
            act = next(iter(sub.values())).loc[common, "actual"]
            rw = next(iter(sub.values())).loc[common, "rw"]
            den = np.sqrt(np.mean((act - rw) ** 2))
            for m in F.columns:
                rows.append({"economy": c, "horizon": h, "set": s, "model": m, "n": len(common),
                             "rel_rw": float(np.sqrt(np.mean((act - F[m]) ** 2)) / den)})
            rows.append({"economy": c, "horizon": h, "set": s, "model": "median", "n": len(common),
                         "rel_rw": float(np.sqrt(np.mean((act - F.median(axis=1)) ** 2)) / den)})
            sub[(s, "median")] = pd.DataFrame({"forecast": F.median(axis=1)})
        # Diebold-Mariano versus the current feature set, same model (Newey-West, h - 1 lags), two-sided.
        for row in rows:
            if (row["economy"], row["horizon"]) != (c, h) or row["set"] == "current" or ("current", row["model"]) not in sub:
                continue
            a_ = sub[(row["set"], row["model"])].loc[common, "forecast"]
            b_ = sub[("current", row["model"])].loc[common, "forecast"]
            d = ((act - a_) ** 2 - (act - b_) ** 2).to_numpy()
            t = d.mean() / _nw_se(d, max(h - 1, 0))
            row["dm_t"], row["dm_p"] = float(t), float(math.erfc(abs(t) / math.sqrt(2.0)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ analysis

def transfer_table(table: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation between selection-sample and holdout MSE ratios, by economy and horizon."""
    S = table.dropna(subset=["select_ratio", "holdout_ratio"])
    return S.groupby(["economy", "horizon"]).apply(
        lambda d: d["select_ratio"].rank().corr(d["holdout_ratio"].rank()), include_groups=False).unstack()


def realtime_selection(table: pd.DataFrame, gains: dict, min_track: int = 60) -> pd.DataFrame:
    """Per economy, horizon and variable: holdout gain of re-choosing the transform every month.

    At each origin the chosen candidate is the one with the largest cumulative loss gain over
    origins whose targets were published by then (at least ``min_track`` of them).
    """
    rows = []
    for (c, h), g in gains.items():
        when, gain = g["origins"], g["gains"]
        pos = np.array([(p - when[0]).n for p in when])
        last = np.searchsorted(pos, pos - h, side="right") - 1
        hold = when >= pd.Period(HOLDOUT_START, "M")
        meta = table[(table["economy"] == c) & (table["horizon"] == h) & ~table["alias"]]
        for var, names in meta.groupby("var")["name"]:
            E = np.vstack([gain[n] for n in names]).astype(float)
            cum = np.cumsum(E, axis=1)
            pick = np.where(last >= min_track - 1, np.argmax(cum[:, np.maximum(last, 0)], axis=0), -1)
            chosen = np.where(pick >= 0, E[np.maximum(pick, 0), np.arange(len(when))], 0.0)
            rows.append({"economy": c, "horizon": h, "var": var, "mean_gain": float(chosen[hold].mean())})
    return pd.DataFrame(rows)


def base_rates(frame: pd.DataFrame, gap_min: float = 0.9, oil_min: float = 25.0, start: str = "1960-01") -> dict:
    """What happened over the next 12 months when headline ran well above core during an oil surge."""
    ref = lambda s: s.set_axis(s.index - 1)  # known-at month -> reference month of the print
    pi, core, oil = ref(frame["pi"]), ref(frame["core"]), ref(frame["oil_yoy"])
    df = pd.DataFrame({"pi": pi, "core": core, "gap": pi - core, "oil": oil,
                       "d12": pi.shift(-12) - pi, "core_d12": core.shift(-12) - core}).loc[start:]
    hit = df[(df["gap"] >= gap_min) & (df["oil"] >= oil_min) & df["d12"].notna()]
    return {"months": len(hit), "median_change": float(hit["d12"].median()), "mean_change": float(hit["d12"].mean()),
            "share_fall_2pp": float((hit["d12"] <= -2).mean()), "median_core_change": float(hit["core_d12"].median()),
            "now": df.iloc[-1][["pi", "core", "gap", "oil"]].to_dict(), "now_month": str(df.index[-1])}


SET_LABELS = {"current": "Current features", "core_anchor": "Core-anchored target", "top1": "+ best transform per variable",
              "top3": "+ best three per variable", "library": "+ full transform library", "lags": "+ lags 1-12 of current"}


def report(screen_res: dict, compare: pd.DataFrame | None, frames: dict, latest: pd.DataFrame | None = None) -> str:
    """Markdown summary of the screen, the model comparison and the base-rate check."""
    from .report import NAMES, md_table

    T, G = screen_res["table"], screen_res["gains"]
    S = T[~T["alias"]].dropna(subset=["select_ratio", "holdout_ratio"])
    top = S.sort_values("select_ratio").groupby(["economy", "horizon", "var"]).head(1)
    sig = S[S["select_cw_t"] > 1.645]
    rt = realtime_selection(T, G)
    L = ["<!-- Generated by `python -m bondpredictor.inflation transforms report`. -->", "",
         "# Do predictor transformations matter?", "",
         f"Every raw predictor got a library of candidate transformations ({len(T):,} candidates over four economies "
         f"and four horizons; {int(T['alias'].sum())} exact aliases of the benchmark dropped).  Each was added to the "
         "own-inflation benchmark and scored out of sample in real time: chosen on forecast origins 2000-2010, judged "
         "on origins from 2013.", "",
         "## One variable at a time", "",
         f"* **Early winners do not stay winners.**  The best transformation of each variable on 2000-2010 beat the "
         f"benchmark on 2013-2026 in {100 * (top['holdout_ratio'] < 1).mean():.0f}% of cases; a random candidate did so in "
         f"{100 * (S['holdout_ratio'] < 1).mean():.0f}%.",
         f"* **Significance does not persist.**  Of the {len(sig):,} candidates significant on 2000-2010 (Clark-West, "
         f"5% one-sided), {100 * (sig['holdout_cw_t'] > 1.645).mean():.0f}% were significant again on 2013-2026, against "
         f"{100 * (S['holdout_cw_t'] > 1.645).mean():.0f}% of all candidates.",
         f"* **Re-choosing every month does not help.**  Picking each variable's transformation by its track record to "
         f"date gave a positive holdout gain for {100 * (rt['mean_gain'] > 0).mean():.0f}% of variables.", "",
         "Rank correlation between a candidate's 2000-2010 and 2013-2026 scores (near zero: past performance does not "
         "predict future performance):", "",
         md_table(transfer_table(T).rename(index=NAMES).round(2), index_name="economy"), ""]
    fam = S.groupby("transform")["holdout_ratio"].agg(median="median", beats=lambda s: (s < 1).mean())
    lag = S.groupby("lag")["holdout_ratio"].agg(median="median", beats=lambda s: (s < 1).mean())
    L += ["Holdout MSE relative to the benchmark by transformation (median over variables, economies and horizons) "
          "and share of candidates that beat the benchmark.  The differences are small; short-run changes do a "
          "little better than levels and long gaps.", "",
          md_table(fam.sort_values("median").round(3).rename(columns={"beats": "share < 1"}), index_name="transformation"), "",
          "By extra delay (months):", "", md_table(lag.round(3).rename(columns={"beats": "share < 1"}), index_name="lag"), ""]
    if compare is not None and not compare.empty:
        med = compare[compare["model"] == "median"]
        piv = med.pivot_table(index=["economy", "horizon"], columns="set", values="rel_rw")
        piv = piv[[s for s in SETS if s in piv.columns]].rename(columns=SET_LABELS)
        piv.index = [f"{NAMES.get(c, c)}, {h}m" for c, h in piv.index]
        wins = med[med["set"] != "current"].groupby("set").apply(
            lambda d: pd.Series({"better": (d["dm_t"] < 0).mean(), "significantly better": ((d["dm_t"] < 0) & (d["dm_p"] < 0.1)).mean(),
                                 "significantly worse": ((d["dm_t"] > 0) & (d["dm_p"] < 0.1)).mean()}), include_groups=False)
        wins.index = [SET_LABELS[s] for s in wins.index]
        wins = wins.map(lambda v: f"{100 * v:.0f}%")
        mean = piv.mean().sort_values()
        anchor = med[med["set"] == "core_anchor"].set_index(["economy", "horizon"])["rel_rw"]
        cur = med[med["set"] == "current"].set_index(["economy", "horizon"])["rel_rw"]
        long_wins = int((anchor.xs(24, level="horizon") < cur.xs(24, level="horizon")).sum())
        L[4:4] = ["**Bottom line.**  Averaged over the four economies and four horizons, the current features "
                  f"({mean['Current features']:.3f} of the no-change error) beat every alternative: "
                  + ", ".join(f"{k.lstrip('+ ').lower()} {v:.3f}" for k, v in mean.items() if k != "Current features")
                  + f".  Anchoring the target on core inflation helps two years out ({long_wins} of 4 economies) but "
                  "hurts at short horizons in the euro area and Japan, where energy swings dominate.", ""]
        L += ["## Letting the models choose", "",
              "The fast econometric ML models (ridge, adaptive lasso, elastic net, kernel ridge, random forest, gradient "
              "boosting) backtested on origins from 2013 with each feature set.  RMSE of their median relative to no "
              "change (below 1 beats it):", "", md_table(piv.round(3), index_name="economy, horizon"), "",
              "How often each alternative beat the current features (median model, 16 economy-horizon pairs; Diebold-"
              "Mariano at 10%):", "", md_table(wins, index_name="feature set"), ""]
    if latest is not None and not latest.empty:
        lt = latest.pivot_table(index="h", columns="set", values="median")
        lt = lt[[c for c in ("current", "core_anchor", "library") if c in lt.columns]].rename(columns=SET_LABELS)
        lt.index = [f"{h} months" for h in lt.index]
        L += ["## What the alternatives say about the US today", "",
              "Median of the same six models at the latest origin (October 2026 print, 3.9%), by feature set:", "",
              md_table(lt.round(2), index_name="horizon"), ""]
    br = base_rates(frames["US"][0])
    L += ["## Stickiness check for the current US forecast", "",
          f"In the {br['months']} US months since 1960 when headline inflation ran at least 0.9 points above core during "
          f"an oil surge of 25% or more, inflation fell a median {abs(br['median_change']):.1f} points over the next "
          f"year; it fell 2 points or more {100 * br['share_fall_2pp']:.0f}% of the time, and core moved a median "
          f"{br['median_core_change']:+.1f} points.  In {br['now_month']} headline was {br['now']['pi']:.1f}%, core "
          f"{br['now']['core']:.1f}% and oil {br['now']['oil']:+.0f}% on the year.", ""]
    return "\n".join(L)
