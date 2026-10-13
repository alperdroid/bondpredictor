"""Robustness checks for out-of-sample forecasts.

Comparisons that account for the number of models tried:

* Diebold–Mariano test of equal MSE between two non-nested forecasts, with
  the Harvey–Leybourne–Newbold small-sample correction and a HAC variance
  (h-1 lags) for overlapping errors.
* Hansen (2005) SPA test, and White's (2000) Reality Check as its "upper"
  p-value.  Null: no model in the set beats the benchmark.  The stationary
  bootstrap (Politis–Romano) keeps the serial dependence of overlapping
  12-month errors.  Testing many ML and linear variants and reporting the
  best is data snooping; these p-values correct for it.

Stability checks: R²_OS by sub-period and regime, and by evaluation start
date.  Re-estimation variants (rolling window, seeds, fixed hyperparameters,
regime exclusion) are produced by :func:`robustness_jobs` and run through the
same backtest engine.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .evaluation import oos_r2

SUBPERIODS = {
    "1990-1999": ("1990-01", "1999-12"),
    "2000-2007": ("2000-01", "2007-12"),
    "2008-2015 (GFC, QE)": ("2008-01", "2015-12"),
    "2016-2021 (low rates)": ("2016-01", "2021-12"),
    "2022- (inflation shock)": ("2022-01", "2030-12"),
}


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _hac_var_of_mean(d: np.ndarray, lags: int) -> float:
    u = d - d.mean()
    T = len(u)
    s = u @ u / T
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (u[L:] @ u[:-L]) / T
    return s / T


def diebold_mariano(actual, f_a, f_b, horizon: int) -> tuple[float, float]:
    """DM statistic (HLN-corrected) for H0: equal MSE.  Positive means ``f_b`` is more accurate.

    Returns (statistic, two-sided p-value).
    """
    a, fa, fb = (np.asarray(v, dtype=float) for v in (actual, f_a, f_b))
    ok = np.isfinite(a) & np.isfinite(fa) & np.isfinite(fb)
    d = (a[ok] - fa[ok]) ** 2 - (a[ok] - fb[ok]) ** 2
    T, h = len(d), horizon
    if T < 24:
        return float("nan"), float("nan")
    var = _hac_var_of_mean(d, max(h - 1, 0))
    if var <= 0:
        return float("nan"), float("nan")
    dm = d.mean() / math.sqrt(var)
    dm *= math.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 1e-12))
    return float(dm), 2 * (1 - _norm_cdf(abs(dm)))


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis–Romano stationary bootstrap: geometric block lengths, circular wrap."""
    idx = np.empty(n, dtype=int)
    p = 1.0 / mean_block
    idx[0] = rng.integers(n)
    new_block = rng.random(n) < p
    starts = rng.integers(0, n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if new_block[t] else (idx[t - 1] + 1) % n
    return idx


def spa_test(losses: pd.DataFrame, benchmark: str, mean_block: float = 12.0, n_boot: int = 2000,
             seed: int = 0) -> dict:
    """Hansen's SPA test that no model has lower expected loss than ``benchmark``.

    ``losses`` has one column per model (including the benchmark) on common
    dates.  Returns the lower, consistent and upper (Reality Check) p-values
    and the model with the largest studentised improvement.
    """
    L = losses.dropna()
    models = [c for c in L.columns if c != benchmark]
    d = (L[[benchmark]].to_numpy() - L[models].to_numpy())  # >0: model better
    n = len(d)
    dbar = d.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(models)))
    for b in range(n_boot):
        boot[b] = d[stationary_bootstrap_indices(n, mean_block, rng)].mean(axis=0)
    omega = np.sqrt(n) * boot.std(axis=0)
    omega = np.where(omega > 0, omega, np.nan)
    tstat = np.sqrt(n) * dbar / omega
    t_spa = max(np.nanmax(tstat), 0.0)
    thresh = -math.sqrt(2 * math.log(math.log(n)))
    centers = {
        "lower": np.maximum(dbar, 0.0),
        "consistent": np.where(tstat >= thresh, dbar, 0.0),
        "upper": dbar,
    }
    # Bootstrap means are centred at dbar.  Subtracting g(dbar) re-centres each
    # model at 0 (the boundary of the null) or, for clearly poor models under
    # "lower"/"consistent", leaves it at its own negative mean.
    out = {"n": n, "statistic": t_spa, "best": models[int(np.nanargmax(tstat))]}
    for name, g in centers.items():
        z = np.sqrt(n) * (boot - g[None, :]) / omega[None, :]
        tb = np.maximum(np.nanmax(z, axis=1), 0.0)
        out[f"p_{name}"] = float(np.mean(tb >= t_spa))
    return out


def squared_errors(forecasts: dict[str, pd.DataFrame], include_benchmark: bool = True) -> pd.DataFrame:
    """Squared forecast errors per model on the dates all models share."""
    cols = {}
    for name, r in forecasts.items():
        r = r.dropna(subset=["actual", "forecast"])
        cols[name] = (r["actual"] - r["forecast"]) ** 2
        if include_benchmark and "mean" not in cols:
            cols["mean"] = (r["actual"] - r["benchmark"]) ** 2
    return pd.DataFrame(cols).dropna()


def r2_table(forecasts: dict[str, pd.DataFrame], periods: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows = {}
    for name, r in forecasts.items():
        r = r.dropna(subset=["actual"])
        row = {}
        for label, (a, b) in periods.items():
            s = r.loc[pd.Period(a, "M") : pd.Period(b, "M")]
            row[label] = oos_r2(s["actual"], s["forecast"], s["benchmark"]) if len(s) >= 24 else np.nan
        rows[name] = row
    return pd.DataFrame(rows).T


def regime_r2(forecasts: dict[str, pd.DataFrame], dummy: pd.Series, labels=("off", "on")) -> pd.DataFrame:
    """R²_OS separately for months where a regime dummy is 0 and 1 (e.g. zero lower bound)."""
    rows = {}
    for name, r in forecasts.items():
        r = r.dropna(subset=["actual"])
        flag = dummy.reindex(r.index).fillna(0) > 0.5
        row = {}
        for lab, mask in zip(labels, (~flag, flag)):
            s = r[mask]
            row[lab] = oos_r2(s["actual"], s["forecast"], s["benchmark"]) if len(s) >= 24 else np.nan
        rows[name] = row
    return pd.DataFrame(rows).T


def start_sensitivity(forecasts: dict[str, pd.DataFrame], starts=("2000-01", "2005-01", "2010-01", "2015-01")) -> pd.DataFrame:
    return r2_table(forecasts, {f"from {s[:4]}": (s, "2100-12") for s in starts})


# ------------------------------------------------------------- re-estimation

ROLLING_WINDOW = 180  # months
ROLLING_MODELS = ["slope", "curve", "all:combo", "ml:rf", "ml:gbm", "ml:nn"]
SEED_VARIANTS = {"ml:nn": [(5, 6, 7, 8, 9), (10, 11, 12, 13, 14)], "ml:rf": [(1,), (2,)], "ml:gbm": [(1,), (2,)]}
REGIME_EXCLUSIONS = {"DE": ["nirp"], "JP": ["ycc"], "US": ["qe"]}
REGIME_MODELS = ["slope", "curve", "ml:nn"]


def robustness_jobs(frame: pd.DataFrame, countries: list[str], start: str = "1990-01", min_train: int = 120) -> list:
    """Backtest variants as ``((country, variant), spec, kwargs)`` jobs for ``pipeline.run_jobs``."""
    from dataclasses import replace

    from .ml import ML_MODELS
    from .pipeline import country_specs

    base_kw = dict(start=start, min_train=min_train)
    jobs = []
    for c in countries:
        specs = {s.name: s for s in country_specs(c, countries, frame, ml=True)}
        for m in ROLLING_MODELS:
            jobs.append(((c, f"{m} | rolling {ROLLING_WINDOW // 12}y"), specs[m], {**base_kw, "window": ROLLING_WINDOW}))
        for m, seed_sets in SEED_VARIANTS.items():
            for seeds in seed_sets:
                spec = replace(specs[m], estimator_kw={**specs[m].estimator_kw, "seeds": seeds})
                jobs.append(((c, f"{m} | seeds {seeds[0]}-{seeds[-1]}" if len(seeds) > 1 else f"{m} | seed {seeds[0]}"), spec, base_kw))
        for m in ("ml:rf", "ml:gbm", "ml:nn"):
            for i, params in enumerate(ML_MODELS[specs[m].estimator].grid):
                spec = replace(specs[m], estimator_kw={**specs[m].estimator_kw, "params": params})
                label = ",".join(f"{k}={v}" for k, v in params.items())
                jobs.append(((c, f"{m} | fixed {label}"), spec, base_kw))
        spec = replace(specs["ml:nn"], estimator_kw={**specs["ml:nn"].estimator_kw, "clip": False})
        jobs.append(((c, "ml:nn | no input clamping"), spec, base_kw))
        for m in REGIME_MODELS:
            regs = [f"{c}.reg_{r}" for r in REGIME_EXCLUSIONS.get(c, []) if f"{c}.reg_{r}" in frame]
            if regs:
                spec = replace(specs[m], exclude_regimes=regs)
                jobs.append(((c, f"{m} | excl. {'+'.join(REGIME_EXCLUSIONS[c])}"), spec, base_kw))
    return jobs


# -------------------------------------------------------------------- report

MAIN_MODELS = ["slope", "cp", "curve", "curve+macro", "all:ridge", "all:combo", "panel:ridge",
               "ml:rf", "ml:gbm", "ml:nn", "ml:nn_fwd_macro", "ml:ensemble"]


def _esc(v) -> str:
    return str(v).replace("|", "\\|")


def _fmt(df: pd.DataFrame, digits: int = 2) -> str:
    """Markdown table: signed R²/statistics, unsigned 3-decimal p-values, unsigned Sharpe/turnover."""

    def kind(col: str) -> str:
        if re.search(r"(^|\s)p(\s|$)|p-value|Reality Check", col):
            return "p"
        if "Sharpe" in col or "turnover" in col or "corr" in col:
            return "plain"
        return "signed"

    kinds = [kind(str(c)) for c in df.columns]

    def cell(v, k):
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return ""
            if k == "p":
                return f"{v:.3f}"
            if k == "plain":
                return f"{v:.{digits}f}"
            return f"{v:+.{digits}f}"
        return _esc(v)

    head = "| " + " | ".join([_esc(df.index.name or "")] + [_esc(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " | ".join([_esc(i)] + [cell(v, k) for v, k in zip(row, kinds)]) + " |"
            for i, row in zip(df.index, df.to_numpy(dtype=object))]
    return "\n".join([head, sep, *body])


def _common(forecasts: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict all forecast frames to the dates every model covers (with realised actuals)."""
    dates = None
    for r in forecasts.values():
        d = r.dropna(subset=["actual", "forecast"]).index
        dates = d if dates is None else dates.intersection(d)
    return {k: r.loc[dates] for k, r in forecasts.items()}


def country_report(c: str, frame: pd.DataFrame, main: dict[str, pd.DataFrame], variants: dict[str, pd.DataFrame],
                   horizon: int = 12, n_boot: int = 2000) -> str:
    from .evaluation import clark_west, evaluate_strategy

    # Skip models whose predictors were never available (e.g. CP without a zero curve).
    main = _common({m: main[m] for m in MAIN_MODELS if m in main and main[m]["features"].astype(bool).any()})
    if not main:
        return f"## {c}\n\nNo forecasts.\n"
    any_r = next(iter(main.values()))
    monthly = frame[f"{c}.rx1_n10"]
    lines = [f"## {c}  ({any_r.index.min()} – {any_r.index.max()}, {len(any_r)} forecasts)", ""]

    # Accuracy, nested test vs the mean, non-nested DM vs the slope model, strategy net of costs.
    rows = {}
    for m, r in main.items():
        cw_t, cw_p = clark_west(r["actual"], r["forecast"], r["benchmark"], horizon)
        row = {"R²_OS": oos_r2(r["actual"], r["forecast"], r["benchmark"]), "CW p": cw_p}
        if "slope" in main and m != "slope":
            dm, p = diebold_mariano(r["actual"], main["slope"]["forecast"], r["forecast"], horizon)
            row.update({"DM vs slope": dm, "DM p": p})
        else:
            row.update({"DM vs slope": np.nan, "DM p": np.nan})
        gross = evaluate_strategy(r, monthly, horizon)
        net = evaluate_strategy(r, monthly, horizon, cost_bp=5.0)
        row.update({"Sharpe": gross.get("sharpe_model", np.nan), "Sharpe net 5bp": net.get("sharpe_model", np.nan),
                    "mean-model Sharpe": gross.get("sharpe_mean", np.nan), "turnover/yr": gross.get("turnover", np.nan)})
        rows[m] = row
    t = pd.DataFrame(rows).T
    t.index.name = "model"
    lines += ["**Accuracy and economic value** (DM > 0: model beats the slope-only model; Sharpe of the timing "
              "strategy, gross and net of 5 bp per unit of turnover)", "", _fmt(t), ""]

    # Data-snooping-robust tests.
    se = squared_errors(main)
    spa_rows = {}
    spa_rows["any model beats the mean"] = spa_test(se, "mean", n_boot=n_boot)
    if "slope" in se:
        spa_rows["any model beats the slope"] = spa_test(se.drop(columns=["mean"]), "slope", n_boot=n_boot)
    ml_cols = [m for m in se.columns if m.startswith("ml:")]
    if ml_cols:
        spa_rows["any ML model beats the mean"] = spa_test(se[["mean", *ml_cols]], "mean", n_boot=n_boot)
        if "slope" in se:
            spa_rows["any ML model beats the slope"] = spa_test(se[["slope", *ml_cols]], "slope", n_boot=n_boot)
    spa = pd.DataFrame(spa_rows).T[["best", "p_consistent", "p_upper", "p_lower"]]
    spa.columns = ["best model", "SPA p (consistent)", "Reality Check p", "SPA p (lower)"]
    spa.index.name = "null hypothesis rejected if p small"
    lines += ["**Data-snooping-robust tests** (Hansen SPA; stationary bootstrap, mean block 12 months)", "", _fmt(spa), ""]

    # Stability.
    sub = r2_table(main, SUBPERIODS)
    sub.index.name = "R²_OS by sub-period"
    lines += [_fmt(sub), ""]
    if f"{c}.zlb" in frame:
        z = regime_r2(main, frame[f"{c}.zlb"], ("rates > 0.25%", "zero lower bound"))
        z.index.name = "R²_OS by regime"
        lines += [_fmt(z), ""]
    st = start_sensitivity(main)
    st.index.name = "R²_OS by evaluation start"
    lines += [_fmt(st), ""]

    # Re-estimation variants, scored on the main common sample.
    if variants:
        dates = any_r.index
        vrows = {}
        for name, r in sorted(variants.items()):
            base = name.split(" | ")[0]
            s = r.dropna(subset=["actual"]).loc[lambda x: x.index.isin(dates)]
            if len(s) < 24:
                continue
            ref = main.get(base)
            vrows[name] = {
                "R²_OS": oos_r2(s["actual"], s["forecast"], s["benchmark"]),
                "baseline R²_OS": oos_r2(ref["actual"], ref["forecast"], ref["benchmark"]) if ref is not None else np.nan,
                "corr with baseline": float(np.corrcoef(s["forecast"], ref.loc[s.index, "forecast"])[0, 1]) if ref is not None else np.nan,
            }
        v = pd.DataFrame(vrows).T
        v.index.name = "variant (baseline = expanding window, CV-tuned)"
        lines += ["**Re-estimation variants**", "", _fmt(v), ""]
    return "\n".join(lines)
