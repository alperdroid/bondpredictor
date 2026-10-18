"""Forecast evaluation for the inflation backtests.

* Accuracy: RMSE relative to the random walk (no change in y/y inflation),
  the standard benchmark since Atkeson and Ohanian (2001), and to UCSV.
* Diebold-Mariano against the random walk, with the Harvey-Leybourne-Newbold
  correction and a HAC variance for overlapping h-step errors.
* Model Confidence Set (Hansen, Lunde and Nason 2011): the set of models
  that cannot be distinguished from the best at level alpha, with the
  T_max statistic and a stationary bootstrap.  Medeiros et al. (2021)
  report it for their inflation horse race.
* Sub-periods by the month being forecast, so "2021-22" means forecasts
  *of* the inflation surge.
* Intervals: rolling conformal intervals for every model (empirical 10th
  and 90th percentiles of the model's own realised errors over the past
  ten years, using only errors realised by the origin), and BART's
  posterior predictive intervals.  Scored by coverage and the interval
  score of Gneiting and Raftery (2007).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..robustness import diebold_mariano, stationary_bootstrap_indices

# Fixed in advance (not chosen by out-of-sample results).
NONLINEAR_ML = ("krr", "rf", "lgbm", "mrf", "bart", "hnn")
BENCHMARKS = ("rw", "ucsv", "ar")

SUBPERIODS = {
    "2000-07": ("2000-01", "2007-12"),
    "2008-20": ("2008-01", "2020-12"),
    "2021-22 surge": ("2021-01", "2022-12"),
    "2023-": ("2023-01", "2100-12"),
}


def target_month(index: pd.PeriodIndex, h: int) -> pd.PeriodIndex:
    """Reference month of the CPI print being forecast (origin + h - 1 with a one-month publication lag)."""
    return index + (h - 1)


def add_ensembles(results: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Equal-weight combinations fixed in advance: non-linear ML mean, ML + UCSV, median of the tier-1 models."""
    out = dict(results)

    def combo(members, how="mean"):
        members = [m for m in members if m in results and not results[m].empty]
        if len(members) < 2:
            return None
        F = pd.concat({m: results[m]["forecast"] for m in members}, axis=1).dropna()
        base = results[members[0]].loc[F.index, ["rw", "actual"]].copy()
        base["forecast"] = F.mean(axis=1) if how == "mean" else F.median(axis=1)
        return base

    ml = combo(NONLINEAR_ML)
    if ml is not None:
        out["ens:ml"] = ml
        if "ucsv" in results:
            u = results["ucsv"]["forecast"].reindex(ml.index)
            e = ml.copy()
            e["forecast"] = 0.5 * ml["forecast"] + 0.5 * u
            out["ens:ml+ucsv"] = e.dropna(subset=["forecast"])
    others = [m for m in results if m not in BENCHMARKS and not m.startswith(("ens:", "deep:", "fm:"))]
    med = combo(others, "median")
    if med is not None:
        out["ens:median"] = med
    return out


def common_sample(results: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    dates = None
    for r in results.values():
        d = r.dropna(subset=["actual", "forecast"]).index
        dates = d if dates is None else dates.intersection(d)
    return {m: r.loc[dates] for m, r in results.items()}


def model_confidence_set(losses: pd.DataFrame, alpha: float = 0.25, n_boot: int = 2000, block: float = 12.0,
                         seed: int = 0) -> pd.Series:
    """MCS p-value of every model (T_max statistic).  Models with p >= alpha form the set."""
    L = losses.dropna().to_numpy()
    names = list(losses.columns)
    T, M = L.shape
    rng = np.random.default_rng(seed)
    idx = np.array([stationary_bootstrap_indices(T, block, rng) for _ in range(n_boot)])
    boot_means = np.stack([L[i].mean(axis=0) for i in idx])  # (B, M)
    means = L.mean(axis=0)
    alive = list(range(M))
    pvals = np.ones(M)
    running = 0.0
    while len(alive) > 1:
        a = np.array(alive)
        d = means[a] - means[a].mean()
        db = boot_means[:, a] - boot_means[:, a].mean(axis=1, keepdims=True)
        var = np.mean((db - d) ** 2, axis=0)
        var = np.where(var > 0, var, np.nan)
        t = d / np.sqrt(var)
        tmax = np.nanmax(t)
        tb = np.nanmax((db - d) / np.sqrt(var), axis=1)
        p = float(np.mean(tb >= tmax))
        running = max(running, p)
        worst = a[int(np.nanargmax(t))]
        pvals[worst] = running
        if p >= alpha:
            for i in alive:
                if i != worst:
                    pvals[i] = max(running, pvals[i]) if pvals[i] < 1 else 1.0
            break
        alive.remove(worst)
    else:
        pvals[alive[0]] = 1.0
    return pd.Series(pvals, index=names)


def conformal_intervals(r: pd.DataFrame, h: int, window: int = 120, min_errors: int = 36,
                        q: tuple[float, float] = (0.1, 0.9)) -> pd.DataFrame:
    """Rolling conformal interval from the model's past realised errors (realised by the origin)."""
    err = (r["actual"] - r["forecast"]).to_numpy()
    f = r["forecast"].to_numpy()
    pos = {p: i for i, p in enumerate(r.index)}
    lo, hi = np.full(len(r), np.nan), np.full(len(r), np.nan)
    for i, d in enumerate(r.index):
        # Errors of origins up to d - h have been realised by d.
        j = pos.get(d - h)
        if j is None:
            past = [k for k, p in enumerate(r.index) if p <= d - h]
            if not past:
                continue
            j = past[-1]
        e = err[max(0, j - window + 1) : j + 1]
        e = e[np.isfinite(e)]
        if len(e) >= min_errors:
            lo[i], hi[i] = f[i] + np.quantile(e, q[0]), f[i] + np.quantile(e, q[1])
    return pd.DataFrame({"lo": lo, "hi": hi}, index=r.index)


def interval_scores(actual, lo, hi, alpha: float = 0.2) -> dict:
    a, lo, hi = (np.asarray(v, float) for v in (actual, lo, hi))
    ok = np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
    a, lo, hi = a[ok], lo[ok], hi[ok]
    if len(a) == 0:
        return {}
    score = (hi - lo) + (2 / alpha) * (lo - a) * (a < lo) + (2 / alpha) * (a - hi) * (a > hi)
    return {"coverage": float(np.mean((a >= lo) & (a <= hi))), "width": float(np.mean(hi - lo)),
            "interval_score": float(np.mean(score)), "n": int(len(a))}


def score_table(results: dict[str, pd.DataFrame], h: int, alpha_mcs: float = 0.25, n_boot: int = 2000) -> pd.DataFrame:
    """Accuracy of every model on the common sample: RMSE ratios, DM vs random walk, MCS p-value."""
    res = common_sample({m: r for m, r in results.items() if not r.empty})
    if not res:
        return pd.DataFrame()
    any_r = next(iter(res.values()))
    a, rw = any_r["actual"], any_r["rw"]
    rmse_rw = float(np.sqrt(np.mean((a - rw) ** 2)))
    rmse_ucsv = float(np.sqrt(np.mean((a - res["ucsv"]["forecast"]) ** 2))) if "ucsv" in res else np.nan
    losses = pd.DataFrame({m: (r["actual"] - r["forecast"]) ** 2 for m, r in res.items()})
    mcs = model_confidence_set(losses, alpha=alpha_mcs, n_boot=n_boot, block=max(h, 6))
    rows = {}
    for m, r in res.items():
        e = r["actual"] - r["forecast"]
        rmse = float(np.sqrt(np.mean(e**2)))
        dm, p = diebold_mariano(a, rw, r["forecast"], h) if m != "rw" else (np.nan, np.nan)
        rows[m] = {"N": len(r), "RMSE": rmse, "rel. RW": rmse / rmse_rw, "rel. UCSV": rmse / rmse_ucsv,
                   "MAE": float(np.mean(np.abs(e))), "DM vs RW": dm, "DM p": p, "MCS p": float(mcs[m]),
                   "in MCS": bool(mcs[m] >= alpha_mcs)}
    t = pd.DataFrame(rows).T
    t.attrs.update(start=str(any_r.index.min()), end=str(any_r.index.max()), rmse_rw=rmse_rw)
    return t


def subperiod_table(results: dict[str, pd.DataFrame], h: int, periods=SUBPERIODS, min_n: int = 12) -> pd.DataFrame:
    """RMSE relative to the random walk by the month being forecast."""
    res = common_sample({m: r for m, r in results.items() if not r.empty})
    rows = {}
    for m, r in res.items():
        tm = target_month(r.index, h)
        row = {}
        for label, (s, e) in periods.items():
            mask = (tm >= pd.Period(s, "M")) & (tm <= pd.Period(e, "M"))
            x = r[mask]
            if len(x) >= min_n:
                row[label] = float(np.sqrt(np.mean((x["actual"] - x["forecast"]) ** 2) / np.mean((x["actual"] - x["rw"]) ** 2)))
            else:
                row[label] = np.nan
        rows[m] = row
    return pd.DataFrame(rows).T


def density_table(results: dict[str, pd.DataFrame], h: int, models=None) -> pd.DataFrame:
    """80% interval coverage and interval score: rolling conformal for each model, plus BART's own intervals."""
    res = common_sample({m: r for m, r in results.items() if not r.empty})
    rows = {}
    for m, r in res.items():
        if models is not None and m not in models:
            continue
        ci = conformal_intervals(r, h)
        rows[f"{m} (conformal)"] = interval_scores(r["actual"], ci["lo"], ci["hi"])
        if "q10" in r:
            ok = ci["lo"].notna()  # same evaluation months
            rows[f"{m} (posterior)"] = interval_scores(r["actual"][ok], r["q10"][ok], r["q90"][ok])
    return pd.DataFrame(rows).T
