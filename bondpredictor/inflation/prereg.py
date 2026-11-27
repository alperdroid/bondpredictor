"""Pre-registered test: the no-memory trend model against the median, both trained from 1985.

The plan, fixed before the median was trained from 1985, is ``docs/inflation_prereg_1985.md``; this module
implements it and nothing else.  ``python -m bondpredictor.inflation prereg run`` produces both arms and the
forward record, ``prereg report`` writes ``docs/inflation_prereg_1985_results.md`` and ``prereg forward``
scores the forward record against a newer data build.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd

from ..robustness import _hac_var_of_mean, _norm_cdf, diebold_mariano
from .evaluation import SUBPERIODS, add_ensembles, target_month

TRAIN_START = "1985-01"
FIRST_ORIGIN = "2000-01"
LAST_ORIGIN = "2026-11"
LAST_SEEN_PRINT = pd.Period("2026-10", "M")  # the latest CPI print in the data the selection was made on
PRIMARY_H = 12
HORIZONS = (3, 6, 12, 24)
ECONOMIES = ("US", "UK", "JP", "EA")
MEMBERS = ("ridge", "adalasso", "enet", "di", "3prf", "csr", "krr", "rf", "lgbm", "mrf", "bart", "hnn", "mlp")
ALPHA = 0.05
DATA_SHA256 = "ef6b80137d896f4db07e0ea60ee6a0fa22afe201d497544d81a440e607a926db"


def arm_specs():
    """Arm A's state spec, arm B's 13 member specs and the random walk."""
    from .backtest import model_zoo
    from .pistar import state_zoo

    zoo = {s.name: s for s in model_zoo(1)}
    members = [replace(zoo[m], train_start=TRAIN_START) for m in MEMBERS]
    a = replace({s.name: s for s in state_zoo()}["pistar_fwd_nostate"], train_start=TRAIN_START)
    return a, members, zoo["rw"]


def run(store: dict, countries=ECONOMIES, n_jobs: int = 1, progress=None, existing: dict | None = None,
        horizons=HORIZONS, start: str = FIRST_ORIGIN, end: str = LAST_ORIGIN) -> dict:
    """``{(economy, horizon, name): frame}``: name ``"A"``, ``"rw"`` or one of ``MEMBERS``."""
    from .backtest import run_jobs
    from .pistar import run_state_jobs

    a, members, rw = arm_specs()
    frames = {c: store["frames"][c] for c in countries}
    kw = dict(start=start, end=end, min_train=120)
    out = dict(existing or {})
    jobs = [((c, h, s.name), s, kw) for s in members + [rw] for c in countries for h in horizons
            if (c, h, s.name) not in out]
    if jobs:
        for key, r in run_jobs(frames, jobs, n_jobs, progress).items():
            if isinstance(r, pd.DataFrame):
                out[key] = r
    todo = [c for c in countries if not all((c, h, "A") in out for h in horizons)]
    if todo:
        for (c, h, _), r in run_state_jobs(frames, [a], todo, kw, n_jobs, progress).items():
            if h in horizons:
                out[(c, h, "A")] = r
    return out


# ------------------------------------------------------------------ arms

def arm_frame(results: dict, c: str, h: int) -> pd.DataFrame | None:
    """Origins with both arms' forecasts: columns A, B (the median), rw and actual (NaN if unpublished)."""
    mem = {m: results[(c, h, m)] for m in MEMBERS if (c, h, m) in results}
    if len(mem) < len(MEMBERS) or (c, h, "A") not in results:
        return None
    B = add_ensembles(mem)["ens:median"]
    A = results[(c, h, "A")]
    idx = A.dropna(subset=["forecast"]).index.intersection(B.dropna(subset=["forecast"]).index)
    return pd.DataFrame({"A": A.loc[idx, "forecast"], "B": B.loc[idx, "forecast"], "rw": B.loc[idx, "rw"],
                         "actual": B.loc[idx, "actual"]})


def scored(f: pd.DataFrame) -> pd.DataFrame:
    return f.dropna(subset=["A", "B", "rw", "actual"])


def pooled_test(frames: dict[str, pd.DataFrame], h: int) -> dict:
    """The primary statistic: DM on the cross-economy mean of RW-scaled squared-error differences (B - A)."""
    parts, rel = [], {}
    for c, f in frames.items():
        f = scored(f)
        if f.empty:
            continue
        s = float(np.mean((f["actual"] - f["rw"]) ** 2))
        parts.append(((f["actual"] - f["B"]) ** 2 - (f["actual"] - f["A"]) ** 2).rename(c) / s)
        rel[c] = {"A": float(np.sqrt(np.mean((f["actual"] - f["A"]) ** 2) / s)),
                  "B": float(np.sqrt(np.mean((f["actual"] - f["B"]) ** 2) / s)), "n": len(f)}
    if not parts:
        return {"dm": np.nan, "p": np.nan, "T": 0, "rel": rel}
    d = pd.concat(parts, axis=1).mean(axis=1).to_numpy()
    T = len(d)
    var = _hac_var_of_mean(d, max(h - 1, 0))
    if T < 24 or var <= 0:
        return {"dm": np.nan, "p": np.nan, "T": T, "rel": rel}
    dm = d.mean() / math.sqrt(var) * math.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 1e-12))
    return {"dm": float(dm), "p": float(2 * (1 - _norm_cdf(abs(dm)))), "T": T, "rel": rel,
            "mean_rel_A": float(np.mean([v["A"] for v in rel.values()])),
            "mean_rel_B": float(np.mean([v["B"] for v in rel.values()]))}


def verdict(test: dict) -> str:
    if not np.isfinite(test.get("dm", np.nan)):
        return "not computable"
    return "A beats B" if test["dm"] > 0 and test["p"] < ALPHA else "A does not beat B"


def holm(p: dict[str, float]) -> dict[str, float]:
    items = sorted((v, k) for k, v in p.items() if np.isfinite(v))
    out, run_max, m = {}, 0.0, len(items)
    for i, (v, k) in enumerate(items):
        run_max = max(run_max, min(1.0, (m - i) * v))
        out[k] = run_max
    return out


# ------------------------------------------------------------------ forward record

def forward_record(results: dict) -> pd.DataFrame:
    """Both arms' forecasts whose target print is after ``LAST_SEEN_PRINT`` (unpublished when selected)."""
    rows = []
    for c in ECONOMIES:
        for h in HORIZONS:
            f = arm_frame(results, c, h)
            if f is None:
                continue
            tm = target_month(f.index, h)
            keep = f[tm > LAST_SEEN_PRINT]
            for k, r in keep.iterrows():
                rows.append({"economy": c, "horizon": h, "origin": str(k), "target_print": str(k + (h - 1)),
                             "A": r["A"], "B": r["B"], "rw": r["rw"]})
    return pd.DataFrame(rows)


def score_forward(record: pd.DataFrame, store: dict) -> pd.DataFrame:
    """Pooled test per horizon on the recorded forecasts whose outcome a newer build now publishes."""
    rows = []
    for h in HORIZONS:
        frames = {}
        for c in ECONOMIES:
            r = record[(record["economy"] == c) & (record["horizon"] == h)]
            if r.empty or c not in store["frames"]:
                continue
            frame = store["frames"][c][0]
            idx = pd.PeriodIndex(r["origin"], freq="M")
            frames[c] = pd.DataFrame({"A": r["A"].to_numpy(), "B": r["B"].to_numpy(), "rw": r["rw"].to_numpy(),
                                      "actual": frame[f"tgt{h}"].reindex(idx).to_numpy()}, index=idx)
        t = pooled_test(frames, h)
        rows.append({"horizon": h, "scored origins": t["T"], "DM": t["dm"], "p": t["p"],
                     "mean rel. RMSE A": t.get("mean_rel_A", np.nan), "mean rel. RMSE B": t.get("mean_rel_B", np.nan),
                     "per economy (n)": ", ".join(f"{c} {v['n']}" for c, v in t["rel"].items())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report

def _f(x, nd=2):
    return "" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _md(df: pd.DataFrame) -> str:
    cells = df.astype(object).where(df.notna(), "")
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    return "\n".join([head, sep] + ["| " + " | ".join(str(v) for v in row) + " |" for row in cells.itertuples(index=False)])


def _rel(f, col):
    return float(np.sqrt(np.mean((f["actual"] - f[col]) ** 2) / np.mean((f["actual"] - f["rw"]) ** 2)))


def report(results: dict, path: str, reference: dict | None = None) -> dict:
    """Write the results file; returns the primary test."""
    from pathlib import Path

    names = {"US": "US", "UK": "UK", "JP": "Japan", "EA": "Euro area"}
    frames = {h: {c: f for c in ECONOMIES if (f := arm_frame(results, c, h)) is not None} for h in HORIZONS}
    primary = pooled_test(frames[PRIMARY_H], PRIMARY_H)
    L = ["# Pre-registered test: results", "",
         "Plan: [`docs/inflation_prereg_1985.md`](inflation_prereg_1985.md), committed before the median was "
         "trained from 1985.  A = the no-memory trend model, B = the median of the 13 tier-1 country models, "
         "both trained from 1985.", "",
         "## Primary comparison (12 months, four economies pooled)", "",
         f"* Diebold-Mariano statistic {_f(primary['dm'])} (positive favours A), two-sided p = {_f(primary['p'], 3)}, "
         f"{primary['T']} origins.",
         f"* Mean RMSE relative to the random walk: A {_f(primary.get('mean_rel_A'), 3)}, "
         f"B {_f(primary.get('mean_rel_B'), 3)}.",
         f"* **Verdict under the pre-registered rule (DM > 0 and p < {ALPHA}): {verdict(primary)}.**", ""]
    # per economy, 12 months
    rows, praw = [], {}
    for c, f in frames[PRIMARY_H].items():
        s = scored(f)
        t, p = diebold_mariano(s["actual"], s["B"], s["A"], PRIMARY_H)
        praw[c] = p
        rows.append({"economy": names[c], "origins": len(s), "A rel. RW": _f(_rel(s, "A")), "B rel. RW": _f(_rel(s, "B")),
                     "A / B": _f(np.sqrt(np.mean((s["actual"] - s["A"]) ** 2) / np.mean((s["actual"] - s["B"]) ** 2))),
                     "DM (+ favours A)": _f(t), "p": _f(p, 3)})
    adj = holm(praw)
    for r, c in zip(rows, frames[PRIMARY_H]):
        r["Holm p"] = _f(adj.get(c, np.nan), 3)
    L += ["## Secondary: per economy, 12 months", "", _md(pd.DataFrame(rows)), ""]
    rows = []
    for h in HORIZONS:
        t = pooled_test(frames[h], h)
        rows.append({"horizon": h, "origins": t["T"], "mean rel. RW, A": _f(t.get("mean_rel_A")),
                     "mean rel. RW, B": _f(t.get("mean_rel_B")), "DM (+ favours A)": _f(t["dm"]), "p": _f(t["p"], 3)})
    L += ["## Secondary: pooled test at every horizon", "", _md(pd.DataFrame(rows)), ""]
    rows = []
    for c, f in frames[PRIMARY_H].items():
        s = scored(f)
        tm = target_month(s.index, PRIMARY_H)
        for arm in ("A", "B"):
            row = {"economy": names[c], "arm": arm}
            for lab, (a, b) in SUBPERIODS.items():
                x = s[(tm >= pd.Period(a, "M")) & (tm <= pd.Period(b, "M"))]
                row[lab] = _f(_rel(x, arm)) if len(x) >= 12 else ""
            rows.append(row)
    L += ["## Secondary: by the month being forecast, 12 months ahead (RMSE relative to the random walk)", "",
          _md(pd.DataFrame(rows)), ""]
    us = frames[PRIMARY_H].get("US")
    if us is not None:
        w = scored(us.loc[(us.index >= pd.Period("2021-01", "M")) & (us.index <= pd.Period("2023-12", "M"))])
        L += ["US origins 2021-01 to 2023-12 (the paper's window): "
              f"A {_f(_rel(w, 'A'))}, B {_f(_rel(w, 'B'))} relative to the random walk ({len(w)} origins).", ""]
    if reference:
        rows = []
        for c, f in frames[PRIMARY_H].items():
            mem = {m: reference[(c, 12, "current", m)] for m in MEMBERS if (c, 12, "current", m) in reference}
            if len(mem) < len(MEMBERS):
                continue
            b60 = add_ensembles(mem)["ens:median"]["forecast"]
            s = scored(f)
            s = s.assign(B60=b60.reindex(s.index)).dropna(subset=["B60"])
            rows.append({"economy": names[c], "origins": len(s), "median from 1960": _f(_rel(s, "B60")),
                         "median from 1985 (B)": _f(_rel(s, "B")), "A": _f(_rel(s, "A"))})
        if rows:
            L += ["## Reference: the median trained from 1960", "",
                  "What the training window alone does to the median (same origins; RMSE relative to the random walk).",
                  "", _md(pd.DataFrame(rows)), ""]
    fwd = forward_record(results)
    if not fwd.empty:
        cnt = fwd.groupby(["economy", "horizon"]).size().unstack()
        L += ["## Forward record", "",
              f"{len(fwd)} forecasts whose target print is after {LAST_SEEN_PRINT} "
              "(`results/inflation/prereg_1985/forward.csv`), by economy and horizon:", "",
              _md(cnt.reset_index().rename(columns={h: f"{h} months" for h in HORIZONS})), "",
              "Score them after a new `build` with `python -m bondpredictor.inflation prereg forward`.", ""]
    Path(path).write_text("\n".join(L) + "\n")
    return primary
