"""Report for ``pistar run``: partial-horizon training and the persistent-state trend model.

Results are ``{(economy, horizon, set, model): frame}`` with sets ``current`` (the models as they are),
``complete`` and ``partial`` (direct models trained on horizons 1..12, see ``partial.py``) and ``state``
(the trend models of ``pistar.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..robustness import diebold_mariano
from .evaluation import SUBPERIODS, add_ensembles, common_sample, target_month
from .report import LABELS

ECONOMIES = ("US", "UK", "JP", "EA")
NAMES = {"US": "US", "UK": "UK", "JP": "Japan", "EA": "Euro area"}
STATE_LABELS = {
    "pistar_fwd": "State + forward loss, partial horizons (the paper's design)",
    "pistar_fwd_complete": "State + forward loss, complete paths only",
    "pistar_cur": "State + one-month loss",
    "pistar_fwd_nostate": "No state (theta = 0) + forward loss, partial horizons",
}
STACK_LABELS = {"current": "as is", "complete": "horizons 1-12, complete origins", "partial": "horizons 1-12 + partial origins"}
PAPER_WINDOW = ("2021-01", "2023-12")  # origins whose latest print is Dec 2020 .. Nov 2023 (paper section 7)
# Paper, Table 6 (12 quarterly SPF dates, average inflation over the next 12 months, pp): bias / RMSE.
PAPER_TABLE6 = {
    "SPF median (paper)": (4.53, 4.86, -0.06, 0.33, 3.44),
    "Deep-pi*, forward-looking, encoder (paper)": (-0.04, 2.25, -1.23, 2.25, 2.25),
    "Deep-pi*, forward-looking, attention (paper)": (0.45, 2.15, -1.15, 2.13, 2.14),
    "AR(12) (paper)": (1.46, 2.47, -1.33, 1.69, 2.12),
    "UC-SV (paper)": (1.19, 2.82, -1.93, 2.53, 2.68),
    "Random walk (paper)": (1.66, 3.46, -2.46, 3.02, 3.25),
}


def _md(df: pd.DataFrame) -> str:
    """Markdown table without the index; NaN as blank."""
    if df.empty:
        return "_(no results)_"
    cells = df.astype(object).where(df.notna(), "")
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in cells.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _label(model: str) -> str:
    return STATE_LABELS.get(model) or LABELS.get(model, model)


def _sel(results, c, h, st):
    return {m: r for (cc, hh, s, m), r in results.items() if (cc, hh, s) == (c, h, st)
            and isinstance(r, pd.DataFrame) and not r.empty}


def _rmse(r):
    e = r["actual"] - r["forecast"]
    return float(np.sqrt(np.mean(e * e)))


def _rel(r):
    return _rmse(r) / float(np.sqrt(np.mean((r["actual"] - r["rw"]) ** 2)))


def _window(r, lo, hi):
    return r.loc[(r.index >= pd.Period(lo, "M")) & (r.index <= pd.Period(hi, "M"))]


def _dm(a, b, h):
    """DM (HLN) of ``b`` against ``a`` on their common months: positive means ``b`` is more accurate."""
    j = a.dropna(subset=["actual", "forecast"]).index.intersection(b.dropna(subset=["forecast"]).index)
    return diebold_mariano(a.loc[j, "actual"], a.loc[j, "forecast"], b.loc[j, "forecast"], h)


def _with_median(results, c):
    """Tier-1 country models at 12 months plus the repository's fixed-in-advance ensembles."""
    cur = _sel(results, c, 12, "current")
    return add_ensembles(cur) if cur else {}


def _f(x, nd=2):
    return "" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _star(p):
    return "" if not np.isfinite(p) else "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ------------------------------------------------------------------ tables

def stacked_table(results) -> pd.DataFrame:
    """12-month RMSE relative to the random walk: each stacked model as is, complete and partial."""
    rows = []
    models = sorted({m for (_, h, s, m) in results if s == "partial" and h == 12})
    for m in models:
        for st in ("current", "complete", "partial"):
            row = {"model": f"{_label(m)}: {STACK_LABELS[st]}"}
            rels = []
            for c in ECONOMIES:
                trio = {s: results.get((c, 12, s, m)) for s in ("current", "complete", "partial")}
                if any(v is None or v.empty for v in trio.values()):
                    continue
                trio = common_sample(trio)
                rel = _rel(trio[st])
                rels.append(rel)
                cell = _f(rel)
                if st != "current":
                    t, p = _dm(trio["current"], trio[st], 12)
                    cell += _star(p) if t > 0 else ("(-)" + _star(p) if _star(p) else "")
                row[NAMES[c]] = cell
            row["mean"] = _f(np.mean(rels)) if rels else ""
            rows.append(row)
    return pd.DataFrame(rows)


def state_table(results, h: int) -> pd.DataFrame:
    """RMSE relative to the random walk at horizon ``h``: state models, benchmarks (and the median at 12)."""
    rows = {}
    for c in ECONOMIES:
        res = {m: r for m, r in _sel(results, c, h, "current").items() if m in ("rw", "ucsv", "ar")}
        if h == 12:
            ens = _with_median(results, c)
            for m in ("ens:median", "mlp", "lgbm"):
                if m in ens:
                    res[m] = ens[m]
        res.update(_sel(results, c, h, "state"))
        if "rw" not in res:
            continue
        res = common_sample(res)
        for m, r in res.items():
            rows.setdefault(m, {})[NAMES[c]] = _rel(r)
            if m != "rw":
                t, p = _dm(res["rw"], r, h)
                rows[m][NAMES[c] + " p"] = p
    out = []
    order = ["rw", "ucsv", "ar", "ens:median", "mlp", "lgbm"] + list(STATE_LABELS)
    for m in [m for m in order if m in rows]:
        rr = rows[m]
        vals = [rr.get(NAMES[c]) for c in ECONOMIES if NAMES[c] in rr]
        out.append({"model": _label(m), **{NAMES[c]: _f(rr.get(NAMES[c])) + _star(rr.get(NAMES[c] + " p", np.nan))
                                           for c in ECONOMIES if NAMES[c] in rr},
                    "mean": _f(np.mean(vals)) if vals else ""})
    return pd.DataFrame(out)


def from1985_table(results, h: int = 12) -> pd.DataFrame:
    """State models trained from 1985 against the same models trained from 1960, 12 months ahead."""
    rows = {}
    for c in ECONOMIES:
        res = {m: r for m, r in _with_median(results, c).items() if m in ("rw", "ucsv", "ens:median")}
        res.update({f"1960:{m}": r for m, r in _sel(results, c, h, "state").items()})
        res.update({f"1985:{m}": r for m, r in _sel(results, c, h, "state85").items()})
        if "rw" not in res or not any(k.startswith("1985:") for k in res):
            continue
        res = common_sample(res)
        for m, r in res.items():
            if m != "rw":
                rows.setdefault(m, {})[NAMES[c]] = _rel(r)
    out = []
    for m in ["ucsv", "ens:median"] + [f"{y}:{n}" for n in STATE_LABELS for y in ("1960", "1985")]:
        if m in rows:
            lab = _label(m) if ":" not in m or m.startswith("ens:") else f"{_label(m.split(':')[1])}, trained from {m[:4]}"
            vals = list(rows[m].values())
            out.append({"model": lab, **{k: _f(v) for k, v in rows[m].items()}, "mean": _f(np.mean(vals))})
    return pd.DataFrame(out)


def decomposition_table(results) -> pd.DataFrame:
    """What each ingredient adds, 12 months ahead: RMSE ratio and DM test of one variant against another."""
    pairs = [("pistar_fwd", "pistar_fwd_complete", "partial horizons (vs complete paths only)"),
             ("pistar_fwd", "pistar_fwd_nostate", "persistent state (vs theta = 0)"),
             ("pistar_fwd", "pistar_cur", "forward loss (vs one-month loss)")]
    rows = []
    for a, b, label in pairs:
        row = {"ingredient": label}
        for c in ECONOMIES:
            ra, rb = results.get((c, 12, "state", a)), results.get((c, 12, "state", b))
            if ra is None or rb is None or ra.empty or rb.empty:
                continue
            both = common_sample({"a": ra, "b": rb})
            t, p = _dm(both["b"], both["a"], 12)
            row[NAMES[c]] = f"{_rmse(both['a']) / _rmse(both['b']):.2f} (p {p:.2f})"
        rows.append(row)
    return pd.DataFrame(rows)


def subperiod_rel(results, h: int = 12) -> pd.DataFrame:
    """RMSE relative to the random walk by the month being forecast, 12 months ahead, every economy."""
    rows = []
    for c in ECONOMIES:
        res = {m: r for m, r in _with_median(results, c).items() if m in ("rw", "ucsv", "ar", "ens:median")}
        res.update(_sel(results, c, h, "state"))
        for m in sorted({m for (cc, hh, s, m) in results if cc == c and hh == h and s == "partial"}):
            res[f"partial:{m}"] = results[(c, h, "partial", m)]
            res[f"current:{m}"] = results[(c, h, "current", m)]
        if "rw" not in res:
            continue
        res = common_sample(res)
        for m, r in res.items():
            if m == "rw":
                continue
            tm = target_month(r.index, h)
            if ":" in m and not m.startswith("ens:"):
                st, name = m.split(":", 1)
                lab = f"{_label(name)} ({STACK_LABELS[st]})"
            else:
                lab = _label(m)
            row = {"economy": NAMES[c], "model": lab}
            for lab, (s, e) in SUBPERIODS.items():
                x = r[(tm >= pd.Period(s, "M")) & (tm <= pd.Period(e, "M"))]
                row[lab] = _f(_rel(x)) if len(x) >= 12 else ""
            rows.append(row)
    return pd.DataFrame(rows)


def paper_window_table(results, c: str = "US") -> pd.DataFrame:
    """The paper's design: origins Jan 2021 - Dec 2023, 12-month average inflation; RMSE relative to the random walk."""
    res = {m: r for m, r in _with_median(results, c).items() if m in ("rw", "ucsv", "ar", "ens:median", "enet", "ridge",
                                                                        "mlp", "hnn", "lgbm")}
    res.update({f"state:{m}": r for m, r in _sel(results, c, 12, "state").items()})
    for st in ("complete", "partial"):
        res.update({f"{st}:{m}": r for m, r in _sel(results, c, 12, st).items()})
    res = {m: _window(r, *PAPER_WINDOW) for m, r in res.items()}
    res = common_sample({m: r for m, r in res.items() if not r.empty})
    q = pd.PeriodIndex([pd.Period(f"{y}-{mm:02d}", "M") for y in (2021, 2022, 2023) for mm in (2, 5, 8, 11)])
    rows = []
    for m, r in res.items():
        st, name = m.split(":", 1) if ":" in m and not m.startswith("ens:") else ("current", m)
        lab = _label(name) + ("" if st in ("current", "state") else f" ({STACK_LABELS[st]})")
        e = r["actual"] - r["forecast"]
        eq = e.reindex(q).dropna()
        rise, fall = eq.iloc[:6], eq.iloc[6:]
        rows.append({"model": lab, "rel. RW (36 origins)": _f(_rel(r)), "RMSE (36)": _f(_rmse(r)),
                     "rise bias / RMSE": f"{rise.mean():+.2f} / {np.sqrt((rise**2).mean()):.2f}",
                     "fall bias / RMSE": f"{fall.mean():+.2f} / {np.sqrt((fall**2).mean()):.2f}",
                     "RMSE (12 SPF dates)": _f(float(np.sqrt((eq**2).mean())))})
    for lab, (rb, rr, fb, fr, allr) in PAPER_TABLE6.items():
        rows.append({"model": lab, "rel. RW (36 origins)": "", "RMSE (36)": "",
                     "rise bias / RMSE": f"{rb:+.2f} / {rr:.2f}", "fall bias / RMSE": f"{fb:+.2f} / {fr:.2f}",
                     "RMSE (12 SPF dates)": f"{allr:.2f}"})
    return pd.DataFrame(rows)


def path_table(results, c: str = "US") -> pd.DataFrame:
    """Real-time 12-month forecasts at the paper's SPF dates."""
    q = pd.PeriodIndex([pd.Period(f"{y}-{mm:02d}", "M") for y in (2021, 2022) for mm in (2, 5, 8, 11)])
    cols = {}
    for key, lab in (((c, 12, "current", "rw"), "random walk"), ((c, 12, "current", "ucsv"), "UCSV"),
                     ((c, 12, "state", "pistar_fwd_complete"), "state, complete"),
                     ((c, 12, "state", "pistar_fwd"), "state, partial"),
                     ((c, 12, "state", "pistar_fwd_nostate"), "no state, partial"),
                     ((c, 12, "current", "ridge"), "ridge"), ((c, 12, "partial", "ridge"), "ridge, partial")):
        if key in results and not results[key].empty:
            cols[lab] = results[key]["forecast"].reindex(q)
    ens = _with_median(results, c)
    if "ens:median" in ens:
        cols["your median"] = ens["ens:median"]["forecast"].reindex(q)
    t = pd.DataFrame(cols)
    t["outcome"] = results[(c, 12, "current", "rw")]["actual"].reindex(q)
    t.index = [f"{p - 1} print" for p in q]
    return t.round(1)


def drivers_table(results, c: str = "US", model: str = "pistar_fwd") -> pd.DataFrame:
    """Change of the trend since January 2019 by predictor group, estimated in real time at each origin."""
    r = results.get((c, 12, "state", model))
    if r is None or r.empty:
        return pd.DataFrame()
    dates = [p for p in ("2020-06", "2021-01", "2021-03", "2021-06", "2021-12", "2022-06", "2022-12", "2023-06")
             if pd.Period(p, "M") in r.index]
    cols = sorted([k for k in r.columns if k.startswith("d19_")])
    t = r.loc[[pd.Period(p, "M") for p in dates], ["pistar"] + cols].copy()
    names = {"base": "initial state and alpha", "own": "own (core, components)"}
    t.columns = ["trend level"] + [names.get(k[4:], k[4:]) for k in cols]
    t["total change"] = t.iloc[:, 1:].sum(axis=1)
    t.index = [str(p) for p in t.index]
    return t.round(2)


def median_table(results) -> pd.DataFrame:
    """Does the state model add to the repository's median combination?  12 months ahead, rel. random walk."""
    rows = {}
    for c in ECONOMIES:
        ens = _with_median(results, c)
        st = results.get((c, 12, "state", "pistar_fwd"))
        if "ens:median" not in ens or st is None or st.empty:
            continue
        members = [m for m in ens if m not in ("rw", "ucsv", "ar") and not m.startswith("ens:")]
        F = pd.concat({m: ens[m]["forecast"] for m in members}, axis=1)
        base = ens["rw"][["rw", "actual"]].copy()
        with_state = base.copy()
        with_state["forecast"] = pd.concat([F, st["forecast"].rename("pistar")], axis=1).median(axis=1)
        half = base.copy()
        half["forecast"] = 0.5 * ens["ens:median"]["forecast"] + 0.5 * st["forecast"]
        res = common_sample({"Median of all models (as is)": ens["ens:median"],
                             "Median including the state model": with_state,
                             "1/2 median + 1/2 state model": half, "State model alone": st, "rw": ens["rw"]})
        for m, r in res.items():
            if m == "rw":
                continue
            first_ = "Median of all models (as is)"
            t, p = _dm(res[first_], r, 12) if m != first_ else (np.nan, np.nan)
            rows.setdefault(m, {})[NAMES[c]] = _f(_rel(r)) + (_star(p) if np.isfinite(t) and t > 0 else "")
            w = {mm: _window(rr, *PAPER_WINDOW) for mm, rr in res.items()}
            if c == "US":
                rows[m]["US, 2021-23 origins"] = _f(_rel(w[m]))
    return pd.DataFrame(rows).T.rename_axis("combination").reset_index()


# ------------------------------------------------------------------ document

def write_report(results: dict, path: str, summary: str | None = None) -> None:
    from pathlib import Path

    from .pistar import LAMBDAS, THETA

    first = min(r.index.min() for r in results.values() if isinstance(r, pd.DataFrame) and not r.empty)
    last = max(r.index.max() for r in results.values() if isinstance(r, pd.DataFrame) and not r.empty)
    L = ["# Partial horizons and a persistent trend state (ECB Working Paper 3307)", "",
         "Two ideas from Montes-Galdón (2026), *Deep-π\\*: a neural network estimate of trend inflation*, "
         "ECB Working Paper 3307, tested on this repository's real-time inflation backtest "
         "(`python -m bondpredictor.inflation pistar run`, then `pistar report`).", "",
         f"Forecast origins {first} to {last}; every model re-estimated at each origin on data published by then. "
         "RMSE is relative to the random walk (below 1 beats \"no change\"); stars mark Harvey-Leybourne-Newbold "
         "Diebold-Mariano p-values below 0.10/0.05/0.01 (`(-)` when the variant is the less accurate one).", ""]
    if summary:
        L += [summary.strip(), ""]
    L += ["## What was tested", "",
          "**Partial-horizon training** (`partial.py`).  A direct 12-month model normally learns only from origins "
          "whose 12-month outcome is published, so in February 2021 its newest training row was February 2020.  The "
          "stacked version trains on horizons 1-12 at once, each origin entering with every horizon already out, on a "
          "common scale (average inflation over the next *m* months minus today's y/y rate, which is exactly the usual "
          "target at *m* = 12), weighted by *m*/12.  *Complete* uses the shorter horizons only for origins whose "
          "12 months are all out; *partial* adds the origins of the last 11 months, scored on the months already "
          "published, as in the paper.  The gap between the two is the effect of the partial rows alone.", "",
          f"**Persistent-state trend** (`pistar.py`).  Monthly seasonally adjusted inflation follows an AR(12) around "
          f"a trend π\\*[k] = θ π\\*[k-1] + (1-θ)(α + x[k]'β), the paper's recursion with a linear innovation instead "
          f"of its neural network; θ = {THETA} (half-life 138 months), x = every predictor except headline CPI "
          "terms.  The trend is an exponentially weighted average of past predictors; forecasts iterate the AR with "
          "the trend decaying towards α.  The forward-looking loss scores the iterated forecasts 1-12 months ahead "
          "from every training origin (with or without partial paths), the one-month loss only the next month.  "
          f"θ = 0 removes the memory.  The ridge penalty on β (grid {', '.join(f'{x:g}' for x in LAMBDAS)}) is chosen "
          "every 12 months by expanding-window validation on the model's own loss.  Not reproduced: the paper's "
          "neural network and attention layer, time-varying volatility and Student-t errors.", ""]
    L += ["## Partial-horizon training, 12 months ahead", "",
          "Stars compare each stacked version with the same model as it is.", "", _md(stacked_table(results)), ""]
    for h in (12, 3, 6, 24):
        t = state_table(results, h)
        if not t.empty:
            L += [f"## Persistent-state model, {h} months ahead", "",
                  "Stars: DM test against the random walk." + (
                      " `Median of all models` is the repository's fixed-in-advance median of the 13 tier-1 country "
                      "models." if h == 12 else ""), "", _md(t), ""]
    L += ["## Which ingredient does the work (12 months ahead)", "",
          "RMSE of the paper's design relative to the variant without one ingredient (below 1: the ingredient "
          "helps), with the DM p-value.", "", _md(decomposition_table(results)), ""]
    t85 = from1985_table(results)
    if not t85.empty:
        L += ["## Robustness: state models trained from 1985", "",
              "The repository's robustness suite also trains from 1985 (`robustness.py`).  Run after the main "
              "results, as a diagnostic of the long training sample.  Same origins; RMSE relative to the random walk.",
              "", _md(t85), ""]
    L += ["## Adding the state model to the median", "", _md(median_table(results)), ""]
    L += ["## By the month being forecast, 12 months ahead", "", _md(subperiod_rel(results)), ""]
    L += ["## The paper's test window (US)", "",
          "Origins January 2021 to December 2023, whose latest print runs from December 2020 to November 2023, as "
          "in the paper's section 7.  The y/y rate 12 months after the latest print is the paper's \"average "
          "inflation over the next twelve months\".  Rise and fall use the paper's 12 quarterly SPF dates (latest "
          "print January, April, July, October; rise = 2021Q1-2022Q2); bias is outcome minus forecast.  The paper's "
          "own numbers (its Table 6) are listed for reference.", "", _md(paper_window_table(results)), "",
          "Real-time forecasts of y/y inflation 12 months after the print shown:", "",
          _md(path_table(results).reset_index().rename(columns={"index": "latest print"})), ""]
    d = drivers_table(results)
    if not d.empty:
        L += ["## What moved the US trend in real time", "",
              "Change of the trend (pp, annualised) since January 2019 by predictor group, from the fit at each "
              "origin (the trend is linear, so the split is exact).", "",
              _md(d.reset_index().rename(columns={"index": "origin"})), ""]
    Path(path).write_text("\n".join(L) + "\n")
