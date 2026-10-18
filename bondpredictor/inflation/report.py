"""Markdown report of the inflation horse race."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import (NONLINEAR_ML, SUBPERIODS, add_ensembles, common_sample, density_table, score_table,
                         subperiod_table)

ORDER = ["rw", "ucsv", "ar", "ridge", "adalasso", "enet", "di", "3prf", "csr", "krr", "rf", "lgbm", "mrf", "bart",
         "hnn", "mlp", "panel:ridge", "panel:lgbm", "panel:mrf", "panel:bart", "panel:hnn",
         "ens:ml", "ens:ml+ucsv", "ens:median", "fm:chronos_bolt", "fm:chronos2", "fm:chronos2_cov", "fm:timesfm25",
         "fm:timesfm3", "fm:timesfm3_cov", "deep:nhits", "deep:patchtst", "deep:tide", "deep:lstm"]
LABELS = {
    "rw": "Random walk (no change)", "ucsv": "UCSV (Stock-Watson)", "ar": "AR-type, own inflation only",
    "ridge": "Ridge", "adalasso": "Adaptive lasso", "enet": "Elastic net", "di": "Diffusion index (PCA)",
    "3prf": "Three-pass regression filter", "csr": "Complete subset regression", "krr": "Kernel ridge (RBF)",
    "rf": "Random forest", "lgbm": "Gradient boosting (LightGBM)", "mrf": "Macroeconomic random forest",
    "bart": "BART", "hnn": "Hemisphere neural net", "mlp": "Plain neural net (MLP)",
    "panel:ridge": "Pooled ridge", "panel:lgbm": "Pooled boosting", "panel:mrf": "Pooled MRF",
    "panel:bart": "Pooled BART", "panel:hnn": "Pooled hemisphere net",
    "ens:ml": "Mean of non-linear ML", "ens:ml+ucsv": "½ non-linear ML + ½ UCSV", "ens:median": "Median of all models",
    "deep:nhits": "N-HiTS (tier 2)", "deep:patchtst": "PatchTST (tier 2)", "deep:tide": "TiDE (tier 2)",
    "deep:lstm": "LSTM (tier 2)",
    "fm:chronos_bolt": "Chronos-Bolt (zero-shot)", "fm:chronos2": "Chronos-2 (zero-shot)",
    "fm:chronos2_cov": "Chronos-2 + macro covariates", "fm:timesfm25": "TimesFM 2.5 (zero-shot)",
    "fm:timesfm3": "TimesFM 3.0 (zero-shot)", "fm:timesfm3_cov": "TimesFM 3.0 + macro covariates",
}
NAMES = {"US": "United States", "UK": "United Kingdom", "JP": "Japan", "EA": "Euro area"}


def _order(models):
    known = [m for m in ORDER if m in models]
    return known + sorted(m for m in models if m not in ORDER)


def _cell(v, kind="2"):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else ""
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        if kind == "p":
            return f"{v:.3f}"
        if kind == "i":
            return f"{v:.0f}"
        if kind == "s":
            return f"{v:+.2f}"
        return f"{v:.2f}"
    return str(v).replace("|", "\\|")


def md_table(df: pd.DataFrame, kinds: dict | None = None, index_name: str = "") -> str:
    kinds = kinds or {}
    head = "| " + " | ".join([index_name] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = []
    for i, row in df.iterrows():
        cells = [_cell(v, kinds.get(c, "2")) if not isinstance(v, str) else v for c, v in row.items()]
        body.append("| " + " | ".join([str(i).replace("|", "\\|")] + cells) + " |")
    return "\n".join([head, sep, *body])


def collect(results: dict, country: str, h: int) -> dict[str, pd.DataFrame]:
    res = {m: r for (c, hh, m), r in results.items() if c == country and hh == h and isinstance(r, pd.DataFrame)
           and not r.empty}
    return add_ensembles(res)


def headline(results, countries, h, n_boot):
    """Relative RMSE per economy; '*' marks members of the 75% model confidence set."""
    cols, long = {}, []
    for c in countries:
        res = collect(results, c, h)
        if not res:
            continue
        t = score_table(res, h, n_boot=n_boot)
        cols[c] = t
        for m, row in t.iterrows():
            long.append({"country": c, "horizon": h, "model": m, **row.to_dict(),
                         "start": t.attrs["start"], "end": t.attrs["end"]})
    models = _order(set().union(*[set(t.index) for t in cols.values()])) if cols else []
    out = pd.DataFrame(index=[LABELS.get(m, m) for m in models])
    rel = pd.DataFrame({c: t["rel. RW"] for c, t in cols.items()}).reindex(models)
    for c, t in cols.items():
        best = t["rel. RW"].min()
        vals = []
        for m in models:
            if m not in t.index:
                vals.append("")
                continue
            v = t.loc[m, "rel. RW"]
            s = f"{v:.2f}" + ("*" if t.loc[m, "in MCS"] else "")
            vals.append(f"**{s}**" if np.isclose(v, best) else s)
        out[f"{c} ({t.attrs['start'][:4]}–)"] = vals
    out["mean"] = [f"{v:.2f}" if np.isfinite(v) else "" for v in rel.mean(axis=1, skipna=False)]
    return out, pd.DataFrame(long)


def _variant_table(robust: dict, results: dict, country: str, h: int) -> pd.DataFrame:
    rows = {}
    for (c, hh, name), r in sorted(robust.items(), key=lambda kv: kv[0][2]):
        if c != country or hh != h or not isinstance(r, pd.DataFrame) or r.empty:
            continue
        base = name.split(" | ")[0]
        b = results.get((c, h, base))
        if b is None:
            continue
        both = common_sample({"v": r, "b": b})
        v, bb = both["v"], both["b"]
        if len(v) < 24:
            continue
        rmse_rw = np.sqrt(np.mean((v["actual"] - v["rw"]) ** 2))
        rows[name] = {"baseline": float(np.sqrt(np.mean((bb["actual"] - bb["forecast"]) ** 2)) / rmse_rw),
                      "variant": float(np.sqrt(np.mean((v["actual"] - v["forecast"]) ** 2)) / rmse_rw),
                      "corr. with baseline": float(np.corrcoef(v["forecast"], bb["forecast"])[0, 1])}
    return pd.DataFrame(rows).T


def _contributions(results, country, h):
    r = results.get((country, h, "hnn"))
    if r is None or r.empty:
        return None
    cols = [c for c in r.columns if c.startswith("c_")]
    periods = {"2000-07": ("2000-01", "2007-12"), "2008-19": ("2008-01", "2019-12"), "2020": ("2020-01", "2020-12"),
               "2021": ("2021-01", "2021-12"), "2022": ("2022-01", "2022-12"), "2023-": ("2023-01", "2100-12")}
    rows = {}
    for lab, (a, b) in periods.items():
        x = r.loc[pd.Period(a, "M") : pd.Period(b, "M"), cols]
        if len(x):
            rows[lab] = x.mean()
    t = pd.DataFrame(rows).T
    t.columns = [c[2:] for c in cols]
    return t


def _gtvp(results, country, h):
    r = results.get((country, h, "mrf"))
    if r is None or r.empty:
        return None
    cols = [c for c in r.columns if c.startswith("b_")]
    periods = {"2000-07": ("2000-01", "2007-12"), "2008-19": ("2008-01", "2019-12"),
               "2020-22": ("2020-01", "2022-12"), "2023-": ("2023-01", "2100-12")}
    t = pd.DataFrame({lab: r.loc[pd.Period(a, "M") : pd.Period(b, "M"), cols].mean() for lab, (a, b) in periods.items()}).T
    t.columns = [c[2:] for c in cols]
    return t


def _post_release(results, countries, horizons=(3, 6)) -> pd.DataFrame:
    """Foundation models scored only on origins after their release (pooled over economies)."""
    from .foundation import MODELS

    rows = {}
    for name, spec in MODELS.items():
        start = pd.Period(spec["released"], "M") + 1
        for h in horizons:
            e_m = e_rw = e_u = 0.0
            n = 0
            for c in countries:
                r, u = results.get((c, h, f"fm:{name}")), results.get((c, h, "ucsv"))
                if r is None or u is None:
                    continue
                x = r.loc[start:].dropna(subset=["actual"]).join(u["forecast"].rename("ucsv"), how="inner").dropna()
                e_m += float(((x["actual"] - x["forecast"]) ** 2).sum())
                e_rw += float(((x["actual"] - x["rw"]) ** 2).sum())
                e_u += float(((x["actual"] - x["ucsv"]) ** 2).sum())
                n += len(x)
            if n:
                rows[(LABELS.get(f"fm:{name}", name), h)] = {"origins from": str(start), "forecasts": n,
                                                             "model rel. RW": np.sqrt(e_m / e_rw),
                                                             "UCSV rel. RW": np.sqrt(e_u / e_rw)}
    t = pd.DataFrame(rows).T
    if not t.empty:
        t.index = [f"{m}, h={h}" for m, h in t.index]
    return t


def build_report(results: dict, robust: dict, store: dict, countries, horizons, latest: pd.DataFrame | None,
                 n_boot: int = 2000) -> tuple[str, pd.DataFrame]:
    L = ["<!-- Generated by `python -m bondpredictor.inflation report`. -->", "",
         "# Inflation forecasts with econometric machine learning", ""]
    L += ["Target: year-on-year CPI inflation (HICP for the euro area) *h* months after the latest published "
          "print.  Forecast origins are month-ends from January 2000 (later for the euro area, whose data start "
          "in 1990), each using only data published by then.  Expanding training window from 1960 where data "
          "exist; at least ten years of training data.  RMSE is relative to the random walk "
          "(no change in the y/y rate); below 1 beats it.  `*` marks the 75% model confidence set "
          "(Hansen-Lunde-Nason, T_max, 2,000 stationary-bootstrap draws); bold is the lowest in each column.  "
          "All models of one economy and horizon are scored on the same months.", ""]
    long_all = []
    for h in horizons:
        tab, long = headline(results, countries, h, n_boot)
        long_all.append(long)
        L += [f"## {h} months ahead: RMSE relative to the random walk", "", md_table(tab, index_name="model"), ""]
    main_h = 12 if 12 in horizons else horizons[0]
    L += [f"## Detail by economy, {main_h} months ahead", ""]
    for c in countries:
        res = collect(results, c, main_h)
        if not res:
            continue
        t = score_table(res, main_h, n_boot=n_boot).reindex(_order(res))
        t.index = [LABELS.get(m, m) for m in t.index]
        t = t[["RMSE", "rel. RW", "rel. UCSV", "MAE", "DM vs RW", "DM p", "MCS p", "in MCS"]]
        L += [f"### {NAMES.get(c, c)} ({t.attrs.get('start', '')} – {t.attrs.get('end', '')} origins)", "",
              "DM > 0: more accurate than the random walk (Harvey-Leybourne-Newbold, two-sided p).", "",
              md_table(t, {"DM vs RW": "s", "DM p": "p", "MCS p": "p"}, "model"), ""]
        sub = subperiod_table(res, main_h).reindex(_order(res))
        sub.index = [LABELS.get(m, m) for m in sub.index]
        L += ["RMSE relative to the random walk by the month being forecast:", "", md_table(sub, index_name="model"), ""]
    pr = _post_release(results, countries)
    if not pr.empty:
        L += ["## Foundation models: pretraining leakage check", "",
              "Chronos and TimesFM were pretrained on public corpora that may contain these CPI series up to each "
              "model's release, so their scores above are not clean out-of-sample evidence.  Origins after release "
              "are; there are few of them, pooled here over the four economies.", "",
              md_table(pr, {"forecasts": "i"}, "model, horizon"), ""]
    L += [f"## Interval forecasts, {main_h} months ahead", "",
          "80% intervals.  *Conformal*: the model's forecast plus the 10th and 90th percentiles of its own errors "
          "over the previous ten years, using only errors realised by the origin.  *Posterior*: BART's "
          "predictive distribution, or a foundation model's own 10% and 90% quantiles.  Interval score "
          "(Gneiting-Raftery, lower is better) penalises width and misses.", ""]
    for c in countries:
        res = collect(results, c, main_h)
        if not res:
            continue
        d = density_table(res, main_h, models=["rw", "ucsv", "ridge", "rf", "lgbm", "mrf", "bart", "hnn",
                                                "panel:bart", "ens:ml", "ens:ml+ucsv", "ens:median",
                                                "fm:chronos2_cov", "fm:timesfm3_cov"])
        if not d.empty:
            rank = {m: i for i, m in enumerate(_order(res))}
            d = d.loc[sorted(d.index, key=lambda i: (rank.get(i.split(" (")[0], 999), i))]
            d.index = [f"{LABELS.get(i.split(' (')[0], i.split(' (')[0])} ({i.split(' (')[1]}" for i in d.index]
            L += [f"### {NAMES.get(c, c)}", "", md_table(d[["coverage", "width", "interval_score", "n"]], {"n": "i"}, "model"), ""]
    for rh in sorted({k[1] for k in robust}, key=lambda h: (h != main_h, h)):
        L += [f"## Robustness of the econometric ML models, {rh} months ahead", "",
              "Each variant re-runs one model with one change, scored with its baseline on the same months "
              "(RMSE relative to the random walk).", ""]
        for c in countries:
            v = _variant_table(robust, results, c, rh)
            if not v.empty:
                L += [f"### {NAMES.get(c, c)}", "", md_table(v, index_name="variant"), ""]
    L += [f"## What drives the forecasts ({main_h} months ahead)", ""]
    for c in countries:
        t = _contributions(results, c, main_h)
        g = _gtvp(results, c, main_h)
        if t is None and g is None:
            continue
        L += [f"### {NAMES.get(c, c)}", ""]
        if t is not None:
            L += ["Hemisphere net: average contribution of each block to the forecast change in y/y inflation (pp), "
                  "by forecast origin:", "", md_table(t, {k: "s" for k in t.columns}, "origins"), ""]
        if g is not None:
            L += ["Macroeconomic random forest: average time-varying coefficients (pp of forecast change per "
                  "standard deviation of each term):", "", md_table(g, {k: "s" for k in g.columns}, "origins"), ""]
    if latest is not None and not latest.empty:
        L += ["## Latest forecasts", "",
              "Every model refitted on all realised data and applied at each economy's latest CPI print.", ""]
        show = ["rw", "ucsv", "ridge", "csr", "krr", "lgbm", "mrf", "hnn", "mlp", "ens:ml", "ens:ml+ucsv", "ens:median"]
        for c in countries:
            x = latest[latest["country"] == c]
            if x.empty:
                continue
            piv = x.pivot_table(index="model", columns="horizon", values="forecast").reindex([m for m in show if m in set(x["model"])])
            piv.columns = [f"h={h} ({x[x.horizon == h]['target_month'].iloc[0]})" for h in piv.columns]
            piv.index = [LABELS.get(m, m) for m in piv.index]
            hs = sorted(x["horizon"].unique())
            fm = {h: results.get((c, h, "fm:timesfm3_cov")) for h in hs}
            if all(f is not None and not f.empty for f in fm.values()):
                last = {h: f.iloc[-1] for h, f in fm.items()}
                piv.loc[LABELS["fm:timesfm3_cov"]] = [last[h]["forecast"] for h in hs]
                piv.loc["TimesFM 3.0 + covariates, 80% band"] = [f"{last[h]['q10']:.1f} to {last[h]['q90']:.1f}" for h in hs]
            L += [f"### {NAMES.get(c, c)}: latest print {x['latest_print'].iloc[0]}, y/y {x['latest_yoy'].iloc[0]:.2f}%", "",
                  md_table(piv, index_name="model"), ""]
    tables = pd.concat(long_all, ignore_index=True) if long_all else pd.DataFrame()
    return "\n".join(L), tables
