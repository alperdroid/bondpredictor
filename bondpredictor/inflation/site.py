"""Data for the interactive site (``docs/site``).

``site_data`` gathers everything the page shows into one JSON-ready dict:
inflation history, the latest forecasts with 80% bands, the backtest paths,
the accuracy league table, sub-period scores, driver coefficients and the
scenario responses.  ``render`` drops it into the page template.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import SUBPERIODS, common_sample, subperiod_table, target_month
from .report import LABELS, NAMES, ORDER, collect

ECONOMIES = ("US", "UK", "JP", "EA")
HORIZONS = (3, 6, 12, 24)
INDEX = {"US": "CPI", "UK": "CPI", "JP": "CPI", "EA": "HICP"}

SHORT = {
    "rw": "No change", "ucsv": "UCSV trend", "ar": "AR benchmark", "ridge": "Ridge", "adalasso": "Adaptive lasso",
    "enet": "Elastic net", "di": "Factor model (PCA)", "3prf": "Three-pass filter", "csr": "Subset regression",
    "krr": "Kernel ridge", "rf": "Random forest", "lgbm": "Gradient boosting", "mrf": "Macro random forest",
    "bart": "BART", "hnn": "Hemisphere net", "mlp": "Neural net (MLP)", "panel:ridge": "Pooled ridge",
    "panel:lgbm": "Pooled boosting", "panel:mrf": "Pooled MRF", "panel:bart": "Pooled BART",
    "panel:hnn": "Pooled hemisphere net", "ens:ml": "Mean of ML", "ens:ml+ucsv": "Half ML, half UCSV",
    "ens:median": "Median of models", "fm:chronos_bolt": "Chronos-Bolt", "fm:chronos2": "Chronos-2",
    "fm:chronos2_cov": "Chronos-2 + macro", "fm:timesfm25": "TimesFM 2.5", "fm:timesfm3": "TimesFM 3.0",
    "fm:timesfm3_cov": "TimesFM 3.0 + macro", "deep:nhits": "N-HiTS", "deep:patchtst": "PatchTST", "deep:tide": "TiDE",
    "deep:lstm": "LSTM",
}


def family(m: str) -> str:
    if m in ("rw", "ucsv", "ar"):
        return "benchmark"
    if m in ("ridge", "adalasso", "enet", "di", "3prf", "csr", "panel:ridge"):
        return "linear"
    if m.startswith("ens:"):
        return "ensemble"
    if m.startswith("fm:"):
        return "foundation"
    if m.startswith("deep:"):
        return "deep"
    return "nonlinear"


# Models whose backtest paths the page can replay.
PATH_MODELS = ("ucsv", "ridge", "adalasso", "csr", "krr", "rf", "lgbm", "mrf", "bart", "hnn", "mlp", "panel:ridge",
               "ens:median", "ens:ml+ucsv", "fm:timesfm3_cov", "fm:chronos2_cov", "deep:tide")
# Models listed beside the median in the latest-forecast table.
LATEST_MODELS = ("ucsv", "ridge", "csr", "krr", "lgbm", "mrf", "hnn", "mlp", "ens:ml+ucsv")


def _r(x, nd: int = 2):
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, nd)


def _series(s: pd.Series, nd: int = 2) -> dict:
    """A monthly series as its first month plus a list of values (None where missing)."""
    s = s.sort_index()
    full = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
    return {"start": str(full.index.min()), "values": [_r(v, nd) for v in full.to_numpy()]}


def _history(store: dict, c: str) -> pd.Series:
    pi = store["frames"][c][0]["pi"].dropna()
    pi.index = pi.index - 1  # known-at month -> reference month of the print
    return pi


def _bands(r: pd.DataFrame, n: int = 120) -> dict:
    """Quantiles of the last ``n`` realised forecast errors (actual minus forecast)."""
    e = (r["actual"] - r["forecast"]).dropna().iloc[-n:]
    return {f"q{int(q * 100)}": float(np.quantile(e, q)) for q in (0.1, 0.25, 0.75, 0.9)}


def site_data(store: dict, results: dict, latest: pd.DataFrame, scenarios: dict | None, scores: pd.DataFrame,
              frames_for_drivers: dict | None = None, asof: str | None = None) -> dict:
    out = {"asof": asof or str(pd.Timestamp.today().date()), "economies": [], "history": {}, "latest": {},
           "paths": {}, "subperiods": {}, "accuracy": [], "models": {}}
    out["subperiodLabels"] = list(SUBPERIODS)
    for c in ECONOMIES:
        hist = _history(store, c)
        out["history"][c] = _series(hist.loc["1990-01":])
        out["economies"].append({"code": c, "name": NAMES[c], "index": INDEX[c], "month": str(hist.index.max()),
                                 "value": _r(hist.iloc[-1])})
        out["latest"][c], out["paths"][c], out["subperiods"][c] = {}, {}, {}
        for h in HORIZONS:
            res = collect(results, c, h)
            lat = latest[(latest.country == c) & (latest.horizon == h)].set_index("model")
            med = float(lat.loc["ens:median", "forecast"])
            entry = {"target": str(lat["target_month"].iloc[0]), "median": _r(med),
                     "bands": {k: _r(med + v) for k, v in _bands(res["ens:median"]).items()},
                     "models": {m: _r(lat.loc[m, "forecast"]) for m in LATEST_MODELS if m in lat.index}}
            for m in ("fm:timesfm3_cov", "fm:chronos2_cov"):
                r = res.get(m)
                if r is not None and not r.empty:
                    row = r.iloc[-1]
                    entry["models"][m] = _r(row["forecast"])
                    if m == "fm:timesfm3_cov":
                        entry["tfmBand"] = [_r(row["q10"]), _r(row["q90"])]
            out["latest"][c][h] = entry
            out["paths"][c][h] = {}
            for m in PATH_MODELS:
                r = res.get(m)
                if r is None or r.empty:
                    continue
                f = r["forecast"].copy()
                f.index = target_month(f.index, h)
                out["paths"][c][h][m] = _series(f)
            sub = subperiod_table({m: r for m, r in res.items() if not m.startswith("deep:")}, h)
            out["subperiods"][c][h] = {m: [_r(v) for v in sub.loc[m, list(SUBPERIODS)]] for m in sub.index}
    for row in scores.to_dict("records"):
        out["accuracy"].append({"c": row["country"], "h": int(row["horizon"]), "m": row["model"],
                                "rel": _r(row["rel. RW"], 3), "mcs": bool(row["in MCS"]), "p": _r(row["DM p"], 4),
                                "n": int(row["N"]), "start": row["start"]})
    for m in ORDER:
        out["models"][m] = {"label": LABELS[m], "short": SHORT[m], "family": family(m)}
    if frames_for_drivers:
        out["drivers"] = drivers_data(frames_for_drivers)
    if scenarios:
        out["scenarios"] = scenario_data(scenarios)
    return out


def drivers_data(frames: dict) -> dict:
    from .explain import boosting_importance, driver_regressions

    d = {}
    for h in (12, 3):
        reg = driver_regressions(frames, h)
        d[f"h{h}"] = [{"driver": r.driver, "c": r.economy, "coef": _r(r.coef, 3), "t": _r(r.t, 2), "p": _r(r.p, 4)}
                      for r in reg.itertuples(index=False)]
        meta = reg.groupby("economy")[["R2", "N", "from", "to"]].first()
        d[f"fit{h}"] = {c: {"r2": _r(meta.loc[c, "R2"]), "n": int(meta.loc[c, "N"]), "from": meta.loc[c, "from"],
                            "to": meta.loc[c, "to"]} for c in meta.index}
    bi = boosting_importance(frames, 12)
    blocks = bi.groupby(["economy", "group"])["gain share"].sum()
    d["blocks"] = {c: {g: _r(v, 3) for g, v in blocks[c].items()} for c in frames}
    return d


def scenario_data(responses: dict) -> dict:
    from .scenario import LEVERS, MODELS, PRESETS

    out = {"levers": [{"key": lv.key, "label": lv.label, "unit": lv.unit, "grid": list(lv.grid), "note": lv.note}
                      for lv in LEVERS],
           "presets": [{"key": p.key, "label": p.label, "blurb": p.blurb, "values": p.values} for p in PRESETS],
           "models": list(MODELS), "resp": {}, "active": {}, "check": []}
    for (c, h, m), r in sorted(responses.items()):
        if not r:
            continue
        out["resp"].setdefault(c, {}).setdefault(str(h), {})[m] = {
            "base": _r(r["base"], 3), "curves": {k: [_r(v, 3) for v in vals] for k, vals in r["curves"].items()}}
        act = out["active"].setdefault(c, {})
        for k, cols in r["active"].items():
            act[k] = sorted(set(act.get(k, [])) | set(cols))
    # How far the lever-by-lever shortcut is from solving each preset jointly (median across models).
    for c in ECONOMIES:
        for h in HORIZONS:
            rs = [responses[(c, h, m)] for m in MODELS if responses.get((c, h, m))]
            for p in PRESETS:
                joint = np.median([r["base"] + r["joint"][p.key] for r in rs])
                add = np.median([r["base"] + r["additive"][p.key] for r in rs])
                base = np.median([r["base"] for r in rs])
                out["check"].append({"c": c, "h": h, "preset": p.key, "joint": _r(joint - base, 3),
                                     "additive": _r(add - base, 3)})
    return out


def render(template: str, data: dict) -> str:
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    if "/*__DATA__*/null" not in template:
        raise ValueError("template has no /*__DATA__*/null placeholder")
    return template.replace("/*__DATA__*/null", payload)


def write_site(data: dict, template_path: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(template_path.read_text(), data))
    return out_path
