"""Which predictors matter, and how significantly.

Three complementary views:

* :func:`driver_regressions`: in-sample predictive regressions of the change
  in y/y inflation over ``h`` months on a dozen harmonised drivers
  (standardised, so coefficients are in percentage points per standard
  deviation), with Newey-West standard errors (``h - 1`` lags) for the
  overlapping targets.  Descriptive, not a forecast evaluation.
* :func:`lasso_selection`: how often the adaptive lasso kept each predictor
  when refitted every December from 2000 on data available at the time,
  and with which sign.  This is what the shrinkage model actually used out
  of sample.
* :func:`boosting_importance`: LightGBM gain importance on all data,
  aggregated by predictor block.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..inference import newey_west
from .backtest import design, make_model

# Harmonised drivers: label -> column per economy (None = not available).
DRIVERS = {
    "Current inflation (y/y)": {c: "pi" for c in ("US", "UK", "JP", "EA")},
    "Recent momentum (6m annualised)": {c: "sa6" for c in ("US", "UK", "JP", "EA")},
    "Headline minus core": {c: "headline_minus_core" for c in ("US", "UK", "JP", "EA")},
    "Inflation expectations": {"US": "mich_lvl", "UK": "breakeven10_lvl", "JP": "tankan_output_prices_lvl",
                               "EA": "consumer_price_exp_lvl"},
    "Unemployment gap": {c: "unemp_gap" for c in ("US", "UK", "JP", "EA")},
    "Wage growth": {"US": "wages_yoy", "UK": "earnings_yoy", "JP": "earnings_yoy"},
    "Oil price inflation": {c: "oil_yoy" for c in ("US", "UK", "JP", "EA")},
    "Producer price inflation": {c: "ppi_yoy" for c in ("US", "UK", "JP", "EA")},
    "Exchange rate (y/y, up = stronger)": {"US": "dollar_yoy", "UK": "reer_yoy", "JP": "reer_yoy", "EA": "neer_yoy"},
    "Supply-chain pressure (GSCPI)": {c: "gscpi_lvl" for c in ("US", "UK", "JP", "EA")},
    "Global inflation": {c: "g_infl" for c in ("US", "UK", "JP", "EA")},
    "Money growth": {"US": "m2_yoy", "UK": "money_yoy", "JP": "money_yoy", "EA": "m3_yoy"},
    "Policy rate change (12m)": {"US": "fedfunds_d12", "UK": "policy_rate_d12", "JP": "call_rate_d12",
                                 "EA": "euribor3m_d12"},
}


def _p_two_sided(t: float) -> float:
    return math.erfc(abs(t) / math.sqrt(2.0))


def driver_regressions(frames: dict, h: int = 12, start: str = "1998-01") -> pd.DataFrame:
    """Coefficient, Newey-West t and p per driver and economy (long format), plus R² and N."""
    rows = []
    for c, (frame, _) in frames.items():
        cols = {lab: m[c] for lab, m in DRIVERS.items() if m.get(c) in frame}
        X = frame[list(cols.values())].copy()
        X.columns = list(cols)
        y = frame[f"tgt{h}"] - frame["pi"]
        data = pd.concat([X, y.rename("_y")], axis=1).loc[pd.Period(start, "M"):].dropna()
        Z = (data[list(cols)] - data[list(cols)].mean()) / data[list(cols)].std()
        A = np.column_stack([np.ones(len(Z)), Z.to_numpy()])
        yv = data["_y"].to_numpy()
        beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
        u = yv - A @ beta
        se = np.sqrt(np.diag(newey_west(A, u, max(h - 1, 0))))
        r2 = 1 - u.var() / yv.var()
        for i, lab in enumerate(cols, start=1):
            t = beta[i] / se[i]
            rows.append({"economy": c, "driver": lab, "coef": beta[i], "t": t, "p": _p_two_sided(t),
                         "R2": r2, "N": len(yv), "from": str(data.index.min()), "to": str(data.index.max())})
    return pd.DataFrame(rows)


def lasso_selection(frames: dict, h: int = 12, start: int = 2000, end: int = 2025, min_train: int = 120,
                    min_hist: int = 60) -> pd.DataFrame:
    """Share of yearly real-time adaptive-lasso fits that keep each predictor, and its average sign."""
    rows = []
    for c, (frame, groups) in frames.items():
        X, g, pi, tgt = design(frame, groups, h)
        y = (tgt - pi).to_numpy()
        Xv = X.to_numpy(float)
        idx = frame.index
        t0 = idx.searchsorted(pd.Period("1960-01", "M"))
        kept, signs, n_fits = {}, {}, 0
        for year in range(start, end + 1):
            k = idx.searchsorted(pd.Period(f"{year}-12", "M"))
            if k >= len(idx):
                break
            train = np.arange(t0, k - h + 1)
            train = train[np.isfinite(y[train])]
            if len(train) < min_train:
                continue
            ok_now = np.isfinite(Xv[k])
            hist = np.isfinite(Xv[train]).sum(axis=0)
            use = [i for i in range(Xv.shape[1]) if ok_now[i] and hist[i] >= min_hist and np.nanstd(Xv[train, i]) > 0]
            names = [X.columns[i] for i in use]
            m = make_model("adalasso", h).fit(Xv[np.ix_(train, use)], y[train], names, [g[n] for n in names])
            n_fits += 1
            for n, b in zip(names, m.beta):
                if abs(b) > 1e-8:
                    kept[n] = kept.get(n, 0) + 1
                    signs[n] = signs.get(n, 0) + np.sign(b)
        for n, k_ in kept.items():
            rows.append({"economy": c, "predictor": n, "group": g.get(n, ""), "selected share": k_ / n_fits,
                         "positive share": (signs[n] / k_ + 1) / 2, "fits": n_fits})
    return pd.DataFrame(rows).sort_values(["economy", "selected share"], ascending=[True, False])


def boosting_importance(frames: dict, h: int = 12) -> pd.DataFrame:
    """LightGBM gain importance (share of total) by predictor block, fitted on all realised data."""
    rows = []
    for c, (frame, groups) in frames.items():
        X, g, pi, tgt = design(frame, groups, h)
        y = (tgt - pi)
        ok = y.notna() & pi.notna() & (X.index >= pd.Period("1960-01", "M"))
        cols = [n for n in X.columns if X.loc[ok, n].notna().sum() >= 60]
        m = make_model("lgbm", h).fit(X.loc[ok, cols].to_numpy(float), y[ok].to_numpy(), cols, [g[n] for n in cols])
        gain = m.m.booster_.feature_importance(importance_type="gain")
        imp = pd.Series(gain / gain.sum(), index=cols)
        for n, v in imp.items():
            rows.append({"economy": c, "predictor": n, "group": g[n], "gain share": v})
    return pd.DataFrame(rows)


def _stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def drivers_report(frames: dict) -> str:
    """Markdown with the three views (regressions at 3 and 12 months, lasso selection, boosting importance)."""
    from .report import NAMES, md_table

    L = ["<!-- Generated by `python -m bondpredictor.inflation explain`. -->", "",
         "# What drives the inflation forecasts", ""]
    for h in (12, 3):
        d = driver_regressions(frames, h)
        d["cell"] = [f"{c:+.2f}{_stars(p)}" for c, p in zip(d["coef"], d["p"])]
        tab = d.pivot(index="driver", columns="economy", values="cell").reindex(list(dict.fromkeys(d["driver"])))
        tab = tab[[c for c in ("US", "UK", "JP", "EA") if c in tab]].fillna("–")
        meta = d.groupby("economy")[["R2", "N", "from", "to"]].first()
        tab.loc["R² / months"] = [f"{meta.loc[c, 'R2']:.2f} / {meta.loc[c, 'N']}" for c in tab.columns]
        L += [f"## Predictive regressions, {h} months ahead", "",
              f"Change in y/y inflation over the next {h} months regressed on standardised drivers "
              f"(origins {meta['from'].min()} to {meta['to'].max()}; euro area from {meta.loc['EA', 'from'] if 'EA' in meta.index else ''}).  "
              "Coefficients are percentage points per standard deviation of the driver; Newey-West standard "
              f"errors with {h - 1} lags; *** p < 0.01, ** p < 0.05, * p < 0.1.  In-sample and descriptive.", "",
              md_table(tab, index_name="driver"), ""]
    ls = lasso_selection(frames, 12)
    L += ["## What the adaptive lasso kept (12 months ahead)", "",
          "Share of the yearly real-time refits (December 2000 to 2025) in which each predictor had a non-zero "
          "coefficient, and the share of those with a positive sign.  Top ten per economy.", ""]
    for c in ("US", "UK", "JP", "EA"):
        x = ls[ls["economy"] == c].head(10).set_index("predictor")[["group", "selected share", "positive share"]]
        if not x.empty:
            L += [f"### {NAMES[c]}", "", md_table(x, index_name="predictor"), ""]
    bi = boosting_importance(frames, 12)
    blocks = bi.groupby(["group", "economy"])["gain share"].sum().unstack()[["US", "UK", "JP", "EA"]].fillna(0.0)
    L += ["## What gradient boosting relied on (12 months ahead)", "",
          "Share of LightGBM's total split gain by predictor block, fitted on all realised data.", "",
          md_table(blocks, index_name="block"), ""]
    for c in ("US", "UK", "JP", "EA"):
        x = bi[bi["economy"] == c].nlargest(8, "gain share").set_index("predictor")[["group", "gain share"]]
        L += [f"### {NAMES[c]}: top predictors", "", md_table(x, index_name="predictor"), ""]
    return "\n".join(L)
