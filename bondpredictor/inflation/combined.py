"""The combined forecast: the headline where it beat the median of the models in the backtest.

It averages up to three forecasts, each the median of six fast models (ridge,
adaptive lasso, elastic net, kernel ridge, random forest, gradient boosting;
see ``pilot.py`` for how each is built):

* ``nowcast``: the direct models, with the nowcast of the month that is not
  yet published and this month's pump prices among their predictors;
* ``bottom_up``: energy, food, core goods and services inflation forecast
  separately and added up;
* ``nowcast+survey_anchor``: the nowcast models forecasting the deviation
  from the professional forecasters' expected path.

``PARTS`` lists the parts used in each economy.  An economy is listed only if
the combination beat the median of that economy's models at most horizons, in
both halves of the backtest (``docs/inflation_pilot.md``); elsewhere the
median of the models stays the headline.  At an origin where a part has no
forecast yet (too little history), the parts that exist are averaged.
"""

from __future__ import annotations

import pandas as pd

from .pilot import medians, run_pilot

KEY = "ens:combined"
MEDIAN = "ens:median"
PARTS: dict[str, tuple[str, ...]] = {
    "US": ("nowcast", "bottom_up", "nowcast+survey_anchor"),
}


def headline_model(code: str, available=None) -> str:
    """The model whose forecast is an economy's headline: the combination where adopted and available."""
    return KEY if code in PARTS and (available is None or KEY in available) else MEDIAN


def combine(F: pd.DataFrame, parts) -> pd.Series:
    """Average of the parts' forecasts that exist at each origin."""
    return F[[p for p in parts if p in F]].mean(axis=1).dropna()


def combined_backtest(pilot: dict, parts: dict | None = None) -> dict:
    """``{(economy, horizon, "ens:combined"): frame}`` with ``forecast``, ``rw`` and ``actual`` per origin."""
    parts = PARTS if parts is None else parts
    out = {}
    for c, ps in parts.items():
        for h in sorted({hh for (cc, hh, s, _) in pilot if cc == c and s in ps}):
            F = medians(pilot, c, h)
            f = combine(F, ps)
            if f.empty:
                continue
            ref = pd.concat([r[["rw", "actual"]] for (cc, hh, s, _), r in pilot.items()
                             if (cc, hh) == (c, h) and s in ps and isinstance(r, pd.DataFrame) and not r.empty])
            ref = ref[~ref.index.duplicated()].reindex(f.index)
            out[(c, h, KEY)] = pd.DataFrame({"forecast": f, "rw": ref["rw"], "actual": ref["actual"]})
    return out


def latest_combined(store: dict, dl, horizons, parts: dict | None = None, n_jobs: int = 1) -> pd.DataFrame:
    """The combination and its parts, every model refitted on all realised data at each economy's latest print.

    Rows like those of ``forecast.latest_forecasts``: models ``ens:combined`` and ``part:<name>``.
    """
    from .forecast import latest_origin

    parts = PARTS if parts is None else parts
    rows = []
    for c, ps in parts.items():
        frame = store["frames"][c][0]
        k = latest_origin(frame)
        direct = [p for p in ps if p != "bottom_up"]
        res = run_pilot(store, dl, [c], tuple(horizons), start=str(k), end=str(k), n_jobs=n_jobs,
                        bottomup="bottom_up" in ps, sets=direct)
        for h in horizons:
            F = medians(res, c, h)
            if F.empty or k not in F.index:
                continue
            base = {"country": c, "horizon": h, "latest_print": str(k - 1), "target_month": str(k - 1 + h),
                    "latest_yoy": float(frame["pi"].loc[k])}
            for p in ps:
                if p in F and pd.notna(F.loc[k, p]):
                    rows.append({**base, "model": f"part:{p}", "forecast": float(F.loc[k, p])})
            f = combine(F.loc[[k]], ps)
            if not f.empty:
                rows.append({**base, "model": KEY, "forecast": float(f.iloc[0])})
    return pd.DataFrame(rows)
