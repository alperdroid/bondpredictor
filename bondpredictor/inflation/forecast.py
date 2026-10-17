"""Forecasts from the latest data.

Every tier-1 model is refitted on all realised targets and applied at each
economy's latest origin: the last month with a published CPI print.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import model_zoo, run_jobs
from .evaluation import NONLINEAR_ML


def latest_origin(frame: pd.DataFrame) -> pd.Period:
    return frame["pi"].dropna().index.max()


def latest_forecasts(frames: dict, horizons, n_jobs: int = 1) -> pd.DataFrame:
    jobs = []
    for c, (frame, _) in frames.items():
        k = str(latest_origin(frame))
        for s in model_zoo(1):
            for h in horizons:
                jobs.append(((c, h, s.name), s, dict(start=k, end=k, explain=True)))
    raw = run_jobs(frames, jobs, n_jobs)
    rows = []
    for (c, h, name), r in raw.items():
        if isinstance(r, pd.DataFrame) and not r.empty:
            row = r.iloc[-1]
            origin = r.index[-1]
            rows.append({"country": c, "horizon": h, "model": name, "latest_print": str(origin - 1),
                         "target_month": str(origin - 1 + h), "latest_yoy": row["rw"], "forecast": row["forecast"],
                         **{k: row[k] for k in r.columns if k.startswith(("q", "c_", "b_")) and pd.notna(row[k])}})
    out = pd.DataFrame(rows)
    ens = []
    for (c, h), g in out.groupby(["country", "horizon"]):
        f = g.set_index("model")["forecast"]
        base = g.iloc[0][["country", "horizon", "latest_print", "target_month", "latest_yoy"]].to_dict()
        ml = [m for m in NONLINEAR_ML if m in f]
        if ml:
            ens.append({**base, "model": "ens:ml", "forecast": f[ml].mean()})
            if "ucsv" in f:
                ens.append({**base, "model": "ens:ml+ucsv", "forecast": 0.5 * f[ml].mean() + 0.5 * f["ucsv"]})
        others = [m for m in f.index if m not in ("rw", "ucsv", "ar")]
        ens.append({**base, "model": "ens:median", "forecast": float(np.median(f[others]))})
    out = pd.concat([out, pd.DataFrame(ens)], ignore_index=True)
    return out.sort_values(["country", "horizon", "model"]).reset_index(drop=True)
