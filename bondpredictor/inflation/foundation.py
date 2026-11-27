"""Tier 2: pretrained time-series foundation models, zero-shot.

* Chronos-2 (Amazon, Oct 2025) and Chronos-Bolt (Nov 2024);
* TimesFM 2.5 (Google, Sep 2025) and TimesFM 3.0 (Aug 2026).

Each model gets the history of y/y inflation known at the origin (up to
``max_context`` months) and returns 24 months of quantiles; horizon ``h``
is step ``h`` and the point forecast is the median.  The ``_cov`` variants
also see a few macro series as past covariates (core inflation, energy and
oil inflation, the global inflation factor, unemployment), which Chronos-2
and TimesFM 3 support natively.  Nothing is fitted, so there is nothing to
re-estimate: each origin is one forward pass.

Caveat: these models were pretrained on large public corpora that may
include the very CPI series being forecast, up to their release dates.
Their out-of-sample scores before release are therefore not clean; only
origins after a model's release are.  The report flags this.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MAX_H = 24
QUANTILES = (0.1, 0.5, 0.9)
COVARIATES = ("core", "energy_yoy", "oil_yoy", "g_infl", "unemp_lvl")

MODELS = {
    "chronos2": {"repo": "amazon/chronos-2", "kind": "chronos2", "released": "2025-10"},
    "chronos2_cov": {"repo": "amazon/chronos-2", "kind": "chronos2", "cov": True, "released": "2025-10"},
    "chronos_bolt": {"repo": "amazon/chronos-bolt-base", "kind": "bolt", "released": "2024-11"},
    "timesfm25": {"repo": "google/timesfm-2.5-200m-pytorch", "kind": "tfm25", "released": "2025-09"},
    "timesfm3": {"repo": "google/timesfm-3.0-pytorch", "kind": "tfm3", "released": "2026-08"},
    "timesfm3_cov": {"repo": "google/timesfm-3.0-pytorch", "kind": "tfm3", "cov": True, "released": "2026-08"},
}


class _Loaded:
    """Loads each checkpoint once and turns contexts into (n, MAX_H, 3) quantile arrays."""

    def __init__(self):
        self.cache = {}

    def get(self, kind, repo):
        key = (kind, repo)
        if key in self.cache:
            return self.cache[key]
        if kind == "chronos2":
            from chronos import Chronos2Pipeline

            m = Chronos2Pipeline.from_pretrained(repo, device_map="cpu")
        elif kind == "bolt":
            from chronos import ChronosBoltPipeline

            m = ChronosBoltPipeline.from_pretrained(repo, device_map="cpu")
        elif kind == "tfm25":
            import timesfm

            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo)
            m.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=128, normalize_inputs=True,
                                             use_continuous_quantile_head=True, force_flip_invariance=True,
                                             infer_is_positive=False, fix_quantile_crossing=True))
        elif kind == "tfm3":
            import timesfm

            m = timesfm.TimesFM3Forecaster.from_pretrained(repo, device="cpu")
        else:
            raise ValueError(kind)
        self.cache[key] = m
        return m

    def quantiles(self, spec: dict, targets: list[np.ndarray], covs: list[dict] | None) -> np.ndarray:
        import torch

        m = self.get(spec["kind"], spec["repo"])
        kind = spec["kind"]
        if kind == "chronos2":
            if covs is None:
                inputs = [torch.tensor(t, dtype=torch.float32) for t in targets]
            else:
                inputs = [{"target": t.astype(np.float32), "past_covariates": {k: v.astype(np.float32) for k, v in c.items()}}
                          for t, c in zip(targets, covs)]
            out = m.predict(inputs, prediction_length=MAX_H, batch_size=64)
            qs = list(m.quantiles) if hasattr(m, "quantiles") else None
            res = []
            for o in out:
                a = o.detach().numpy()[0]  # (n_quantiles, h)
                idx = [int(np.argmin(np.abs(np.asarray(qs) - q))) for q in QUANTILES] if qs else [0, a.shape[0] // 2, -1]
                res.append(a[idx].T)
            return np.stack(res)
        if kind == "bolt":
            q, _ = m.predict_quantiles([torch.tensor(t, dtype=torch.float32) for t in targets], prediction_length=MAX_H,
                                       quantile_levels=list(QUANTILES))
            return q.detach().numpy()
        if kind == "tfm25":
            _, qf = m.forecast(horizon=MAX_H, inputs=[t.astype(np.float32) for t in targets])
            # TimesFM 2.5 quantile head: mean, then deciles 0.1..0.9.
            return np.stack([qf[:, :MAX_H, 1], qf[:, :MAX_H, 5], qf[:, :MAX_H, 9]], axis=-1)
        if kind == "tfm3":
            past = None if covs is None else [np.stack([c[k] for k in sorted(c)]).astype(np.float32) for c in covs]
            res = []
            for o in m.predict_batch([t.astype(np.float32) for t in targets], MAX_H, past_only_covariates=past,
                                     return_quantiles=True):
                q = np.asarray(o.quantiles)[:MAX_H]  # (h, 9): deciles 0.1 ... 0.9
                res.append(q[:, [0, 4, 8]])
            return np.stack(res)
        raise ValueError(kind)


def _covariates(frame: pd.DataFrame, rows: slice) -> dict[str, np.ndarray]:
    out = {}
    for c in COVARIATES:
        if c in frame:
            v = frame[c].iloc[rows].astype(float)
            out[c] = v.ffill().bfill().fillna(0.0).to_numpy()
    return out


def run_foundation(store: dict, codes, names=tuple(MODELS), horizons=(3, 6, 12, 24), start: str = "2000-01",
                   end: str | None = None, max_context: int = 512, progress=None) -> dict:
    loader = _Loaded()
    out = {}
    for c in codes:
        frame = store["frames"][c][0]
        pi = frame["pi"]
        # Default: up to the latest CPI print, so the last origin is a live forecast.
        end_p = pd.Period(end, "M") if end else pi.dropna().index.max()
        idx = frame.index
        origins = [k for k in range(len(idx)) if pd.Period(start, "M") <= idx[k] <= end_p and np.isfinite(pi.iloc[k])]
        first = int(np.argmax(pi.notna().to_numpy()))
        ctx_rows = [slice(max(first, k - max_context + 1), k + 1) for k in origins]
        targets = [pi.iloc[r].astype(float).interpolate(limit_area="inside").to_numpy() for r in ctx_rows]
        for name in names:
            spec = MODELS[name]
            covs = [_covariates(frame, r) for r in ctx_rows] if spec.get("cov") else None
            q = loader.quantiles(spec, targets, covs)  # (n_origins, MAX_H, 3)
            for h in horizons:
                rows = []
                for j, k in enumerate(origins):
                    tgt = frame[f"tgt{h}"].iloc[k] if f"tgt{h}" in frame else np.nan
                    rows.append({"date": idx[k], "forecast": float(q[j, h - 1, 1]), "rw": float(pi.iloc[k]),
                                 "actual": tgt, "q10": float(q[j, h - 1, 0]), "q90": float(q[j, h - 1, 2])})
                out[(c, h, f"fm:{name}")] = pd.DataFrame(rows).set_index("date")
            if progress:
                progress((c, name), f"{len(origins)} origins")
    return out
