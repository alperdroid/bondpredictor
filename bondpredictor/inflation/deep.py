"""Tier 2: generic deep time-series networks, trained globally across countries.

N-HiTS (Challu et al. 2023), PatchTST (Nie et al. 2023), TiDE (Das et al.
2023) and an LSTM, from Nixtla's ``neuralforecast``.  These are univariate:
each sees only the history of y/y inflation, but one network is trained on
the inflation series of ~40 economies (the OECD prices database, minus
hyperinflation episodes) so it can learn shared dynamics, the usual way to
make deep forecasters work on short macro series.

Timing matches the econometric models: at origin ``k`` the four target
series are known up to month ``k - 1`` and the other economies up to
``k - 2``.  Networks are refitted every ``refit_every`` months on that
information and applied at each origin in between with the latest context.
The network outputs the next 24 months at once; horizon ``h`` is step ``h``.

Pretrained foundation models (Chronos, TimesFM, Moirai) were not run:
their weights are on Hugging Face, which the sandbox could not reach.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from . import config as cfg
from .data import yoy

log = logging.getLogger(__name__)

MAX_H = 24


def deep_panel(store: dict, codes, max_abs: float = 25.0, start: str = "1970-01") -> dict[str, pd.Series]:
    """y/y inflation by reference month: the four targets plus the OECD economies (lag in ``attrs``)."""
    series = {}
    for c in codes:
        f = store["frames"][c][0]
        s = f["pi"].dropna().copy()
        s.index = s.index - 1  # known-at month -> reference month (CPI lag 1)
        s.attrs["lag"] = 1
        series[c] = s
    panel = store["oecd_panel"]
    targets = {cfg.COUNTRIES[c].oecd_code for c in codes}
    for area in panel.columns:
        if area in cfg.DEEP_PANEL_EXCLUDE or area in targets:
            continue
        s = yoy(panel[area]).dropna()
        s = s[s.index >= pd.Period(start, "M")]
        wild = s.index[s.abs() > max_abs]
        if len(wild):
            s = s[s.index > wild.max()]
        if len(s) >= 120:
            s.attrs["lag"] = cfg.GLOBAL_FACTOR_LAG
            series[area] = s
    return series


def _long(series: dict[str, pd.Series], known_at: pd.Period) -> pd.DataFrame:
    """Panel in neuralforecast's long format with only data published by the end of ``known_at``."""
    parts = []
    for uid, s in series.items():
        s = s[s.index <= known_at - s.attrs.get("lag", 1)]
        if len(s) < 60:
            continue
        parts.append(pd.DataFrame({"unique_id": uid, "ds": s.index.to_timestamp(), "y": s.to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def make_models(names, seed: int = 0, max_steps: int = 600, input_size: int = 60):
    from neuralforecast.models import LSTM, NHITS, PatchTST, TiDE

    common = dict(h=MAX_H, input_size=input_size, max_steps=max_steps, scaler_type="robust", random_seed=seed,
                  enable_progress_bar=False, enable_model_summary=False, logger=False, accelerator="cpu")
    zoo = {
        "nhits": lambda: NHITS(**common),
        "patchtst": lambda: PatchTST(**common),
        "tide": lambda: TiDE(**common),
        "lstm": lambda: LSTM(**{**common, "input_size": input_size}),
    }
    return [zoo[n]() for n in names]


def run_deep(store: dict, codes, names=("nhits", "patchtst", "tide", "lstm"), horizons=(3, 6, 12, 24),
             start: str = "2000-01", end: str | None = None, refit_every: int = 24, seed: int = 0,
             max_steps: int = 600, progress=None) -> dict:
    """Backtest results keyed like the econometric ones: (country, h, "deep:<name>")."""
    from neuralforecast import NeuralForecast

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    series = deep_panel(store, codes)
    end_p = pd.Period(end, "M") if end else store["last_month"]
    origins = pd.period_range(pd.Period(start, "M"), end_p, freq="M")
    rows = {(c, h, n): [] for c in codes for h in horizons for n in names}
    nf, fitted_at = None, None
    for k in origins:
        if nf is None or (k - fitted_at).n >= refit_every:
            train = _long(series, k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nf = NeuralForecast(models=make_models(names, seed, max_steps), freq="MS")
                nf.fit(df=train)
            fitted_at = k
            if progress:
                progress(("deep", str(k)), f"fitted on {train['unique_id'].nunique()} series")
        ctx = _long({c: series[c] for c in codes}, k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = nf.predict(df=ctx)
        fc = fc.reset_index() if "unique_id" not in fc.columns else fc
        for c in codes:
            f = fc[fc["unique_id"] == c].sort_values("ds")
            if len(f) < MAX_H:
                continue
            pi_k = series[c].get(k - 1, np.nan)
            if not np.isfinite(pi_k):
                continue
            for h in horizons:
                actual = series[c].get(k - 1 + h, np.nan)
                for n, col in zip(names, [type(m).__name__ for m in nf.models]):
                    rows[(c, h, n)].append({"date": k, "forecast": float(f[col].iloc[h - 1]), "rw": float(pi_k),
                                            "actual": actual, "refit": fitted_at == k})
    return {(c, h, f"deep:{n}"): pd.DataFrame(r).set_index("date") for (c, h, n), r in rows.items() if r}
