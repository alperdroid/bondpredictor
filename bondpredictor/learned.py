"""Predictors estimated from data: Cochrane–Piazzesi, cycle, global and PCA factors.

Each one is a :class:`Transformer`.  The backtest re-fits it at every
forecast origin on the information set available then (future returns
masked), and applies it to that origin's predictors, so no step leaks
future data.  The same objects are used for full-sample in-sample work.
"""

from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd


def _lstsq_with_const(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


class Transformer:
    outputs: list[str] = []

    def inputs(self) -> list[str]:
        return []

    def required_columns(self, available) -> list[str]:
        """Columns of the data frame this transformer reads."""
        return [c for c in self.inputs() if c in set(available)]

    def fit(self, info: pd.DataFrame) -> "Transformer":
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, info: pd.DataFrame) -> pd.DataFrame:
        return self.fit(info).transform(info)

    def clone(self) -> "Transformer":
        return copy.deepcopy(self)


class ReturnFactor(Transformer):
    """Fitted value of the mean of ``targets`` regressed on ``regressors``.

    With one-year forward rates as regressors and average one-year excess
    returns on 2–5 year bonds as targets, this is the Cochrane–Piazzesi
    (2005) factor gamma'f_t.
    """

    def __init__(self, regressors: list[str], targets: list[str], output: str, min_obs: int = 60):
        self.regressors = list(regressors)
        self.targets = list(targets)
        self.outputs = [output]
        self.min_obs = min_obs
        self.beta: np.ndarray | None = None

    def inputs(self):
        return self.regressors + self.targets

    def fit(self, info):
        if not set(self.inputs()) <= set(info.columns):
            self.beta = None
            return self
        X = info[self.regressors]
        y = info[self.targets].mean(axis=1, skipna=False)
        ok = X.notna().all(axis=1) & y.notna()
        if ok.sum() < self.min_obs:
            self.beta = None
            return self
        self.beta = _lstsq_with_const(y[ok].to_numpy(), X[ok].to_numpy())
        return self

    def transform(self, data):
        if self.beta is None or not set(self.regressors) <= set(data.columns):
            return pd.DataFrame({self.outputs[0]: np.nan}, index=data.index)
        X = data[self.regressors].to_numpy(dtype=float)
        val = self.beta[0] + X @ self.beta[1:]
        return pd.DataFrame({self.outputs[0]: val}, index=data.index)


def cp_factor(country: str, maturities=(2, 3, 4, 5), n_forwards: int = 5, horizon: int = 12, name: str = "cp") -> ReturnFactor:
    from .returns import rx_column

    return ReturnFactor(
        regressors=[f"{country}.fwd_{k}" for k in range(1, n_forwards + 1)],
        targets=[f"{country}.{rx_column(horizon, n)}" for n in maturities],
        output=f"{country}.{name}",
    )


class GlobalAverage(Transformer):
    """Equal-weighted cross-country average of country factors.

    With CP factors this is the global CP factor of Dahlquist–Hasseltoft
    (2013).  They use GDP weights; equal weights avoid needing GDP in a
    common currency.  Countries whose factor is missing are skipped.
    """

    def __init__(self, factors: list[Transformer], output: str = "G.cp"):
        self.factors = factors
        self.outputs = [output]

    def inputs(self):
        return sorted({c for f in self.factors for c in f.inputs()})

    def fit(self, info):
        for f in self.factors:
            f.fit(info)
        return self

    def transform(self, data):
        parts = [f.transform(data).iloc[:, 0] for f in self.factors]
        avg = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
        return pd.DataFrame({self.outputs[0]: avg}, index=data.index)


class CycleFactor(Transformer):
    """Cieslak–Povala (2015) cycle factor.

    Each yield is regressed on trend inflation ``tau``; the residuals
    (cycles) are then regressed jointly on average future excess returns
    and the fitted value is the factor.
    """

    def __init__(self, yields: list[str], tau: str, targets: list[str], output: str, min_obs: int = 60):
        self.yields = list(yields)
        self.tau = tau
        self.targets = list(targets)
        self.outputs = [output]
        self.min_obs = min_obs
        self.loadings: np.ndarray | None = None
        self.beta: np.ndarray | None = None

    def inputs(self):
        return self.yields + [self.tau] + self.targets

    def _cycles(self, data):
        Y = data[self.yields].to_numpy(dtype=float)
        tau = data[self.tau].to_numpy(dtype=float)[:, None]
        return Y - self.loadings[0][None, :] - tau * self.loadings[1][None, :]

    def fit(self, info):
        self.beta = None
        if not set(self.inputs()) <= set(info.columns):
            return self
        base = info[self.yields + [self.tau]].notna().all(axis=1)
        if base.sum() < self.min_obs:
            return self
        Y = info.loc[base, self.yields].to_numpy()
        tau = info.loc[base, self.tau].to_numpy()
        A = np.column_stack([np.ones(len(tau)), tau])
        self.loadings, *_ = np.linalg.lstsq(A, Y, rcond=None)  # 2 x n
        cyc = self._cycles(info)
        y = info[self.targets].mean(axis=1, skipna=False).to_numpy()
        ok = np.isfinite(cyc).all(axis=1) & np.isfinite(y)
        if ok.sum() < self.min_obs:
            return self
        self.beta = _lstsq_with_const(y[ok], cyc[ok])
        return self

    def transform(self, data):
        if self.beta is None:
            return pd.DataFrame({self.outputs[0]: np.nan}, index=data.index)
        cyc = self._cycles(data)
        return pd.DataFrame({self.outputs[0]: self.beta[0] + cyc @ self.beta[1:]}, index=data.index)


def cycle_factor(country: str, yield_maturities=(1, 2, 5, 7, 10), target_maturities=(2, 3, 4, 5), horizon: int = 12) -> CycleFactor:
    from .returns import maturity_label, rx_column

    return CycleFactor(
        yields=[f"{country}.y_{maturity_label(n)}" for n in yield_maturities],
        tau=f"{country}.tau",
        targets=[f"{country}.{rx_column(horizon, n)}" for n in target_maturities],
        output=f"{country}.cycle",
    )


class PCAFactors(Transformer):
    """Principal components of a panel (Ludvigson–Ng macro factors).

    At each fit, columns are kept if observed at the forecast origin and
    over at least ``min_coverage`` of the last ``lookback`` months.  They
    are standardised with in-sample moments and remaining gaps are
    mean-imputed.  Discontinued or late-starting series therefore drop in
    and out automatically.
    """

    def __init__(self, pattern: str, k: int, output_prefix: str, standardize: bool = True,
                 min_coverage: float = 0.8, lookback: int | None = 240, min_obs: int = 60):
        self.pattern = re.compile(pattern)
        self.k = k
        self.output_prefix = output_prefix
        self.outputs = [f"{output_prefix}{i + 1}" for i in range(k)]
        self.standardize = standardize
        self.min_coverage = min_coverage
        self.lookback = lookback
        self.min_obs = min_obs
        self.cols: list[str] = []

    def required_columns(self, available):
        return [c for c in available if self.pattern.search(c)]

    def fit(self, info):
        self.cols = []
        cols = [c for c in info.columns if self.pattern.search(c)]
        if not cols:
            return self
        X = info[cols].dropna(how="all")
        if self.lookback:
            X = X.iloc[-self.lookback :]
        if len(X) < self.min_obs or X.index[-1] != info.index[-1]:
            return self
        cov = X.notna().mean()
        keep = [c for c in cols if cov[c] >= self.min_coverage and pd.notna(X[c].iloc[-1])]
        if len(keep) < max(2, self.k):
            return self
        X = X[keep]
        self.mean = X.mean()
        self.std = X.std().replace(0, np.nan) if self.standardize else pd.Series(1.0, index=keep)
        keep = [c for c in keep if pd.notna(self.std[c])]
        Z = ((X[keep] - self.mean[keep]) / self.std[keep]).fillna(0.0).to_numpy()
        _, _, vt = np.linalg.svd(Z, full_matrices=False)
        V = vt[: self.k].T
        # Sign convention: largest absolute loading positive.
        V = V * np.sign(V[np.abs(V).argmax(axis=0), np.arange(V.shape[1])])
        self.cols, self.V = keep, V
        return self

    def transform(self, data):
        k_eff = self.V.shape[1] if self.cols else 0
        out = pd.DataFrame(np.nan, index=data.index, columns=self.outputs)
        if not self.cols:
            return out
        Z = (data[self.cols] - self.mean[self.cols]) / self.std[self.cols]
        present = data[self.cols].notna().mean(axis=1) >= 0.5
        F = Z.fillna(0.0).to_numpy() @ self.V
        F[~present.to_numpy()] = np.nan
        out.iloc[:, :k_eff] = F
        return out
