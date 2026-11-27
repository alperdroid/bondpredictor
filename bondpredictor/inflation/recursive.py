"""Recursive least squares for real-time direct forecasts."""

from __future__ import annotations

import numpy as np


def recursive_ols(Z: np.ndarray, y: np.ndarray, origins: np.ndarray, h: int, t0: int = 0, min_train: int = 120,
                  ridge: float = 1e-9) -> np.ndarray:
    """OLS forecasts at each origin ``k`` from the rows ``t0 <= j <= k - h`` with complete data.

    Uses running sums of the normal equations, so each origin costs one small solve.  The tiny
    ridge is proportional to each diagonal element, so forecasts do not depend on column scaling;
    the rescaling by standard deviations only conditions the arithmetic.
    """
    n, p = Z.shape
    sd = np.nanstd(Z, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    Zs = Z / sd
    ok = np.isfinite(Zs).all(axis=1) & np.isfinite(y)
    ok[:t0] = False
    Zw = np.where(ok[:, None], Zs, 0.0)
    yw = np.where(ok, y, 0.0)
    A = np.cumsum(Zw[:, :, None] * Zw[:, None, :], axis=0)
    b = np.cumsum(Zw * yw[:, None], axis=0)
    cnt = np.cumsum(ok)
    j = origins - h
    good = (j >= 0) & np.isfinite(Zs[origins]).all(axis=1)
    good[good] &= cnt[j[good]] >= min_train
    out = np.full(len(origins), np.nan)
    if good.any():
        Aj, bj = A[j[good]], b[j[good]]
        diag = np.einsum("kii->ki", Aj)
        # A floor keeps an all-zero column (e.g. a tax dummy before its first event) solvable.
        diag = diag + 1e-12 * diag.mean(axis=1, keepdims=True)
        reg = Aj + ridge * diag[:, :, None] * np.eye(p)
        beta = np.linalg.solve(reg, bj[..., None])[..., 0]
        out[good] = (Zs[origins[good]] * beta).sum(axis=1)
    return out
