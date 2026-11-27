"""Unobserved-components model with stochastic volatility (Stock and Watson 2007).

    pi_t = tau_t + eta_t,     eta_t ~ N(0, exp(h_eta_t))
    tau_t = tau_{t-1} + eps_t, eps_t ~ N(0, exp(h_eps_t))
    h_t = h_{t-1} + gamma * N(0, 1)  for both log variances

The hard-to-beat univariate inflation benchmark.  As in Stock and Watson
the volatility-of-volatility ``gamma`` is fixed (0.2 per quarter, so
0.2/sqrt(3) per month), so there is nothing to estimate and one filtering
pass gives real-time trend estimates: the filtered ``tau_t`` uses data up
to ``t`` only.  A Rao-Blackwellised particle filter integrates ``tau_t``
out with a Kalman step and samples only the two log variances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ucsv_filter(pi: pd.Series, gamma: float = 0.2 / np.sqrt(3), n_particles: int = 2000, seed: int = 0) -> pd.Series:
    """Filtered trend ``tau_{t|t}`` of an (annualised, seasonally adjusted) monthly inflation series."""
    rng = np.random.default_rng(seed)
    x = pi.to_numpy(dtype=float)
    first = np.flatnonzero(np.isfinite(x))
    out = np.full(len(x), np.nan)
    if len(first) < 24:
        return pd.Series(out, index=pi.index)
    x0 = x[first[0] : first[0] + 24]
    x0 = x0[np.isfinite(x0)]
    v0 = float(np.var(x0)) or 1.0
    N = n_particles
    m = np.full(N, float(np.mean(x0)))
    P = np.full(N, v0)
    h_eps = np.log(v0 / 10.0) + rng.normal(size=N)
    h_eta = np.log(v0 / 2.0) + rng.normal(size=N)
    w = np.full(N, 1.0 / N)
    for t in range(first[0], len(x)):
        h_eps += gamma * rng.normal(size=N)
        h_eta += gamma * rng.normal(size=N)
        Pp = P + np.exp(h_eps)
        if np.isfinite(x[t]):
            S = Pp + np.exp(h_eta)
            e = x[t] - m
            logl = -0.5 * (np.log(S) + e * e / S)
            K = Pp / S
            m = m + K * e
            P = (1.0 - K) * Pp
            lw = np.log(w) + logl
            lw -= lw.max()
            w = np.exp(lw)
            w /= w.sum()
            if 1.0 / np.sum(w * w) < N / 2:  # systematic resampling
                u = (rng.random() + np.arange(N)) / N
                i = np.minimum(np.searchsorted(np.cumsum(w), u), N - 1)
                m, P, h_eps, h_eta = m[i], P[i], h_eps[i], h_eta[i]
                w = np.full(N, 1.0 / N)
        else:
            P = Pp
        out[t] = float(np.sum(w * m))
    return pd.Series(out, index=pi.index)
