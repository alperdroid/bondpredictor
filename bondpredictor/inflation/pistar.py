"""Persistent-state trend model: a linear Deep-pi* (Montes-Galdón 2026, ECB Working Paper 3307).

Monthly seasonally adjusted inflation (annualised) follows an autoregression around a slowly moving trend
that the predictors drive:

    s[k+1]    = sum_p phi_p s[k+1-p] + (1 - sum_p phi_p) pistar[k] + e[k+1]
    pistar[k] = theta pistar[k-1] + (1 - theta) (alpha + x[k]' beta)

Row ``k`` is the forecast origin at the end of month ``k`` (latest CPI print for ``k - 1``, see
``features``), ``s[k]`` the one-sided seasonally adjusted rate of that print (``sa1``) and ``x[k]`` the
standardised predictors published by then.  As in the paper, the trend reads every predictor except
headline CPI itself (core, components, prices, slack, money, expectations, global); headline enters
through the lags.  The paper's neural network ``g(x)`` becomes a linear function and its bound on the
innovation, which does not bind for usual values, is dropped.  With ``theta`` fixed the trend is then
linear in ``(pistar[t0 - 1], alpha, beta)``:

    pistar[k] = theta^n pistar[t0 - 1] + (1 - theta^n) alpha + z[k]' beta,   n = k - t0 + 1
    z[k]      = theta z[k-1] + (1 - theta) x[k],

an exponentially weighted average of past predictors with a half-life of log(0.5) / log(theta) months
(138 at the paper's theta = 0.995).  ``theta = 0`` removes the memory: the "trend" is then just
``alpha + x[k]' beta``, which isolates what the persistent state adds.  Forecasts iterate the
autoregression with future innovations at their mean, so the trend decays from ``pistar[k]`` towards
``alpha`` at rate ``theta``.

Two losses, each re-estimated at every origin on data published by then:

* ``cur``: one-month-ahead errors (the paper's "Econometrician's loss").  Substituting
  ``c = (1 - sum phi) (pistar0, alpha, beta)`` makes it a ridge regression on the lags, the trend basis
  and ``z``.
* ``fwd``: errors of the iterated forecasts 1..H months ahead from every training origin (the "Central
  Banker's loss").  ``partial=True`` keeps origins whose H months are only partly published, scored on
  the months already out, as in the paper; ``partial=False`` waits until all H are out (the paper's
  footnote 3 counterfactual).  Given ``phi`` the forecasts are linear in the trend parameters, so ``phi``
  is optimised numerically and the rest is a ridge solve at each step (variable projection).  The loss has
  two local optima in ``phi`` (persistent and trend-led), so every fit starts from both the previous
  solution and the one-month-loss solution and keeps the better one.

The ridge penalty on ``beta`` is chosen every ``tune_every`` months by expanding-window validation on the
model's own loss, with ``phi`` re-estimated for each candidate penalty on each fold.  Year-on-year forecasts follow from the monthly path:

    pi[k+h] = pi[k] + mean(a[k+h-11 .. k+h]) - mean(s[k-11 .. k]),

with ``a`` the published rates up to ``k`` and the forecast path after (seasonal factors of the same
calendar months cancel between the two means).  Consumption-tax and VAT changes are taken out of the
monthly series at the pass-through estimated from earlier events, and that share of their known effect
on y/y inflation is added back.  The paper's time-varying volatility, Student-t errors and
attention layer are not reproduced; errors are Gaussian with constant variance, like the other models here.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

THETA = 0.995  # the paper's calibration: half-life of 138 months
LAGS = 12
PATH = 12  # months scored by the forward-looking loss
LAMBDAS = tuple(np.logspace(-4.0, 0.5, 7))  # ridge penalty grid for unit-variance state predictors
HEADLINE = ("pi", "pi_d1", "pi_d3", "pi_d12", "sa1", "sa3", "sa6", "pi_avg36", "pi_avg120", "pi_vol")
OUT_HORIZONS = (3, 6, 12, 24)
SUM_PHI_MAX = 0.98  # keeps the weight of the trend, 1 - sum(phi), above 0.02
CONTRIB_BASE = "2019-01"  # contributions are also reported as changes since this month


@dataclass
class StateSpec:
    name: str
    loss: str = "fwd"  # "cur" (one month ahead) or "fwd" (iterated forecasts 1..path months ahead)
    partial: bool = True  # fwd only: score origins whose path is only partly published
    theta: float = THETA
    lags: int = LAGS
    path: int = PATH
    refit_every: int = 1
    tune_every: int = 12
    lambdas: tuple = LAMBDAS
    train_start: str = "1960-01"
    kw: dict = field(default_factory=dict)


def state_zoo() -> list[StateSpec]:
    return [
        StateSpec("pistar_fwd"),  # persistent state, forward-looking loss with partial horizons (the paper)
        StateSpec("pistar_fwd_complete", partial=False),  # same, scored only once all 12 months are out
        StateSpec("pistar_cur", loss="cur"),  # persistent state, one-month-ahead loss
        StateSpec("pistar_fwd_nostate", theta=0.0),  # no memory: the trend is today's predictors only
    ]


# ------------------------------------------------------------------ algebra

def responses(phi: np.ndarray, theta: float, nmax: int) -> np.ndarray:
    """Coefficients of the iterated forecast ``s_hat[j + h]``, h = 1..nmax.

    Columns: the lags ``s[j], s[j-1], ..., s[j-P+1]``, then ``pistar[j]``, then ``alpha``.
    """
    P = len(phi)
    w = 1.0 - float(np.sum(phi))
    C = np.zeros((P + nmax, P + 2))
    for lag in range(P):
        C[P - 1 - lag, lag] = 1.0  # row P - 1 + t holds position t relative to j; t <= 0 are published
    for t in range(1, nmax + 1):
        r = P - 1 + t
        row = phi @ C[r - P : r][::-1]  # sum_p phi_p * C[t - p]
        tt = theta ** (t - 1)
        row[P] += w * tt
        row[P + 1] += w * (1.0 - tt)
        C[r] = row
    return C[P:]


def state_basis(Xz: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights of ``pistar[t0 - 1]`` and ``alpha`` and the filtered predictors ``z`` for each row from ``t0``."""
    from scipy.signal import lfilter

    n = len(Xz)
    e = theta ** np.arange(1, n + 1)
    z = lfilter([1.0 - theta], [1.0, -theta], Xz, axis=0) if theta > 0 else Xz.copy()
    return e, 1.0 - e, z


def lag_matrix(s: np.ndarray, P: int) -> np.ndarray:
    """Row ``i``: ``s[i], s[i-1], ..., s[i-P+1]`` (NaN before the start)."""
    n = len(s)
    L = np.full((n, P), np.nan)
    for p in range(P):
        L[p:, p] = s[: n - p]
    return L


def future_matrix(s: np.ndarray, H: int) -> np.ndarray:
    """Row ``i``: ``s[i+1], ..., s[i+H]`` (NaN past the end)."""
    n = len(s)
    F = np.full((n, H), np.nan)
    for h in range(1, H + 1):
        F[: n - h, h - 1] = s[h:]
    return F


class _Problem:
    """Loss rows and design pieces for one fit (all arrays local to rows t0..k)."""

    def __init__(self, s, target_ok, U, penalised, L, F, spec: StateSpec):
        self.spec, self.U, self.L = spec, U, L
        n = len(s)
        H = 1 if spec.loss == "cur" else spec.path
        self.H = H
        Fh = F[:, :H].copy()
        ok_t = np.zeros_like(Fh, dtype=bool)
        for h in range(1, H + 1):
            ok_t[: n - h, h - 1] = target_ok[h:]
        base = np.isfinite(L).all(axis=1) & np.isfinite(U).all(axis=1)
        M = ok_t & np.isfinite(Fh) & base[:, None]
        if spec.loss == "fwd" and not spec.partial:
            full = np.zeros(n, dtype=bool)
            full[: max(n - H, 0)] = True  # all H months published by the origin of the fit
            M &= full[:, None]
        self.M = M
        self.F = np.where(M, Fh, 0.0)
        self.Lz = np.where(base[:, None], L, 0.0)
        self.Uz = np.where(base[:, None], U, 0.0)
        self.penalised = penalised
        self.alpha_col = 1  # U columns: [pistar0 weight, alpha weight, z ...]

    def rows(self, lo: int = 0, hi: int | None = None, realised_by: int | None = None) -> np.ndarray:
        """Mask of loss cells for origins in [lo, hi) that are published by local row ``realised_by``."""
        M = self.M.copy()
        n = len(M)
        hi = n if hi is None else hi
        M[:lo] = False
        M[hi:] = False
        if realised_by is not None:
            j = np.arange(n)[:, None]
            h = np.arange(1, self.H + 1)[None, :]
            if self.spec.loss == "fwd" and not self.spec.partial:
                M &= (j + self.H) <= realised_by
            else:
                M &= (j + h) <= realised_by
        return M

    # ---- cur: a linear ridge in [lags, U]
    def solve_cur(self, lam: float, M: np.ndarray):
        m = M[:, 0]
        X = np.hstack([self.Lz[m], self.Uz[m]])
        y = self.F[m, 0]
        N = max(len(y), 1)
        pen = np.zeros(X.shape[1])
        pen[self.L.shape[1]:][self.penalised] = lam
        G = X.T @ X / N + np.diag(pen)
        b = np.linalg.solve(G + 1e-10 * np.eye(len(G)), X.T @ y / N)
        P = self.L.shape[1]
        phi = b[:P]
        w = max(1.0 - phi.sum(), 1.0 - SUM_PHI_MAX)
        return phi, b[P:] / w

    # ---- fwd: variable projection over phi
    def _normal(self, phi, M):
        R = responses(phi, self.spec.theta, self.H)
        A, B, Ca = R[:, : len(phi)], R[:, len(phi)], R[:, len(phi) + 1]
        Mf = M.astype(float)
        Y = (self.F - self.Lz @ A.T) * Mf
        b2, b1, c2 = Mf @ (B * B), Mf @ (B * Ca), Mf @ (Ca * Ca)
        U = self.Uz
        G = (U * b2[:, None]).T @ U
        g1 = U.T @ b1
        G[:, self.alpha_col] += g1
        G[self.alpha_col, :] += g1
        G[self.alpha_col, self.alpha_col] += c2.sum()
        r = U.T @ (Y @ B)
        r[self.alpha_col] += float(np.sum(Y @ Ca))
        return G, r, float(np.sum(Y * Y)), max(int(M.sum()), 1)

    def solve_psi(self, phi, lam, M):
        G, r, yy, N = self._normal(phi, M)
        pen = np.zeros(len(G))
        pen[self.penalised] = lam * N
        A = G + np.diag(pen) + 1e-10 * N * np.eye(len(G))
        psi = np.linalg.solve(A, r)
        loss = (yy - 2.0 * psi @ r + psi @ G @ psi + float(np.sum(pen * psi * psi))) / N
        return psi, loss

    def fit_fwd(self, lam, M, phi0):
        from scipy.optimize import minimize

        def obj(phi):
            return self.solve_psi(phi, lam, M)[1]

        cons = [{"type": "ineq", "fun": lambda p: SUM_PHI_MAX - p.sum()}, {"type": "ineq", "fun": lambda p: p.sum()}]
        x0 = np.clip(phi0, -0.95, 0.95)
        if x0.sum() > SUM_PHI_MAX - 0.01:
            x0 = x0 * (SUM_PHI_MAX - 0.02) / x0.sum()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(obj, x0, method="SLSQP", bounds=[(-0.95, 0.95)] * len(x0), constraints=cons,
                           options={"maxiter": 200, "ftol": 1e-10})
        phi = res.x if np.isfinite(res.fun) else x0
        return phi, self.solve_psi(phi, lam, M)[0]

    def fit_fwd_multi(self, lam, M, phi_prev=None):
        """``fit_fwd`` from the previous solution and from the one-month-loss solution; the lower loss wins.

        The forward loss is not convex in ``phi``: it has a persistent optimum (sum phi near 0.8) and a
        trend-led one (sum phi near 0.1).  A single warm start would make the estimate depend on the path
        of earlier fits.
        """
        starts = [self.solve_cur(lam, M)[0]]
        if phi_prev is not None:
            starts.append(np.asarray(phi_prev, float))
        best = None
        for x0 in starts:
            phi, psi = self.fit_fwd(lam, M, x0)
            loss = self.solve_psi(phi, lam, M)[1]
            if best is None or loss < best[0]:
                best = (loss, phi, psi)
        return best[1], best[2]

    def loss_of(self, phi, psi, M) -> float:
        """Mean squared error of the model's forecasts over the cells in ``M``."""
        R = responses(phi, self.spec.theta, self.H)
        A, B, Ca = R[:, : len(phi)], R[:, len(phi)], R[:, len(phi) + 1]
        pistar = self.Uz @ psi
        pred = self.Lz @ A.T + pistar[:, None] * B[None, :] + psi[self.alpha_col] * Ca[None, :]
        e = (self.F - pred)[M]
        return float(np.mean(e * e)) if e.size else np.inf


# ------------------------------------------------------------------ backtest

def _select(X: np.ndarray, lo: int, k: int, min_hist: int) -> list[int]:
    ok_now = np.isfinite(X[k])
    hist = np.isfinite(X[lo:k]).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(X[lo : k + 1], axis=0)
    return [i for i in range(X.shape[1]) if ok_now[i] and hist[i] >= min_hist and np.isfinite(sd[i]) and sd[i] > 1e-12]


def state_inputs(frame: pd.DataFrame, groups: dict[str, str]):
    """Predictor names (every grouped column except headline CPI terms and horizon-specific columns)."""
    names = [c for c in frame.columns if c in groups and c not in HEADLINE
             and not c.startswith(("drop_h", "mech_h", "tax_h"))]
    return names, [groups[c] for c in names]


def tax_impulse(frame: pd.DataFrame) -> np.ndarray:
    """Annualised monthly-rate effect of each consumption-tax/VAT change, in the row that publishes it.

    A change of ``c`` pp in the price level lifts that month's annualised rate by ``12 c``.  It is taken
    out of the series the model is estimated on, scaled by the pass-through estimated so far
    (``passthrough``); forecasts add back that share of the known change in the tax contribution to y/y
    inflation (``tax_h{h}``, announced events only).
    """
    from . import config as cfg

    c = cfg.COUNTRIES[frame.attrs["country"]]
    out = np.zeros(len(frame))
    lag = c.cpi[0].lag
    for e in c.tax_events:
        r = frame.index.searchsorted(pd.Period(e.effective, "M") + lag)
        if 0 <= r < len(frame) and frame.index[r] == pd.Period(e.effective, "M") + lag:
            out[r] += 12.0 * e.change
    return out


def passthrough(s_raw: np.ndarray, impulse: np.ndarray, k: int, window: int = 6, default: float = 1.0) -> float:
    """Share of the statutory tax change that reached the CPI, from the events published by row ``k``.

    Each event's jump is its month's rate minus the mean of the ``window`` months on either side; the
    estimate is the mean jump relative to the statutory impulse, clipped to [0, 1.5] (``default`` before any
    event is fully observed).
    """
    vals = []
    for r in np.flatnonzero(impulse != 0):
        if r - window < 0 or r + window > k or not np.isfinite(s_raw[r]):
            continue
        around = np.r_[s_raw[r - window : r], s_raw[r + 1 : r + window + 1]]
        around = around[np.isfinite(around)]
        if len(around) >= window:
            vals.append((s_raw[r] - around.mean()) / impulse[r])
    return float(np.clip(np.mean(vals), 0.0, 1.5)) if vals else default


def target_mask(frame: pd.DataFrame) -> np.ndarray:
    """False for monthly rates that touch an interpolated CPI month (inputs only, never targets)."""
    pi = frame["pi"]
    pt = frame["tgt12"].shift(12)  # y/y with interpolated months blanked
    bad = (pi.notna() & pt.isna()).to_numpy().copy()
    bad[:13] = False
    bad[1:] |= bad[:-1].copy()  # the rate into and out of a filled month (one way: no wrap-around)
    return ~bad


def yoy_forecast(pi_k: float, s_last12: np.ndarray, path: np.ndarray, h: int) -> float:
    """Y/y inflation ``h`` months after the latest print from its monthly annualised rates.

    ``pi[k+h] = pi[k] + mean(a[k+h-11 .. k+h]) - mean(a[k-11 .. k])``, with ``a`` the published rates
    (``s_last12``, oldest first) followed by the forecast ``path`` (``path[0]`` is ``k + 1``).
    """
    a = np.concatenate([s_last12, path])
    return pi_k + float(np.mean(a[h : h + 12])) - float(np.mean(s_last12))


def run_state_backtest(frame: pd.DataFrame, groups: dict[str, str], spec: StateSpec, start: str = "2000-01",
                       end: str | None = None, min_train: int = 120, min_hist: int = 60,
                       horizons=OUT_HORIZONS, explain: bool = True) -> dict[int, pd.DataFrame]:
    """One result frame per horizon, in the format of ``backtest.run_backtest``."""
    idx = frame.index
    names, gnames = state_inputs(frame, groups)
    X = frame[names].to_numpy(float)
    s_raw = frame["sa1"].to_numpy(float)
    impulse = tax_impulse(frame)
    has_tax = bool(np.any(impulse != 0))
    s, gamma = s_raw, 0.0
    pi = frame["pi"].to_numpy(float)
    tax = {h: (frame[f"tax_h{h}"].to_numpy(float) if f"tax_h{h}" in frame else np.zeros(len(frame))) for h in horizons}
    tok = target_mask(frame) & np.isfinite(s_raw)
    t0 = max(int(idx.searchsorted(pd.Period(spec.train_start, "M"))), 0)
    first = int(idx.searchsorted(pd.Period(start, "M")))
    last = len(idx) - 1 if end is None else int(idx.searchsorted(pd.Period(end, "M"), side="right")) - 1
    base_row = int(idx.searchsorted(pd.Period(CONTRIB_BASE, "M")))
    P, nmax = spec.lags, max(max(horizons), spec.path)
    rows = {h: [] for h in horizons}
    lam, phi_prev, tuned_at, fitted_at = spec.lambdas[len(spec.lambdas) // 2], None, -10**9, -10**9
    fit = None
    for k in range(first, last + 1):
        if not (np.isfinite(pi[k]) and np.all(np.isfinite(s_raw[k - 11 : k + 1]))):
            continue
        due = fit is None or k - fitted_at >= spec.refit_every
        if due:
            if has_tax:  # tax steps out of the estimation series, at the pass-through seen so far
                gamma = passthrough(s_raw, impulse, k)
                s = s_raw - gamma * impulse
            cols = _select(X, t0, k, min_hist)
            if not cols:
                continue
            Xk = X[t0 : k + 1][:, cols]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mu, sd = np.nanmean(Xk, axis=0), np.nanstd(Xk, axis=0)
            Xz = np.where(np.isfinite(Xk), (Xk - mu) / sd, 0.0)
            e, w, z = state_basis(Xz, spec.theta)
            zsd = z.std(axis=0)
            zsd = np.where(zsd > 1e-12, zsd, 1.0)
            U = np.column_stack([e, w, z / zsd])
            keep = [0, 1] + [2 + i for i in range(len(cols))]
            if spec.theta == 0.0:
                keep = keep[1:]  # no initial state to estimate
            U = U[:, keep]
            penalised = np.array([j >= (2 if spec.theta > 0 else 1) for j in range(U.shape[1])])
            sk = s[t0 : k + 1]
            prob = _Problem(sk, tok[t0 : k + 1], U, penalised, lag_matrix(sk, P), future_matrix(sk, spec.path), spec)
            if spec.theta == 0.0:
                prob.alpha_col = 0
            n_orig = int(prob.M.any(axis=1).sum())
            if n_orig < min_train:
                continue
            M_all = prob.rows()
            if k - tuned_at >= spec.tune_every:
                lam = _tune(prob, spec, M_all, phi_prev)
                tuned_at = k
            if spec.loss == "cur":
                phi, psi = prob.solve_cur(lam, M_all)
            else:
                phi, psi = prob.fit_fwd_multi(lam, M_all, phi_prev)
            phi_prev = phi
            fit = dict(cols=cols, mu=mu, sd=sd, zsd=zsd, keep=keep, phi=phi, psi=psi, U=U, n=n_orig, lam=lam)
            fitted_at = k
        else:
            # Frozen parameters: extend the state with this month's predictors.
            cols, mu, sd = fit["cols"], fit["mu"], fit["sd"]
            Xk = X[t0 : k + 1][:, cols]
            Xz = np.where(np.isfinite(Xk), (Xk - mu) / sd, 0.0)
            e, w, z = state_basis(Xz, spec.theta)
            fit["U"] = np.column_stack([e, w, z / fit["zsd"]])[:, fit["keep"]]
        phi, psi, U = fit["phi"], fit["psi"], fit["U"]
        a_col = 1 if spec.theta > 0 else 0
        pistar_k = float(U[-1] @ psi)
        R = responses(phi, spec.theta, nmax)
        lags_k = s[k - P + 1 : k + 1][::-1]
        path = R[:, :P] @ lags_k + R[:, P] * pistar_k + R[:, P + 1] * psi[a_col]
        extra = {"pistar": pistar_k, "alpha": float(psi[a_col]), "sum_phi": float(np.sum(phi)), "lambda": fit["lam"],
                 "tax_pass": gamma}
        if explain:
            extra.update(_contributions(fit, gnames, a_col, base_row - t0, spec))
        for h in horizons:
            fc = yoy_forecast(pi[k], s[k - 11 : k + 1], path, h) + gamma * float(np.nan_to_num(tax[h][k]))
            rows[h].append({"date": idx[k], "forecast": fc, "rw": pi[k],
                            "actual": frame[f"tgt{h}"].iloc[k] if f"tgt{h}" in frame else np.nan,
                            "n_train": fit["n"], "n_features": len(fit["cols"]), "refit": fitted_at == k, **extra})
    return {h: (pd.DataFrame(r).set_index("date") if r else pd.DataFrame()) for h, r in rows.items()}


def _contributions(fit, gnames, a_col, base_local, spec) -> dict:
    """Trend split by predictor group (exact: the trend is linear), at the origin and as change since 2019-01."""
    U, psi, cols = fit["U"], fit["psi"], fit["cols"]
    zpart = U[:, a_col + 1 :] * psi[a_col + 1 :][None, :]
    out = {}
    groups = sorted({gnames[c] for c in cols})
    for g in groups:
        m = np.array([gnames[c] == g for c in cols])
        series = zpart[:, m].sum(axis=1)
        out[f"c_{g}"] = float(series[-1])
        if 0 <= base_local < len(series) - 1:
            out[f"d19_{g}"] = float(series[-1] - series[base_local])
    rest = U[:, : a_col + 1] @ psi[: a_col + 1]
    out["c_base"] = float(rest[-1])
    if 0 <= base_local < len(rest) - 1:
        out["d19_base"] = float(rest[-1] - rest[base_local])
    return out


def _tune(prob: _Problem, spec: StateSpec, M_all: np.ndarray, phi_prev) -> float:
    """Ridge penalty by expanding-window validation on the model's own loss (three blocks over the last half)."""
    n = len(prob.M)
    origins = np.flatnonzero(prob.M.any(axis=1))
    if len(origins) < 120:
        return spec.lambdas[len(spec.lambdas) // 2]
    edges = np.linspace(origins[len(origins) // 2], origins[-1] + 1, 4).astype(int)
    sse = np.zeros(len(spec.lambdas))
    for v0, v1 in zip(edges[:-1], edges[1:]):
        M_fit = prob.rows(0, v0, realised_by=v0)
        M_val = prob.rows(v0, v1, realised_by=n - 1)
        if M_fit.sum() < 60 or M_val.sum() == 0:
            continue
        for i, lam in enumerate(spec.lambdas):
            if spec.loss == "cur":
                phi, psi = prob.solve_cur(lam, M_fit)
            else:  # phi re-estimated for each penalty on the fold's own training cells
                phi, psi = prob.fit_fwd_multi(lam, M_fit, phi_prev)
            sse[i] += prob.loss_of(phi, psi, M_val) * M_val.sum()
    return spec.lambdas[int(np.argmin(sse))] if sse.any() else spec.lambdas[len(spec.lambdas) // 2]


# ------------------------------------------------------------------ jobs

_FRAMES: dict = {}


def _run_state_job(job):
    (country, name), spec, kw = job
    frame, groups = _FRAMES[country]
    import threadpoolctl

    try:
        with threadpoolctl.threadpool_limits(limits=1):  # small matrices: threads only contend across processes
            return (country, name), run_state_backtest(frame, groups, spec, **kw)
    except Exception as err:  # keep the other jobs running
        log.exception("job %s failed", (country, name))
        return (country, name), err


def run_state_jobs(frames: dict, specs: list[StateSpec], countries, kw: dict, n_jobs: int = 1, progress=None) -> dict:
    """{(country, horizon, name): results} for every spec and economy."""
    global _FRAMES
    _FRAMES = frames
    jobs = [((c, s.name), s, kw) for s in specs for c in countries]
    out = {}

    def take(key, res):
        if isinstance(res, Exception):
            if progress:
                progress(key, res)
            return
        for h, r in res.items():
            out[(key[0], h, key[1])] = r
        if progress:
            progress(key, f"{sum(len(r) for r in res.values())} rows over {len(res)} horizons")

    if n_jobs <= 1:
        for j in jobs:
            take(*_run_state_job(j))
        return out
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("fork")) as pool:
        for f in as_completed([pool.submit(_run_state_job, j) for j in jobs]):
            take(*f.result())
    return out
