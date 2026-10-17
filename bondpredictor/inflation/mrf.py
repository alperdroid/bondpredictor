"""Macroeconomic Random Forest (Goulet Coulombe 2024, "The Macroeconomy as a Random Forest", JAE).

A small linear model ``y_t = X_t beta_t`` whose coefficients are
"generalised time-varying parameters": ``beta_t`` is the average, over a
bagged forest, of the leaf coefficients selected by the state variables
``S_t`` (all predictors).  Every node holds a ridge regression on ``X``,
and splits are chosen to minimise the children's ridge residuals.  Because
the forecast is linear in ``X`` within each leaf, the model can extrapolate
(a plain tree forest cannot), which matters for an inflation surge.

As in the paper:

* ridge penalty on the non-intercept leaf coefficients;
* "random-walk" regularisation: leaf coefficients are estimated with the
  leaf's neighbours in time added at weights ``zeta`` (order 1) and
  ``zeta**2`` (order 2), so ``beta_t`` evolves smoothly;
* trees grown on moving-block bootstrap samples, ``mtry`` = 1/3.

Here ``X`` is an intercept plus a few interpretable terms (by default the
base-effect/momentum term, the latest inflation rate, the unemployment gap
and oil inflation).  Split search uses prefix sums of ``X'X`` and ``X'y``
over each candidate ordering, so each split costs a batch of k x k solves.
"""

from __future__ import annotations

import numpy as np

from .econ import InflationModel

LINEAR_TERMS = ("mech", "pi", "unemp_gap", "oil_yoy")


def _ridge_solve(XtX, Xty, lam_diag):
    """Batched ridge: XtX (..., k, k), Xty (..., k), penalty diagonal (..., k)."""
    A = XtX + lam_diag[..., :, None] * np.eye(XtX.shape[-1])
    return np.linalg.solve(A, Xty[..., None])[..., 0]


def _sse(yty, XtX, Xty, b):
    return yty - 2 * np.einsum("...k,...k->...", b, Xty) + np.einsum("...k,...kl,...l->...", b, XtX, b)


class _Tree:
    def __init__(self, rng, mtry, min_leaf, lam, n_cuts):
        self.rng, self.mtry, self.min_leaf, self.lam, self.n_cuts = rng, mtry, min_leaf, lam, n_cuts
        self.nodes = []  # (feature, threshold, left, right) or leaf marker

    def _penalty(self, k, n):
        """Ridge penalty lam * n on the slopes (n may be an array of node sizes); none on the intercept."""
        n = np.asarray(n, float)[..., None]
        lam = np.broadcast_to(self.lam * n, n.shape[:-1] + (k,)).copy()
        lam[..., 0] = 1e-8
        return lam

    def grow(self, X, y, S):
        self.nodes = []
        self._grow(np.arange(len(y)), X, y, S)
        return self

    def _grow(self, idx, X, y, S):
        node_id = len(self.nodes)
        self.nodes.append(None)
        best = self._best_split(idx, X, y, S)
        if best is None:
            self.nodes[node_id] = ("leaf", idx)
            return node_id
        j, thr = best
        go_left = S[idx, j] <= thr
        left = self._grow(idx[go_left], X, y, S)
        right = self._grow(idx[~go_left], X, y, S)
        self.nodes[node_id] = ("split", j, thr, left, right)
        return node_id

    def _best_split(self, idx, X, y, S):
        n, k = len(idx), X.shape[1]
        if n < 2 * self.min_leaf:
            return None
        Xn, yn = X[idx], y[idx]
        tot_xx, tot_xy, tot_yy = Xn.T @ Xn, Xn.T @ yn, yn @ yn
        b0 = _ridge_solve(tot_xx, tot_xy, self._penalty(k, n))
        parent = _sse(tot_yy, tot_xx, tot_xy, b0)
        best_gain, best = 1e-9 * tot_yy, None
        feats = self.rng.choice(S.shape[1], size=max(1, int(self.mtry * S.shape[1])), replace=False)
        outer = np.einsum("ni,nj->nij", Xn, Xn)
        for j in feats:
            s = S[idx, j]
            ok = np.isfinite(s)
            if ok.sum() < 2 * self.min_leaf:
                continue
            order = np.argsort(np.where(ok, s, np.inf), kind="stable")
            ss = s[order]
            # Candidate cut positions: quantiles of the node, at least min_leaf on each side.
            m = ok.sum()
            pos = np.unique(np.linspace(self.min_leaf, m - self.min_leaf, self.n_cuts).astype(int))
            pos = pos[(pos >= self.min_leaf) & (pos <= m - self.min_leaf)]
            pos = pos[ss[pos - 1] < ss[pos]]  # only between distinct values
            if len(pos) == 0:
                continue
            cxx = np.cumsum(outer[order], axis=0)
            cxy = np.cumsum(Xn[order] * yn[order, None], axis=0)
            cyy = np.cumsum(yn[order] ** 2)
            lxx, lxy, lyy = cxx[pos - 1], cxy[pos - 1], cyy[pos - 1]
            rxx, rxy, ryy = tot_xx - lxx, tot_xy - lxy, tot_yy - lyy
            bl = _ridge_solve(lxx, lxy, self._penalty(k, pos))
            br = _ridge_solve(rxx, rxy, self._penalty(k, n - pos))
            sse = _sse(lyy, lxx, lxy, bl) + _sse(ryy, rxx, rxy, br)
            i = int(np.argmin(sse))
            gain = parent - sse[i]
            if gain > best_gain:
                best_gain, best = gain, (j, 0.5 * (ss[pos[i] - 1] + ss[pos[i]]))
        return best

    def finish(self, X, y, time_pos, zeta):
        """Leaf coefficients with random-walk regularisation (time neighbours at weights zeta, zeta^2)."""
        k = X.shape[1]
        lookup = {t: i for i, t in enumerate(time_pos)}
        for nid, node in enumerate(self.nodes):
            if node[0] != "leaf":
                continue
            idx = node[1]
            w = {int(i): 1.0 for i in idx}
            if zeta > 0:
                in_leaf = {int(time_pos[i]) for i in idx}
                for i in idx:
                    t = int(time_pos[i])
                    for d, wt in ((1, zeta), (2, zeta * zeta)):
                        for nb in (t - d, t + d):
                            if nb not in in_leaf and nb in lookup:
                                j = lookup[nb]
                                w[j] = max(w.get(j, 0.0), wt)
            rows = np.fromiter(w.keys(), int)
            ww = np.fromiter(w.values(), float)
            Xw = X[rows] * ww[:, None]
            b = _ridge_solve(Xw.T @ X[rows], Xw.T @ y[rows], self._penalty(k, ww.sum()))
            self.nodes[nid] = ("leaf", b)
        return self

    def leaf_beta(self, s):
        node = self.nodes[0]
        while node[0] == "split":
            _, j, thr, left, right = node
            v = s[j]
            node = self.nodes[left if (np.isfinite(v) and v <= thr) else right]
        return node[1]


class MacroRandomForest(InflationModel):
    name = "mrf"
    nan_ok = True  # state variables may be missing; the linear part is filled below
    grid = [{"min_leaf": 20, "lam": 0.1}, {"min_leaf": 40, "lam": 0.1}]

    def __init__(self, horizon=12, params=None, seed=0, n_trees=50, zeta=0.75, block=12, mtry=1 / 3, n_cuts=20,
                 linear_terms=LINEAR_TERMS, **kw):
        super().__init__(horizon, params, seed, **kw)
        self.n_trees, self.zeta, self.block, self.mtry, self.n_cuts = n_trees, zeta, block, mtry, n_cuts
        self.linear_terms = linear_terms
        self.n_folds = 2

    def _design(self, Z):
        cols = self._cols(self.linear_terms)
        L = np.where(np.isfinite(Z[:, cols]), Z[:, cols], 0.0)
        return np.column_stack([np.ones(len(Z)), L])

    def _fit(self, Z, y, params, n_trees=None):
        rng = np.random.default_rng(self.seed)
        X = self._design(Z)
        n = len(y)
        self.trees = []
        for _ in range(n_trees or self.n_trees):
            # Moving-block bootstrap of the time index.
            starts = rng.integers(0, max(n - self.block, 1), size=int(np.ceil(n / self.block)))
            boot = np.concatenate([np.arange(s, min(s + self.block, n)) for s in starts])[:n]
            t = _Tree(rng, self.mtry, params["min_leaf"], params["lam"], self.n_cuts)
            t.grow(X[boot], y[boot], Z[boot])
            t.finish(X[boot], y[boot], boot, self.zeta)
            self.trees.append(t)

    def _cv(self, Z, y):
        # Cheaper forests for tuning.
        if len(self.grid) == 1:
            return dict(self.grid[0])
        from ..models import cv_folds

        sse = np.zeros(len(self.grid))
        for end_fit, v0, v1 in cv_folds(len(y), self.horizon, self.n_folds, 0.5, min_fit=60):
            for i, p in enumerate(self.grid):
                self._fit(Z[:end_fit], y[:end_fit], p, n_trees=max(10, self.n_trees // 3))
                sse[i] += np.sum((y[v0:v1] - self._predict(Z[v0:v1])) ** 2)
        return dict(self.grid[int(np.argmin(sse))])

    def betas(self, Z):
        """Average leaf coefficients per row (intercept first), on standardised inputs."""
        return np.array([np.mean([t.leaf_beta(z) for t in self.trees], axis=0) for z in Z])

    def _predict(self, Z):
        X = self._design(Z)
        return np.einsum("nk,nk->n", X, self.betas(Z))

    def gtvp(self, X):
        """Time-varying coefficients for raw rows ``X``, in the target's units per standard deviation of each term."""
        Z = self._prep(np.asarray(X, float))
        names = ["intercept"] + [n for n in self.names if n in self.linear_terms]
        return names, self.betas(Z) * self.ysd
