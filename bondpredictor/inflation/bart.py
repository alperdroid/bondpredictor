"""Bayesian additive regression trees (Chipman, George and McCulloch 2010).

BART is the non-parametric workhorse of recent Bayesian macro forecasting
(e.g. Huber, Koop, Onorante, Pfarrhofer and Schreiner 2023; Clark, Huber,
Koop, Marcellino and Pfarrhofer 2023 for tail forecasts of inflation).  It
gives a full predictive distribution, which is used for the density
forecasts in the report.

Implementation: Bayesian backfitting with grow/prune Metropolis-Hastings
moves, the default priors of Chipman et al. (m = 50 trees, alpha = 0.95,
beta = 2, k = 2, nu = 3, q = 0.9), and transition ratios as in Kapelner and
Bleich (2016, bartMachine).  Cut points are drawn from a fixed grid of
training quantiles.  Posterior draws of the trees are kept (thinned) so new
rows can be predicted after fitting.
"""

from __future__ import annotations

import math

import numpy as np

from .econ import InflationModel


class _Tree:
    """Growable binary tree; ``leaf`` holds the leaf index of every training row."""

    def __init__(self, n):
        self.var = [-1]
        self.cut = [0.0]
        self.left = [-1]
        self.right = [-1]
        self.parent = [-1]
        self.depth = [0]
        self.mu = [0.0]
        self.leaf = np.zeros(n, dtype=np.int32)

    def leaves(self):
        return [i for i in range(len(self.var)) if self.left[i] == -1 and self.var[i] != -2]

    def nog(self):
        """Nodes whose two children are both leaves."""
        return [i for i in range(len(self.var)) if self.left[i] >= 0
                and self.left[self.left[i]] == -1 and self.left[self.right[i]] == -1]

    def add_children(self, node, var, cut):
        for _ in range(2):
            self.var.append(-1)
            self.cut.append(0.0)
            self.left.append(-1)
            self.right.append(-1)
            self.parent.append(node)
            self.depth.append(self.depth[node] + 1)
            self.mu.append(0.0)
        L, R = len(self.var) - 2, len(self.var) - 1
        self.var[node], self.cut[node], self.left[node], self.right[node] = var, cut, L, R
        return L, R

    def remove_children(self, node):
        L, R = self.left[node], self.right[node]
        for c in (L, R):
            self.var[c] = -2  # dead node (indices are never reused)
        self.var[node], self.left[node], self.right[node] = -1, -1, -1
        return L, R

    def snapshot(self):
        alive = np.array(self.var)
        return (np.array(self.var, dtype=np.int32), np.array(self.cut), np.array(self.left, dtype=np.int32),
                np.array(self.right, dtype=np.int32), np.array(self.mu)) if len(alive) else None


def _route(snap, X):
    var, cut, left, right, mu = snap
    node = np.zeros(len(X), dtype=np.int32)
    while True:
        internal = left[node] >= 0
        if not internal.any():
            return mu[node]
        v = var[node[internal]]
        go_left = X[np.flatnonzero(internal), v] <= cut[node[internal]]
        node[internal] = np.where(go_left, left[node[internal]], right[node[internal]])


class BART(InflationModel):
    name = "bart"
    grid = [{}]

    def __init__(self, horizon=12, params=None, seed=0, n_trees=50, n_burn=300, n_draws=300, thin=2,
                 alpha=0.95, beta=2.0, k=2.0, nu=3.0, q=0.9, n_cuts=100, min_leaf=5, **kw):
        super().__init__(horizon, params, seed, **kw)
        self.m, self.n_burn, self.n_draws, self.thin = n_trees, n_burn, n_draws, thin
        self.alpha, self.beta, self.k, self.nu, self.q = alpha, beta, k, nu, q
        self.n_cuts, self.min_leaf = n_cuts, min_leaf

    # -- priors
    def _p_split(self, depth):
        return self.alpha * (1.0 + depth) ** (-self.beta)

    def _loglik(self, n, s, sig2, tau2):
        """Log marginal likelihood of a leaf (mean integrated out), up to terms that cancel."""
        return -0.5 * math.log(1.0 + n * tau2 / sig2) + 0.5 * tau2 * s * s / (sig2 * (sig2 + n * tau2))

    def _fit(self, Z, y, params):
        from scipy.stats import chi2

        rng = np.random.default_rng(self.seed)
        n, p = Z.shape
        self.lo, self.hi = float(y.min()), float(y.max())
        span = self.hi - self.lo if self.hi > self.lo else 1.0
        ys = (y - self.lo) / span - 0.5
        self.span = span
        tau = 0.5 / (self.k * math.sqrt(self.m))
        tau2 = tau * tau
        sig2_hat = float(np.var(ys))
        lam = sig2_hat * chi2.ppf(1 - self.q, self.nu) / self.nu
        sig2 = sig2_hat
        # Cut-point grid per variable: training quantiles.
        cuts = []
        for j in range(p):
            u = np.unique(np.quantile(Z[:, j], np.linspace(0, 1, self.n_cuts + 2)[1:-1]))
            cuts.append(u)
        splittable = [j for j in range(p) if len(cuts[j]) > 0]
        trees = [_Tree(n) for _ in range(self.m)]
        fits = np.zeros((self.m, n))  # each tree's fitted values
        total = fits.sum(axis=0)
        self.draws, self.sig_draws = [], []
        n_iter = self.n_burn + self.n_draws * self.thin
        for it in range(n_iter):
            for t_i, tree in enumerate(trees):
                resid = ys - (total - fits[t_i])
                self._mh_step(tree, Z, resid, cuts, splittable, sig2, tau2, rng)
                # Draw leaf means from their conditional posterior.
                leaves = tree.leaves()
                cnt = np.bincount(tree.leaf, minlength=len(tree.var))
                sm = np.bincount(tree.leaf, weights=resid, minlength=len(tree.var))
                for lf in leaves:
                    v = 1.0 / (cnt[lf] / sig2 + 1.0 / tau2)
                    tree.mu[lf] = rng.normal(v * sm[lf] / sig2, math.sqrt(v))
                new = np.asarray(tree.mu)[tree.leaf]
                total += new - fits[t_i]
                fits[t_i] = new
            err = ys - total
            sig2 = (self.nu * lam + err @ err) / rng.chisquare(self.nu + n)
            if it >= self.n_burn and (it - self.n_burn) % self.thin == 0:
                self.draws.append([t.snapshot() for t in trees])
                self.sig_draws.append(math.sqrt(sig2))
        self.sig_draws = np.array(self.sig_draws)

    def _mh_step(self, tree, Z, resid, cuts, splittable, sig2, tau2, rng):
        leaves = tree.leaves()
        nog = tree.nog()
        grow = len(leaves) == 1 or rng.random() < 0.5
        if grow:
            node = leaves[rng.integers(len(leaves))]
            rows = np.flatnonzero(tree.leaf == node)
            if len(rows) < 2 * self.min_leaf:
                return
            j = splittable[rng.integers(len(splittable))]
            zj = Z[rows, j]
            ok = cuts[j][(cuts[j] >= zj.min()) & (cuts[j] < zj.max())]
            if len(ok) == 0:
                return
            c = ok[rng.integers(len(ok))]
            go_left = zj <= c
            nL, nR = int(go_left.sum()), int((~go_left).sum())
            if nL < self.min_leaf or nR < self.min_leaf:
                return
            sL, sR = float(resid[rows[go_left]].sum()), float(resid[rows[~go_left]].sum())
            d = tree.depth[node]
            lik = (self._loglik(nL, sL, sig2, tau2) + self._loglik(nR, sR, sig2, tau2)
                   - self._loglik(nL + nR, sL + sR, sig2, tau2))
            ps = self._p_split(d)
            struct = math.log(ps) + 2 * math.log(1 - self._p_split(d + 1)) - math.log(1 - ps)
            # Proposed tree: one more leaf; nodes with two leaf children = current ones,
            # minus node's parent (if it was one), plus node.
            par = tree.parent[node]
            w2_new = len(nog) + 1 - (1 if par >= 0 and par in nog else 0)
            p_grow = 1.0 if len(leaves) == 1 else 0.5
            trans = math.log(0.5 / p_grow) + math.log(len(leaves)) - math.log(w2_new)
            if math.log(rng.random()) < lik + struct + trans:
                L, R = tree.add_children(node, j, c)
                tree.leaf[rows[go_left]] = L
                tree.leaf[rows[~go_left]] = R
        else:
            if not nog:
                return
            node = nog[rng.integers(len(nog))]
            L, R = tree.left[node], tree.right[node]
            inL, inR = tree.leaf == L, tree.leaf == R
            nL, nR = int(inL.sum()), int(inR.sum())
            sL, sR = float(resid[inL].sum()), float(resid[inR].sum())
            d = tree.depth[node]
            lik = (self._loglik(nL + nR, sL + sR, sig2, tau2)
                   - self._loglik(nL, sL, sig2, tau2) - self._loglik(nR, sR, sig2, tau2))
            ps = self._p_split(d)
            struct = -(math.log(ps) + 2 * math.log(1 - self._p_split(d + 1)) - math.log(1 - ps))
            n_leaves_new = len(leaves) - 1
            p_grow_new = 1.0 if n_leaves_new == 1 else 0.5
            trans = math.log(p_grow_new / 0.5) + math.log(len(nog)) - math.log(n_leaves_new)
            if math.log(rng.random()) < lik + struct + trans:
                tree.remove_children(node)
                tree.leaf[inL | inR] = node

    def _sum_draws(self, Z):
        return np.array([sum(_route(s, Z) for s in draw) for draw in self.draws])  # (draws, n)

    def _predict(self, Z):
        f = self._sum_draws(Z).mean(axis=0)
        return (f + 0.5) * self.span + self.lo

    def predict_draws(self, X, seed: int = 0):
        """Posterior predictive draws for raw rows ``X``: (draws, rows), in target units."""
        Z = self._prep(np.asarray(X, float))
        f = self._sum_draws(Z)
        rng = np.random.default_rng(seed)
        f = f + rng.normal(size=f.shape) * self.sig_draws[:, None]
        z = (f + 0.5) * self.span + self.lo
        return self.ybar + self.ysd * z

    def predict_quantiles(self, X, qs=(0.1, 0.5, 0.9)):
        return np.quantile(self.predict_draws(X), qs, axis=0).T
