"""Hemisphere neural network: a "neural Phillips curve" (Goulet Coulombe 2025, JBES).

The forecast is a sum of hemispheres, each a small deep network that sees
only one economically meaningful block of predictors:

    change in inflation = b + h_own(own dynamics) + h_expect(surveys)
                          + h_slack(real activity) + h_cost(energy, import
                          prices, FX) + h_money(rates, money) + h_global(...)
                          + linear tax-calendar term

so every forecast decomposes into contributions (e.g. how much of a
predicted rise comes from slack versus cost pressure).  The plain MLP below
is the same machinery with a single hemisphere over all predictors.

Small-sample safeguards: inputs standardised and clamped to the training
range; dropout and weight decay; an ensemble of networks trained on
moving-block bootstrap samples, each early-stopped on its out-of-bag blocks
(as in the paper).
"""

from __future__ import annotations

import numpy as np

from .econ import InflationModel

HEMISPHERES = ("own", "expect", "slack", "cost", "money", "global")


def _torch():
    import torch

    torch.set_num_threads(1)
    return torch


class _Net:
    """Builds the hemisphere network lazily so importing this module does not need torch."""

    @staticmethod
    def build(torch, blocks: list[list[int]], linear: list[int], width: int, depth: int, dropout: float):
        nn = torch.nn

        class Hemi(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = [torch.tensor(b, dtype=torch.long) for b in blocks]
                self.lin_idx = torch.tensor(linear, dtype=torch.long)
                nets = []
                for b in blocks:
                    layers, d_in = [], len(b)
                    for _ in range(depth):
                        layers += [nn.Linear(d_in, width), nn.ReLU(), nn.Dropout(dropout)]
                        d_in = width
                    layers.append(nn.Linear(d_in, 1))
                    nets.append(nn.Sequential(*layers))
                self.nets = nn.ModuleList(nets)
                self.lin = nn.Linear(len(linear), 1, bias=False) if linear else None
                self.bias = nn.Parameter(torch.zeros(1))

            def parts(self, x):
                out = [net(x[:, idx]) for net, idx in zip(self.nets, self.blocks)]
                if self.lin is not None:
                    out.append(self.lin(x[:, self.lin_idx]))
                return torch.cat(out, dim=1)

            def forward(self, x):
                return self.parts(x).sum(dim=1) + self.bias

        return Hemi()


class HemisphereNN(InflationModel):
    name = "hnn"
    clip = True
    grid = [{"width": 32, "depth": 2, "dropout": 0.2, "weight_decay": 1e-3}]
    hemispheres: tuple[str, ...] | None = HEMISPHERES  # None: one hemisphere over everything (plain MLP)

    def __init__(self, horizon=12, params=None, seed=0, n_bags=10, block=12, max_epochs=400, patience=40,
                 lr=3e-3, **kw):
        super().__init__(horizon, params, seed, **kw)
        self.n_bags, self.block, self.max_epochs, self.patience, self.lr = n_bags, block, max_epochs, patience, lr

    def _layout(self):
        if self.hemispheres is None:
            return [list(range(len(self.names)))], [], ["all"]
        blocks, labels = [], []
        for h in self.hemispheres:
            idx = [i for i, g in enumerate(self.groups) if g == h]
            if idx:
                blocks.append(idx)
                labels.append(h)
        linear = [i for i, g in enumerate(self.groups) if g not in self.hemispheres]
        if linear:
            labels.append("calendar")
        return blocks, linear, labels

    def _fit(self, Z, y, params):
        torch = _torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        blocks, linear, self.labels = self._layout()
        X = torch.tensor(Z, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)
        n = len(y)
        n_blocks = int(np.ceil(n / self.block))
        self.nets = []
        for _ in range(self.n_bags):
            # Moving-block bootstrap; out-of-bag blocks are the early-stopping set.
            chosen = rng.integers(0, n_blocks, size=n_blocks)
            inbag = np.concatenate([np.arange(b * self.block, min((b + 1) * self.block, n)) for b in chosen])
            oob = np.setdiff1d(np.arange(n), inbag)
            if len(oob) < 12:
                oob = rng.choice(n, size=max(12, n // 5), replace=False)
            net = _Net.build(torch, blocks, linear, params["width"], params["depth"], params["dropout"])
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=params["weight_decay"])
            xi, yi = X[inbag], Y[inbag]
            xo, yo = X[oob], Y[oob]
            best, best_state, wait = np.inf, None, 0
            for _epoch in range(self.max_epochs):
                net.train()
                opt.zero_grad()
                loss = torch.mean((net(xi) - yi) ** 2)
                loss.backward()
                opt.step()
                net.eval()
                with torch.no_grad():
                    val = float(torch.mean((net(xo) - yo) ** 2))
                if val < best - 1e-6:
                    best, wait = val, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.nets.append(net)

    def _predict(self, Z):
        torch = _torch()
        X = torch.tensor(Z, dtype=torch.float32)
        with torch.no_grad():
            return np.mean([net(X).numpy() for net in self.nets], axis=0)

    def contributions(self, X) -> tuple[list[str], np.ndarray]:
        """Each hemisphere's contribution to the forecast for raw rows ``X`` (target units).

        Contributions plus the constant (``ybar`` + bias) add up to the forecast.
        """
        torch = _torch()
        Z = torch.tensor(self._prep(np.asarray(X, float)), dtype=torch.float32)
        with torch.no_grad():
            parts = np.mean([net.parts(Z).numpy() for net in self.nets], axis=0)
        return self.labels, parts * self.ysd


class MLP(HemisphereNN):
    name = "mlp"
    hemispheres = None
    grid = [{"width": 64, "depth": 2, "dropout": 0.2, "weight_decay": 1e-3}]
