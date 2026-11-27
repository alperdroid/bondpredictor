"""Partial-horizon training and the persistent-state trend model: algebra, identities and no look-ahead."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bondpredictor.inflation.backtest import Spec, run_backtest
from bondpredictor.inflation.features import build_frame
from bondpredictor.inflation.partial import horizon_parts, stacked_cells
from bondpredictor.inflation.pistar import (StateSpec, _Problem, future_matrix, lag_matrix, passthrough, responses,
                                            run_state_backtest, state_basis, yoy_forecast)
from test_inflation import _synthetic

FAST = dict(lambdas=(1e-3, 1e-1), train_start="1985-01")


def _frame():
    return build_frame(_synthetic(), "US")


def _scramble(frame, k, h=12, seed=5):
    """Every predictor after origin ``k`` and every target not yet published at ``k`` replaced by noise."""
    rng = np.random.default_rng(seed)
    noisy = frame.copy()
    after = noisy.index > k
    feats = [c for c in noisy.columns if not c.startswith("tgt")]
    noisy.loc[after, feats] = rng.normal(size=(after.sum(), len(feats)))
    for c in [c for c in noisy.columns if c.startswith("tgt")]:
        hh = int(c[3:])
        late = noisy.index > k - hh
        noisy.loc[late, c] = rng.normal(size=late.sum()) * 10
    return noisy


# ------------------------------------------------------------------ algebra

def test_responses_reproduce_the_iterated_forecast():
    rng = np.random.default_rng(0)
    phi, theta = np.array([0.4, 0.2, -0.1]), 0.9
    lags = rng.normal(size=3)  # s[j], s[j-1], s[j-2]
    pistar, alpha = 2.5, 1.0
    hist = list(lags[::-1])
    for t in range(1, 9):
        trend = theta ** (t - 1) * pistar + (1 - theta ** (t - 1)) * alpha
        hist.append(sum(phi[p] * hist[-1 - p] for p in range(3)) + (1 - phi.sum()) * trend)
    R = responses(phi, theta, 8)
    assert np.allclose(R @ np.r_[lags, pistar, alpha], hist[3:])


def test_state_basis_is_the_trend_recursion():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    beta, alpha, p0 = np.array([0.5, -1.0, 2.0]), 1.5, 3.0
    for theta in (0.95, 0.0):
        e, w, z = state_basis(X, theta)
        prev, path = p0, []
        for r in range(len(X)):
            prev = theta * prev + (1 - theta) * (alpha + X[r] @ beta) if theta > 0 else alpha + X[r] @ beta
            path.append(prev)
        assert np.allclose(e * p0 + w * alpha + z @ beta, path)


def test_yoy_from_the_monthly_path_is_exact():
    rng = np.random.default_rng(2)
    logp = np.cumsum(0.002 + 0.003 * rng.normal(size=80))
    a = 1200 * np.diff(logp)  # annualised monthly rates; a[i] belongs to logp[i + 1]
    k = 40  # a[k] is the latest published rate
    pi_k = 100 * (logp[k + 1] - logp[k + 1 - 12])
    for h in (1, 3, 6, 12, 24):
        expected = 100 * (logp[k + 1 + h] - logp[k + 1 + h - 12])
        assert yoy_forecast(pi_k, a[k - 11 : k + 1], a[k + 1 : k + 1 + 24], h) == pytest.approx(expected)


def test_variable_projection_solves_the_stacked_least_squares():
    rng = np.random.default_rng(3)
    n, P, H = 150, 2, 4
    s = rng.normal(size=n).cumsum() * 0.1 + rng.normal(size=n)
    U = np.column_stack([0.99 ** np.arange(1, n + 1), 1 - 0.99 ** np.arange(1, n + 1), rng.normal(size=(n, 3))])
    spec = StateSpec("t", lags=P, path=H, theta=0.99)
    prob = _Problem(s, np.ones(n, bool), U, np.array([False, False, True, True, True]), lag_matrix(s, P),
                    future_matrix(s, H), spec)
    phi = np.array([0.3, 0.1])
    M = prob.rows()
    psi, loss = prob.solve_psi(phi, 0.0, M)
    # explicit stacked design: one row per (origin, horizon) cell
    R = responses(phi, spec.theta, H)
    D, y = [], []
    for j, h in zip(*np.nonzero(M)):
        D.append(R[h, P] * U[j] + R[h, P + 1] * np.eye(U.shape[1])[1])
        y.append(s[j + h + 1] - R[h, :P] @ prob.Lz[j])
    ref, *_ = np.linalg.lstsq(np.array(D), np.array(y), rcond=None)
    assert np.allclose(psi, ref, atol=1e-6)
    assert loss == pytest.approx(prob.loss_of(phi, psi, M), rel=1e-6)


def test_passthrough_recovers_the_share_of_a_tax_step():
    rng = np.random.default_rng(4)
    s = 2.0 + 0.3 * rng.normal(size=120)
    impulse = np.zeros(120)
    impulse[50] = 12 * 3.0  # a 3 pp step in the price level
    s[50] += 0.6 * impulse[50]
    assert passthrough(s, impulse, k=119) == pytest.approx(0.6, abs=0.03)
    assert passthrough(s, impulse, k=53) == 1.0  # the months after the step are not all out yet


# ------------------------------------------------------------------ partial horizons

def test_stacked_target_is_average_inflation_ahead_and_the_usual_target_at_12():
    frame, groups = _frame()
    _, _, _, HF, Z, pi = horizon_parts(frame, groups)
    assert np.allclose(Z[11], (frame["tgt12"] - frame["pi"]).to_numpy(), equal_nan=True)
    # Without seasonal adjustment error the identity is exact: rebuild with sa1 = the raw annualised rate.
    raw = frame.copy()
    logp = np.log(_synthetic().countries["US"].cpi.reindex(frame.index))
    raw["sa1"] = 1200 * logp.diff()
    _, _, _, _, Zr, _ = horizon_parts(raw, groups)
    a = raw["sa1"].to_numpy()
    j = 200
    for m in (1, 4, 11):
        assert Zr[m - 1, j] == pytest.approx(a[j + 1 : j + 1 + m].mean() - pi[j], abs=1e-9)


def test_stacked_cells_respect_publication():
    frame, groups = _frame()
    _, _, _, _, Z, pi = horizon_parts(frame, groups)
    ok = np.isfinite(pi)
    k = 300
    for partial in (True, False):
        J, M = stacked_cells(k, 30, 12, Z, ok, partial)
        assert np.all(J + M <= k)
        assert J.max() == (k - 1 if partial else k - 12)
        assert set(M[J == k - 12]) == set(range(1, 13))


@pytest.mark.parametrize("model", ["ridge", "lgbm"])
@pytest.mark.parametrize("stack", ["partial", "complete"])
def test_stacked_backtest_ignores_future_data(model, stack):
    frame, groups = _frame()
    k = pd.Period("2012-03", "M")
    spec = Spec(model, model, refit_every=1, train_start="1985-01", stack=stack)
    base = run_backtest(frame, groups, 12, spec, start=str(k), end=str(k), min_train=60, min_hist=24)
    again = run_backtest(_scramble(frame, k), groups, 12, spec, start=str(k), end=str(k), min_train=60, min_hist=24)
    assert base["forecast"].iloc[0] == pytest.approx(again["forecast"].iloc[0])


def test_only_partial_training_learns_from_the_latest_origins():
    frame, groups = _frame()
    k = pd.Period("2012-03", "M")
    spec = Spec("ridge", "ridge", refit_every=1, train_start="1985-01")
    run = lambda f, st: run_backtest(f, groups, 12, replace(spec, stack=st), start=str(k), end=str(k),
                                     min_train=60, min_hist=24)["forecast"].iloc[0]
    moved = frame.copy()  # change the predictors of an origin whose 12 months are not all out at k
    moved.loc[k - 5, "unemp_lvl"] += 5.0
    assert run(frame, None) == pytest.approx(run(moved, None))
    assert run(frame, "complete") == pytest.approx(run(moved, "complete"))
    assert run(frame, "partial") != pytest.approx(run(moved, "partial"))


# ------------------------------------------------------------------ state model

@pytest.mark.parametrize("spec", [StateSpec("fwd", **FAST), StateSpec("fwd_c", partial=False, **FAST),
                                  StateSpec("cur", loss="cur", **FAST), StateSpec("nostate", theta=0.0, **FAST)])
def test_state_backtest_ignores_future_data(spec):
    frame, groups = _frame()
    k = pd.Period("2012-03", "M")
    base = run_state_backtest(frame, groups, spec, start=str(k), end=str(k), min_train=60, min_hist=24)
    again = run_state_backtest(_scramble(frame, k), groups, spec, start=str(k), end=str(k), min_train=60, min_hist=24)
    for h, r in base.items():
        assert r["forecast"].iloc[0] == pytest.approx(again[h]["forecast"].iloc[0])
        assert r["rw"].iloc[0] == frame.loc[k, "pi"]


def test_state_contributions_add_up_to_the_trend():
    frame, groups = _frame()
    res = run_state_backtest(frame, groups, StateSpec("fwd", **FAST), start="2012-01", end="2012-03",
                             min_train=60, min_hist=24)[12]
    parts = res[[c for c in res.columns if c.startswith("c_")]].sum(axis=1)
    assert np.allclose(parts, res["pistar"])
    assert (res["sum_phi"] <= 0.98 + 1e-9).all()
