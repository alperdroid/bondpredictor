import numpy as np
import pandas as pd
import pytest

from bondpredictor.evaluation import clark_west, oos_r2, timing_strategy
from bondpredictor.features import inflation_uncertainty, one_sided_hp, trend_inflation
from bondpredictor.inference import hodrick_1b, newey_west, null_bootstrap, predictive_regression
from bondpredictor.models import OLS, PCR, Combination, Lasso, Ridge


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = 0.5 + X @ np.array([1.0, -0.5, 0.0]) + 0.1 * rng.normal(size=400)
    return X, y


def test_ols_recovers_coefficients(linear_data):
    X, y = linear_data
    m = OLS().fit(X, y)
    pred = m.predict(np.array([[1.0, 1.0, 1.0]]))[0]
    assert pred == pytest.approx(0.5 + 1.0 - 0.5, abs=0.05)


def test_ridge_small_penalty_equals_ols_and_pcr_full_rank(linear_data):
    X, y = linear_data
    x0 = X[:5]
    ols = OLS().fit(X, y).predict(x0)
    assert np.allclose(Ridge(alpha=1e-10).fit(X, y).predict(x0), ols, atol=1e-6)
    assert np.allclose(PCR(k=3).fit(X, y).predict(x0), ols, atol=1e-8)


def test_ridge_cv_and_lasso_shrinkage(linear_data):
    X, y = linear_data
    r = Ridge(horizon=1).fit(X, y)
    assert r.alpha_ in Ridge.alphas
    lasso = Lasso(alpha=0.02, horizon=1).fit(X, y)
    assert abs(lasso.beta[2]) < 1e-3 < abs(lasso.beta[0])  # irrelevant predictor dropped
    assert np.all(Lasso(alpha=10.0).fit(X, y).beta == 0)


def test_combination_is_mean_of_univariate_forecasts(linear_data):
    X, y = linear_data
    x0 = X[:3]
    uni = [OLS().fit(X[:, [j]], y).predict(x0[:, [j]]) for j in range(3)]
    assert np.allclose(Combination().fit(X, y).predict(x0), np.mean(uni, axis=0))


def test_newey_west_zero_lags_is_white():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(200), rng.normal(size=200)])
    u = rng.normal(size=200) * (1 + np.abs(X[:, 1]))
    V = newey_west(X, u, 0)
    Q = np.linalg.inv(X.T @ X)
    white = Q @ (X.T * u**2) @ X @ Q
    assert np.allclose(V, white)


def _overlapping_sample(T=480, h=12, beta=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.95 * x[t - 1] + rng.normal()
    r1 = beta * x + rng.normal(size=T)  # r1[t]: return t -> t+1
    idx = pd.period_range("1980-01", periods=T, freq="M")
    r1 = pd.Series(r1, index=idx)
    rh = r1.rolling(h).sum().shift(-(h - 1))
    return pd.DataFrame({"x": x}, index=idx), r1, rh


def test_predictive_regression_reports_hodrick():
    X, r1, rh = _overlapping_sample(beta=0.3)
    res = predictive_regression(rh, X, 12, one_period=r1)
    assert set(res.table.columns) == {"coef", "t_nw", "t_hodrick"}
    assert res.table.loc["x", "t_hodrick"] > 2 and res.r2 > 0.1


def test_hodrick_1b_positive_definite():
    X, r1, _ = _overlapping_sample()
    A = np.column_stack([np.ones(len(X)), X["x"].to_numpy()])
    V = hodrick_1b(A, (r1 - r1.mean()).to_numpy(), 12)
    assert np.all(np.linalg.eigvalsh(V) > 0)


def test_bootstrap_rejects_true_predictor_and_not_null():
    X, r1, rh = _overlapping_sample(beta=0.3, seed=3)
    res = predictive_regression(rh, X, 12)
    p = null_bootstrap(X, r1, 12, res, n_boot=100)
    assert p["x"] < 0.05
    X0, r10, rh0 = _overlapping_sample(beta=0.0, seed=4)
    p0 = null_bootstrap(X0, r10, 12, predictive_regression(rh0, X0, 12), n_boot=100)
    assert p0["x"] > 0.05


def test_oos_r2_and_clark_west():
    rng = np.random.default_rng(2)
    signal = rng.normal(size=300)
    actual = signal + rng.normal(size=300)
    assert oos_r2(actual, signal, np.zeros(300)) > 0.3
    t, p = clark_west(actual, signal, np.zeros(300), 1)
    assert t > 3 and p < 0.01


def test_timing_strategy_weights_follow_forecast():
    idx = pd.period_range("2000-01", periods=120, freq="M")
    rx = pd.Series(np.random.default_rng(0).normal(0.002, 0.02, 120), index=idx)
    fc = pd.Series(np.where(np.arange(120) % 2, 0.05, -0.05), index=idx)
    s = timing_strategy(fc, rx, 12, bounds=(-1, 2))
    assert (s["weight"][fc.reindex(s.index) > 0] > 0).all()
    assert s["weight"].between(-1, 2).all()


def test_trend_inflation_and_hp_filter():
    idx = pd.period_range("1990-01", periods=200, freq="M")
    const = pd.Series(0.02, index=idx)
    assert np.allclose(trend_inflation(const).dropna(), 0.02)
    line = pd.Series(np.arange(80, dtype=float), index=pd.period_range("1990Q1", periods=80, freq="Q"))
    trend = one_sided_hp(line, 1600)
    assert np.allclose(trend.iloc[20:], line.iloc[20:], atol=1e-3)


def test_inflation_uncertainty_rises_with_volatility():
    rng = np.random.default_rng(5)
    idx = pd.period_range("1980-01", periods=300, freq="M")
    shocks = np.r_[0.001 * rng.normal(size=150), 0.005 * rng.normal(size=150)]
    pi = pd.Series(0.02 + np.cumsum(shocks) * 0.1 + shocks, index=idx)
    u = inflation_uncertainty(pi)
    assert u.iloc[-1] > 2 * u.iloc[140]
