import numpy as np
import pandas as pd
import pytest

from bondpredictor.backtest import ModelSpec, information_set, run_backtest
from bondpredictor.robustness import diebold_mariano, spa_test, stationary_bootstrap_indices

from test_backtest import synthetic_frame

sklearn = pytest.importorskip("sklearn")


def test_ml_models_fit_and_clip():
    from bondpredictor.ml import GradientBoosting, NeuralNet, RandomForest

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = 0.5 * X[:, 0] + 0.1 * rng.normal(size=300)
    for cls in (RandomForest, GradientBoosting, NeuralNet):
        m = cls(horizon=1).fit(X, y)
        assert m.tuned_ in cls.grid
        assert np.corrcoef(m.predict(X[:50]), y[:50])[0, 1] > 0.8
    # Clamping: an input far outside the training range predicts like the edge.
    nn = NeuralNet(horizon=1, params=NeuralNet.grid[0], seeds=(0,)).fit(X, y)
    edge = np.array([[X[:, 0].max(), 0, 0, 0]])
    far = np.array([[X[:, 0].max() * 50, 0, 0, 0]])
    assert nn.predict(far)[0] == pytest.approx(nn.predict(edge)[0])
    nn_free = NeuralNet(horizon=1, params=NeuralNet.grid[0], seeds=(0,), clip=False).fit(X, y)
    assert abs(nn_free.predict(far)[0]) > abs(nn.predict(far)[0])


def test_refit_every_freezes_model_without_lookahead():
    f = synthetic_frame()
    spec = ModelSpec("rf", "A.rx12_n10", ["A.x"], estimator="rf", estimator_kw={"params": {"max_depth": 2, "min_samples_leaf": 10}},
                     refit_every=12)
    res = run_backtest(f, spec, start="1992-01", end="1996-12", min_train=60)
    # Scramble everything after an origin; forecasts up to it must not move.
    origin = pd.Period("1994-06", "M")
    g = f.copy()
    g.loc[g.index > origin, :] = 99.0
    g.loc[g.index > origin - 12, "A.rx12_n10"] = 99.0
    res2 = run_backtest(g, spec, start="1992-01", end=str(origin), min_train=60)
    assert np.allclose(res.loc[:origin, "forecast"].to_numpy(), res2["forecast"].to_numpy())


def test_rolling_window_information_set():
    f = synthetic_frame()
    info = information_set(f, 200, window=60)
    assert len(info) == 60 and info.index[-1] == f.index[200]
    assert info["A.rx12_n10"].last_valid_index() == f.index[188]


def test_diebold_mariano_detects_better_forecast():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=400)
    actual = signal + rng.normal(size=400)
    stat, p = diebold_mariano(actual, np.zeros(400), signal, 1)
    assert stat > 3 and p < 0.01
    stat0, p0 = diebold_mariano(actual, signal + 0.01 * rng.normal(size=400), signal, 1)
    assert p0 > 0.05


def test_stationary_bootstrap_indices():
    idx = stationary_bootstrap_indices(500, 10, np.random.default_rng(0))
    assert idx.min() >= 0 and idx.max() < 500
    runs = np.mean(np.diff(idx) == 1)
    assert 0.8 < runs < 0.97  # mostly consecutive, new block w.p. 1/10


def test_spa_rejects_only_when_a_model_is_better():
    rng = np.random.default_rng(2)
    T = 400
    base = rng.normal(size=T) ** 2
    # Competitors no better than the benchmark (size checked by simulation: ~6% at 5%).
    noise_models = {f"m{i}": base + 0.05 + rng.normal(0, 0.3, T) for i in range(5)}
    null = pd.DataFrame({"bench": base, **noise_models})
    res0 = spa_test(null, "bench", n_boot=500)
    assert res0["p_consistent"] > 0.2
    better = null.copy()
    better["good"] = base - 0.3 + rng.normal(0, 0.3, T)
    res1 = spa_test(better, "bench", n_boot=500)
    assert res1["best"] == "good" and res1["p_consistent"] < 0.01
    assert res1["p_lower"] <= res1["p_consistent"] <= res1["p_upper"]
