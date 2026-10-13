import numpy as np
import pandas as pd
import pytest

from bondpredictor.acm import acm
from bondpredictor.backtest import ModelSpec, PanelSpec, information_set, run_backtest, run_panel_backtest
from bondpredictor.curves import TabulatedCurve
from bondpredictor.learned import PCAFactors, ReturnFactor, Transformer


def synthetic_frame(T=360, countries=("A", "B"), beta=0.4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1980-01", periods=T, freq="M")
    cols = {}
    for c in countries:
        x = np.zeros(T)
        for t in range(1, T):
            x[t] = 0.9 * x[t - 1] + rng.normal()
        r1 = pd.Series(0.002 + beta * 0.01 * x + 0.02 * rng.normal(size=T), index=idx)
        cols[f"{c}.x"] = pd.Series(x, index=idx)
        cols[f"{c}.noise"] = pd.Series(rng.normal(size=T), index=idx)
        cols[f"{c}.rx1_n10"] = r1
        cols[f"{c}.rx12_n10"] = r1.rolling(12).sum().shift(-11)
    return pd.DataFrame(cols)


def test_drop_unrealized_blanks_returns_past_cutoff():
    from bondpredictor.dataset import drop_unrealized

    f = synthetic_frame()
    cut = f.index[200]
    g = drop_unrealized(f, cut)
    assert g["A.rx12_n10"].last_valid_index() == cut - 12
    assert g["A.rx1_n10"].last_valid_index() == cut - 1
    assert g["A.x"].equals(f["A.x"])


def test_information_set_masks_unrealised_returns():
    f = synthetic_frame()
    info = information_set(f, 100)
    assert info["A.rx12_n10"].last_valid_index() == f.index[100 - 12]
    assert info["A.rx1_n10"].last_valid_index() == f.index[99]
    assert info["A.x"].last_valid_index() == f.index[100]


class Spy(Transformer):
    outputs = ["A.spy"]
    seen: list = []

    def inputs(self):
        return ["A.rx12_n10"]

    def fit(self, info):
        Spy.seen.append((info.index[-1], info["A.rx12_n10"].last_valid_index()))
        return self

    def transform(self, data):
        return pd.DataFrame({"A.spy": 0.0}, index=data.index)


def test_no_lookahead_in_transformers_or_forecasts():
    f = synthetic_frame()
    Spy.seen = []
    spec = ModelSpec("t", "A.rx12_n10", ["A.x", "A.spy"], [Spy()], "ols")
    res = run_backtest(f, spec, start="1992-01", min_train=60)
    assert Spy.seen and all(last <= t - 12 for t, last in Spy.seen)

    # Scrambling everything that is unknown at an origin must not move its forecast.
    origin = pd.Period("1995-06", "M")
    g = f.copy()
    later = g.index > origin
    g.loc[later, :] = np.random.default_rng(9).normal(size=(later.sum(), g.shape[1]))
    g.loc[g.index > origin - 12, "A.rx12_n10"] = 99.0
    g.loc[g.index > origin - 1, "A.rx1_n10"] = 99.0
    res2 = run_backtest(g, spec, start="1992-01", end=str(origin), min_train=60)
    common = res.loc[: origin, "forecast"]
    assert np.allclose(common.to_numpy(), res2["forecast"].to_numpy())


def test_backtest_finds_real_predictor():
    from bondpredictor.evaluation import oos_r2

    f = synthetic_frame(T=480, beta=0.6)
    res = run_backtest(f, ModelSpec("x", "A.rx12_n10", ["A.x"]), start="1990-01", min_train=96).dropna()
    assert oos_r2(res["actual"], res["forecast"], res["benchmark"]) > 0.1


def test_feature_missing_at_origin_is_dropped():
    f = synthetic_frame()
    f.loc[f.index[-30:], "A.noise"] = np.nan
    res = run_backtest(f, ModelSpec("x", "A.rx12_n10", ["A.x", "A.noise"]), start="2005-01", min_train=60)
    assert res["features"].iloc[-1] == "A.x" and "A.noise" in res["features"].iloc[0]


def test_return_factor_recovers_cp_weights():
    rng = np.random.default_rng(1)
    T = 400
    F = rng.normal(size=(T, 3))
    gamma = np.array([0.1, -1.0, 2.0, -0.5])
    df = pd.DataFrame(F, columns=["f1", "f2", "f3"], index=pd.period_range("1980-01", periods=T, freq="M"))
    df["r1"] = gamma[0] + F @ gamma[1:] + 0.01 * rng.normal(size=T)
    df["r2"] = df["r1"] + 0.01 * rng.normal(size=T)
    rf = ReturnFactor(["f1", "f2", "f3"], ["r1", "r2"], "cp").fit(df)
    assert np.allclose(rf.beta, gamma, atol=0.01)


def test_pca_factors_skip_discontinued_series():
    rng = np.random.default_rng(2)
    T = 300
    common = rng.normal(size=T)
    idx = pd.period_range("1990-01", periods=T, freq="M")
    df = pd.DataFrame({f"Z.m_{i}": common + 0.3 * rng.normal(size=T) for i in range(6)}, index=idx)
    df.loc[idx[-24:], "Z.m_0"] = np.nan  # discontinued
    pca = PCAFactors(r"^Z\.m_", 2, "Z.mf").fit(df)
    assert "Z.m_0" not in pca.cols and len(pca.cols) == 5
    out = pca.transform(df)
    assert out.columns.tolist() == ["Z.mf1", "Z.mf2"]
    assert abs(np.corrcoef(out["Z.mf1"].iloc[-100:], common[-100:])[0, 1]) > 0.9


def test_panel_backtest_runs_for_each_country():
    f = synthetic_frame(T=360)
    spec = PanelSpec("p", ["A", "B"], "{c}.rx12_n10", ["{c}.x"], estimator="ols")
    out = run_panel_backtest(f, spec, start="1995-01", min_train=96)
    assert set(out) == {"A", "B"}
    assert out["A"]["features"].iloc[-1] == "x"


def test_acm_on_synthetic_factor_curve():
    rng = np.random.default_rng(3)
    T = 360
    idx = pd.period_range("1980-01", periods=T, freq="M")
    lvl, slp, cur = (np.zeros(T) for _ in range(3))
    lvl[0], slp[0] = 0.05, -0.01
    for t in range(1, T):
        lvl[t] = 0.0005 + 0.99 * lvl[t - 1] + 0.001 * rng.normal()
        slp[t] = -0.0002 + 0.97 * slp[t - 1] + 0.002 * rng.normal()
        cur[t] = 0.95 * cur[t - 1] + 0.003 * rng.normal()
    m = np.arange(1, 121) / 12.0
    lam = 0.6
    l1 = (1 - np.exp(-lam * m)) / (lam * m)
    l2 = l1 - np.exp(-lam * m)
    Y = lvl[:, None] + slp[:, None] * l1[None, :] + cur[:, None] * l2[None, :]
    curve = TabulatedCurve(pd.DataFrame(Y, index=idx, columns=list(m)))
    res = acm(curve, n_factors=3)
    assert res.fit_rmse_bp < 10
    assert res.term_premium[10.0].notna().all()
