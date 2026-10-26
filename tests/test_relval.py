import numpy as np
import pandas as pd
import pytest

from bondpredictor import relval as rv


def synthetic_markets(n=8, T=360, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1980-01", periods=T, freq="M")
    out = {}
    for i in range(n):
        code = f"M{i}"
        y = 0.05 + np.cumsum(rng.normal(0, 0.002, T))
        short = y - 0.01 + rng.normal(0, 0.003, T)
        infl = 0.02 + np.cumsum(rng.normal(0, 0.001, T))
        out[code] = rv.Market(code, pd.Series(y, idx), pd.Series(short, idx), pd.Series(infl, idx))
    return out


def test_skip_month_convention():
    mk = synthetic_markets()
    t = pd.Period("1995-06", "M")
    base = rv.build_panel(mk, lag=1)
    bumped = {c: rv.Market(c, m.yld.copy(), m.short, m.infl) for c, m in mk.items()}
    bumped["M0"].yld.loc[t] += 0.01  # change the month-t average yield only
    lagged = rv.build_panel(bumped, lag=1)
    # With the lag the position formed at t earns avg(t+1) -> avg(t+2): untouched by month t.
    assert lagged.fwd.loc[t, "M0"] == pytest.approx(base.fwd.loc[t, "M0"])
    naive_base, naive = rv.build_panel(mk, lag=0), rv.build_panel(bumped, lag=0)
    assert naive.fwd.loc[t, "M0"] != pytest.approx(naive_base.fwd.loc[t, "M0"])


def test_signals_and_weights_use_no_future_data():
    mk = synthetic_markets()
    t0 = pd.Period("1995-06", "M")
    scrambled = {}
    rng = np.random.default_rng(9)
    for c, m in mk.items():
        y, s, i = m.yld.copy(), m.short.copy(), m.infl.copy()
        later = y.index > t0
        y[later] = rng.normal(0.05, 0.02, later.sum())
        s[later] = rng.normal(0.03, 0.02, later.sum())
        i[later] = rng.normal(0.02, 0.02, later.sum())
        scrambled[c] = rv.Market(c, y, s, i)
    a, b = rv.build_panel(mk), rv.build_panel(scrambled)
    for sig in rv.SIGNALS:
        wa = rv.weights(a.signals[sig], a.vol).loc[:t0]
        wb = rv.weights(b.signals[sig], b.vol).loc[:t0]
        pd.testing.assert_frame_equal(wa, wb)


def test_weights_are_dollar_and_risk_balanced():
    idx = pd.period_range("2000-01", periods=3, freq="M")
    sig = pd.DataFrame([[1, 2, 3, 4, 5, np.nan], [5, 4, 3, 2, 1, 0], [1, 2, np.nan, np.nan, np.nan, np.nan]],
                       index=idx, columns=list("ABCDEF"), dtype=float)
    vol = pd.DataFrame(0.02, index=idx, columns=sig.columns)
    w = rv.weights(sig, vol, min_n=5, vol_scale=False)
    assert w.iloc[0].clip(lower=0).sum() == pytest.approx(1) and w.iloc[0].clip(upper=0).sum() == pytest.approx(-1)
    assert w.iloc[0]["E"] > 0 > w.iloc[0]["A"] and np.isnan(w.iloc[0]["F"])
    assert w.iloc[2].isna().all()  # fewer than min_n markets: no position
    vol.loc[idx[0], "E"] = 0.04  # a riskier market gets a smaller notional
    w2 = rv.weights(sig, vol, min_n=5, vol_scale=True)
    assert w2.iloc[0]["E"] < w.iloc[0]["E"]


def test_informative_signal_earns_and_placebo_rejects():
    mk = synthetic_markets(n=10, T=400, seed=1)
    panel = rv.build_panel(mk)
    noise = np.random.default_rng(2).normal(0, 1, panel.fwd.shape)
    panel.signals["oracle"] = panel.fwd / panel.fwd.std() + 3 * noise  # deliberately uses the future
    res = rv.run_strategy(panel, "oracle", "1985-01")
    perf = rv.performance(res)
    assert perf["SR"] > 1.0 and perf["SR net 5bp"] < perf["SR"]
    assert rv.placebo_pvalue(panel, "oracle", perf["SR"], "1985-01", n_perm=50) == 0.0
    sr_random = rv._sr(rv.run_strategy(panel, "momentum", "1985-01"))
    assert abs(sr_random) < 0.6  # random-walk yields: no real signal
