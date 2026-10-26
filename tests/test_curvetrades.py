import numpy as np
import pandas as pd
import pytest

from bondpredictor import curvetrades as ct
from bondpredictor.curves import SvenssonCurve


def svensson_curve(level, slope, curv, start="1970-01"):
    idx = pd.period_range(start, periods=len(level), freq="M")
    params = pd.DataFrame({"beta0": level, "beta1": slope, "beta2": curv, "beta3": 0.0, "tau1": 2.0, "tau2": 1.0},
                          index=idx)
    return SvenssonCurve(params)


def random_curve(T=480, seed=0):
    rng = np.random.default_rng(seed)
    level = 5 + np.cumsum(rng.normal(0, 0.2, T))
    slope = -1.5 + np.cumsum(rng.normal(0, 0.15, T)) * 0.3
    curv = np.cumsum(rng.normal(0, 0.1, T)) * 0.3
    return svensson_curve(level, slope, curv)


def test_trade_weights_are_duration_neutral():
    T = 120
    level = 5 + np.cumsum(np.random.default_rng(1).normal(0, 0.3, T))  # pure parallel moves
    curve = svensson_curve(level, np.zeros(T), np.zeros(T))
    funding = curve.yield_at(1 / 12)
    for trade in (ct.Trade("X", (2, 10)), ct.Trade("X", (2, 5, 10))):
        f = ct.trade_frame(curve, funding, trade)
        dl = pd.Series(level / 100, index=curve.index).diff().shift(-1)
        ten_year = 10 * dl.abs()  # a 10y zero's loss for the same move
        ok = f["ret"].notna() & ten_year.notna()
        assert ok.sum() > 100
        assert (f["ret"][ok].abs() <= 0.05 * ten_year[ok] + 1e-12).all()


def test_signals_use_no_future_data():
    curve = random_curve()
    t0 = pd.Period("1995-06", "M")
    params = curve.params.copy()
    later = params.index > t0
    params.loc[later, ["beta0", "beta1", "beta2"]] = np.random.default_rng(5).normal(0, 3, (later.sum(), 3))
    other = SvenssonCurve(params)
    for trade in (ct.Trade("X", (2, 10)), ct.Trade("X", (2, 5, 10))):
        a = ct.trade_frame(curve, curve.yield_at(1 / 12), trade)
        b = ct.trade_frame(other, other.yield_at(1 / 12), trade)
        cols = [c for c in a.columns if c.startswith("pos_")] + ["vol", "level", "carry_ann"]
        pd.testing.assert_frame_equal(a.loc[:t0, cols], b.loc[:t0, cols])


def test_delay_shifts_the_return_window():
    idx = pd.period_range("2000-01", periods=60, freq="M")
    f = pd.DataFrame({"ret": np.arange(60, dtype=float), "vol": 1.0, "short": 0.03,
                      **{f"pos_{s}": 1.0 for s in ct.SIGNALS}}, index=idx)
    now, later = ct.portfolio({"x": f}, "combo"), ct.portfolio({"x": f}, "combo", lag=1)
    assert now.loc[idx[5], "ret"] == 5.0 and later.loc[idx[5], "ret"] == 6.0
    assert np.isnan(later["ret"].reindex(idx).iloc[-1])


def test_placebo_detects_real_timing_but_not_static_exposure():
    rng = np.random.default_rng(3)
    idx = pd.period_range("1980-01", periods=480, freq="M")
    signal = pd.Series(np.sign(rng.normal(size=480)), index=idx).rolling(6, min_periods=1).mean()
    timed = pd.DataFrame({"ret": 0.8 * signal + rng.normal(0, 1, 480), "vol": 1.0, "short": 0.03,
                          **{f"pos_{s}": signal for s in ct.SIGNALS}}, index=idx)
    sr = ct._sr(ct.portfolio({"x": timed}, "combo"))
    assert ct.shift_placebo({"x": timed}, "combo", sr, "1980-01", "2019-12", n_perm=100) < 0.05
    static = timed.copy()
    static["ret"] = 0.2 + rng.normal(0, 1, 480)  # positive average return, no timing
    for s in ct.SIGNALS:
        static[f"pos_{s}"] = 1.0
    sr2 = ct._sr(ct.portfolio({"x": static}, "combo"))
    assert sr2 > 0.4 and ct.shift_placebo({"x": static}, "combo", sr2, "1980-01", "2019-12", n_perm=100) > 0.3
