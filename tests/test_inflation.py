"""Inflation module: real-time features, no look-ahead in the backtest, models and evaluation."""

import numpy as np
import pandas as pd
import pytest

from bondpredictor.inflation import config as cfg
from bondpredictor.inflation.backtest import Spec, run_backtest
from bondpredictor.inflation.data import (CountryData, InflationData, contiguous, fill_price_gaps, global_factor,
                                          splice)
from bondpredictor.inflation.evaluation import (conformal_intervals, interval_scores, model_confidence_set,
                                                 subperiod_table)
from bondpredictor.inflation.features import build_frame, sa_onesided, tax_feature


def _synthetic(n_months=420, seed=0, start="1985-01"):
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, periods=n_months, freq="M")
    season = np.tile([0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.2, 0.0, 0.1, 0.05, 0.05], n_months // 12 + 1)[:n_months]
    trend = 0.2 + 0.1 * np.sin(np.arange(n_months) / 40)
    m = trend + season + 0.15 * rng.normal(size=n_months)
    cpi = pd.Series(100 * np.exp(np.cumsum(m) / 100), index=idx)
    core = pd.Series(100 * np.exp(np.cumsum(trend + 0.05 * rng.normal(size=n_months)) / 100), index=idx)
    unemp = pd.Series(6 + np.cumsum(0.05 * rng.normal(size=n_months)), index=idx)
    ppi = pd.Series(100 * np.exp(np.cumsum(m + 0.3 * rng.normal(size=n_months)) / 100), index=idx)
    oil = pd.Series(50 * np.exp(np.cumsum(0.05 * rng.normal(size=n_months))), index=idx)
    cd = CountryData("US", cpi, core, {}, oil, {"unemp": unemp, "ppi": ppi})
    panel = pd.DataFrame({a: cpi * (1 + 0.01 * i) for i, a in enumerate(cfg.GLOBAL_FACTOR_AREAS)})
    data = InflationData({"US": cd}, panel, global_factor(panel), idx[-1])
    return data


def test_sa_onesided_uses_only_past_data():
    rng = np.random.default_rng(1)
    idx = pd.period_range("1990-01", periods=200, freq="M")
    m = pd.Series(np.tile(np.arange(12) / 10, 17)[:200] + 0.1 * rng.normal(size=200), index=idx)
    full = sa_onesided(m)
    k = 150
    scrambled = m.copy()
    scrambled.iloc[k + 1 :] = rng.normal(size=200 - k - 1) * 5
    assert np.allclose(full.iloc[: k + 1].dropna(), sa_onesided(scrambled).iloc[: k + 1].dropna())
    # and the seasonal pattern is removed: SA changes no longer depend on the calendar month
    by_month = full.dropna().groupby(full.dropna().index.month).mean()
    assert by_month.std() < 0.05 < m.groupby(m.index.month).mean().std()


def test_features_are_real_time():
    data = _synthetic()
    full, groups = build_frame(data, "US")
    k = pd.Period("2010-06", "M")
    cut = InflationData(
        {"US": CountryData("US", *[s.loc[:k] if s is not None else None for s in
                                   (data.countries["US"].cpi, data.countries["US"].core)], {},
                           data.countries["US"].oil.loc[:k],
                           {n: s.loc[:k] for n, s in data.countries["US"].predictors.items()})},
        data.oecd_panel.loc[: k - 2], global_factor(data.oecd_panel.loc[: k - 2]), k)
    part, _ = build_frame(cut, "US")
    feats = [c for c in groups] + ["ucsv_tau"]
    a, b = full.loc[:k, feats], part.loc[:k, feats]
    pd.testing.assert_frame_equal(a, b, check_freq=False)


def test_backtest_forecast_ignores_future_data():
    data = _synthetic()
    frame, groups = build_frame(data, "US")
    k, h = pd.Period("2012-03", "M"), 12
    spec = Spec("ridge", "ridge", refit_every=1, train_start="1985-01")
    base = run_backtest(frame, groups, h, spec, start=str(k), end=str(k), min_train=60, min_hist=24)
    rng = np.random.default_rng(5)
    noisy = frame.copy()
    after = noisy.index > k
    feat_cols = [c for c in noisy.columns if not c.startswith("tgt")]
    noisy.loc[after, feat_cols] = rng.normal(size=(after.sum(), len(feat_cols)))
    unrealised = noisy.index > k - h
    noisy.loc[unrealised, f"tgt{h}"] = rng.normal(size=unrealised.sum()) * 10
    again = run_backtest(noisy, groups, h, spec, start=str(k), end=str(k), min_train=60, min_hist=24)
    assert base["forecast"].iloc[0] == pytest.approx(again["forecast"].iloc[0])
    assert base["rw"].iloc[0] == frame.loc[k, "pi"]


def test_random_walk_and_targets():
    data = _synthetic()
    frame, groups = build_frame(data, "US")
    r = run_backtest(frame, groups, 3, Spec("rw", "rw"), start="2000-01")
    assert (r["forecast"] == r["rw"]).all()
    d = pd.Period("2001-05", "M")
    assert r.loc[d, "actual"] == pytest.approx(frame.loc[d + 3, "pi"])


def test_tax_feature_timing_and_announcement():
    idx = pd.period_range("2012-01", "2015-12", freq="M")
    ev = [cfg.TaxEvent("2014-04", 3.0, "2012-08")]
    t12 = tax_feature(idx, ev, 12)
    # Origin 2013-06 (latest print May 2013): the y/y rate 12 months on (May 2014) includes the hike.
    assert t12[pd.Period("2013-06", "M")] == 3.0
    # Before the announcement the hike is unknown.
    assert t12[pd.Period("2012-07", "M")] == 0.0
    # Origin 2014-06: the current rate includes it and the one a year on does not.
    assert t12[pd.Period("2014-06", "M")] == -3.0
    t3 = tax_feature(idx, ev, 3)
    assert t3[pd.Period("2014-02", "M")] == 3.0  # latest print Jan 2014, target Apr 2014
    assert t3[pd.Period("2013-12", "M")] == 0.0


def test_splice_ratio_and_level():
    a = pd.Series([10.0, 11.0], index=pd.period_range("2000-03", periods=2, freq="M"))
    b = pd.Series([1.0, 2.0, 2.2], index=pd.period_range("2000-01", periods=3, freq="M"))
    s = splice([a, b], "ratio")
    # rescaled at the first common month (March: 10 vs 2.2)
    assert s.iloc[0] == pytest.approx(10 / 2.2) and s.iloc[2] == 10.0
    s = splice([a, b], "level")
    assert s.iloc[0] == 1.0 and s.iloc[3] == 11.0


def test_mcs_keeps_best_and_drops_bad():
    rng = np.random.default_rng(0)
    T = 300
    e = rng.normal(size=T)
    e2 = rng.normal(size=T)  # an independent forecast that is just as good
    L = pd.DataFrame({"good": e**2, "also_good": e2**2, "bad": (e + 2.0 + rng.normal(size=T)) ** 2})
    p = model_confidence_set(L, alpha=0.25, n_boot=500, block=6)
    assert p["good"] >= 0.25 and p["also_good"] >= 0.25 and p["bad"] < 0.05


def test_conformal_uses_only_realised_errors():
    idx = pd.period_range("2000-01", periods=120, freq="M")
    r = pd.DataFrame({"forecast": 2.0, "actual": 2.0 + np.r_[np.ones(60), 100 * np.ones(60)], "rw": 2.0}, index=idx)
    ci = conformal_intervals(r, h=12, min_errors=36)
    # Origin 70 can only use errors from origins <= 58, which are all +1.
    assert ci["hi"].iloc[70] == pytest.approx(3.0)
    assert np.isnan(ci["lo"].iloc[40])  # not enough realised errors yet
    s = interval_scores([1.0, 5.0], [0.0, 0.0], [2.0, 2.0], alpha=0.2)
    assert s["coverage"] == 0.5 and s["interval_score"] == pytest.approx(2 + 0.5 * 10 * 3)


def test_subperiods_by_target_month():
    idx = pd.period_range("2019-01", "2023-12", freq="M")
    r = pd.DataFrame({"forecast": 1.0, "rw": 0.0, "actual": 1.0}, index=idx)
    t = subperiod_table({"m": r}, h=12, min_n=6)
    assert t.loc["m", "2021-22 surge"] == 0.0


@pytest.mark.parametrize("name", ["mrf", "bart", "hnn", "adalasso", "3prf", "csr", "di"])
def test_models_fit_and_predict(name):
    from bondpredictor.inflation.backtest import make_model

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 6))
    X[:20, 5] = np.nan
    y = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.normal(size=n)
    names = ["mech", "pi", "unemp_gap", "oil_yoy", "a", "b"]
    groups = ["own", "own", "slack", "cost", "money", "global"]
    kw = {"bart": {"n_burn": 50, "n_draws": 50}, "hnn": {"n_bags": 2, "max_epochs": 50},
          "mrf": {"n_trees": 10}}.get(name, {})
    m = make_model(name, 1, **kw).fit(X[:150], y[:150], names, groups)
    pred = m.predict(X[150:])
    assert pred.shape == (50,) and np.all(np.isfinite(pred))
    assert np.corrcoef(pred, y[150:])[0, 1] > 0.5


def test_pooled_engine_no_lookahead_and_min_train():
    from bondpredictor.inflation.panel import run_panel_backtest

    a, _ = build_frame(_synthetic(seed=0), "US")
    late = _synthetic(seed=1, n_months=300, start="1995-01")  # shorter history
    late.countries["UK"] = late.countries.pop("US")
    late.countries["UK"].code = "UK"
    b, gb = build_frame(late, "UK")
    frames = {"US": (a, build_frame(_synthetic(seed=0), "US")[1]), "UK": (b, gb)}
    spec = Spec("panel:ridge", "ridge", refit_every=12, train_start="1985-01")
    k, h = pd.Period("2012-03", "M"), 12
    res = run_panel_backtest(frames, h, spec, start="2004-01", end=str(k), min_train=120)
    # The late economy needs 120 realised targets: its CPI y/y starts 1996-01, so not before ~2007.
    assert res["UK"].index.min() >= pd.Period("2006-12", "M") and res["US"].index.min() == pd.Period("2004-01", "M")
    rng = np.random.default_rng(3)
    noisy = {}
    for c, (f, g) in frames.items():
        f = f.copy()
        cols = [col for col in f.columns if not col.startswith("tgt")]
        f.loc[f.index > k, cols] = rng.normal(size=((f.index > k).sum(), len(cols)))
        f.loc[f.index > k - h, f"tgt{h}"] = rng.normal(size=(f.index > k - h).sum())
        noisy[c] = (f, g)
    again = run_panel_backtest(noisy, h, spec, start="2004-01", end=str(k), min_train=120)
    for c in res:
        assert again[c].loc[k, "forecast"] == pytest.approx(res[c].loc[k, "forecast"])


def test_splice_keeps_missing_months_on_the_calendar():
    idx = pd.period_range("2024-01", "2025-06", freq="M")
    s = pd.Series(np.arange(len(idx), dtype=float) + 100, index=idx).drop(pd.Period("2024-10", "M"))
    out = splice([s], "ratio")
    assert len(out) == len(idx) and np.isnan(out[pd.Period("2024-10", "M")])
    # A 12-month change after the gap compares the right months.
    assert out.diff(12)[pd.Period("2025-06", "M")] == 12.0
    assert contiguous(s).index.equals(idx)


def test_fill_price_gaps_interpolates_short_internal_gaps_only():
    idx = pd.period_range("2025-01", "2025-12", freq="M")
    s = pd.Series(100 * 1.01 ** np.arange(12), index=idx)
    s.iloc[[0, 5, 8, 9, 10]] = np.nan  # leading gap, a 1-month gap, a 3-month gap
    out, filled = fill_price_gaps(s, max_gap=2)
    assert filled == [pd.Period("2025-06", "M")]
    assert out[pd.Period("2025-06", "M")] == pytest.approx(100 * 1.01**5)
    assert np.isnan(out.iloc[0]) and out.iloc[8:11].isna().all()


def test_cpi_gap_is_an_input_but_never_a_target():
    data = _synthetic()
    cd = data.countries["US"]
    gap = pd.Period("2010-10", "M")
    true_pi = 100 * np.log(cd.cpi / cd.cpi.shift(12))
    cd.cpi, filled = fill_price_gaps(cd.cpi.drop(gap))
    cd.filled = {"CPI": filled}
    frame, _ = build_frame(data, "US")
    after = pd.Period("2011-03", "M")
    assert frame.loc[after, "pi"] == pytest.approx(true_pi[after])  # no positional shift after the gap
    assert np.isfinite(frame.loc[gap, "pi"])  # filled, usable as an input
    for p in (gap, gap + 12):
        assert np.isnan(frame.loc[p - 3, "tgt3"]) and np.isnan(frame.loc[p - 12, "tgt12"])
