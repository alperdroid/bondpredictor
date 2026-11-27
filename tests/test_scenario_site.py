"""Scenario responses and the site renderer."""

import math

import numpy as np
import pandas as pd
import pytest

from bondpredictor.inflation.backtest import Spec, run_backtest
from bondpredictor.inflation.features import build_frame
from bondpredictor.inflation.scenario import LEVER_BY_KEY, PRESETS, _shifted_rows, model_responses
from bondpredictor.inflation.site import render
from test_inflation import _synthetic


def test_price_levers_shift_in_log_points():
    shift = LEVER_BY_KEY["oil"].shift("US", 100)
    assert set(shift) == {"oil_yoy", "oil_c3", "oil_c1"}
    assert all(v == pytest.approx(100 * math.log(2)) for v in shift.values())
    # Japan's currency lever: a stronger yen lowers USD/JPY
    fx = LEVER_BY_KEY["currency"].shift("JP", 10)
    assert fx["reer_yoy"] > 0 and fx["reer_yoy"] == pytest.approx(-fx["usdjpy_yoy"])
    assert LEVER_BY_KEY["wages"].shift("EA", 2) == {}


def test_shifted_rows_touch_only_known_columns():
    base = np.array([1.0, 2.0, 3.0])
    rows = _shifted_rows(base, ["oil_yoy", "pi", "unemp_lvl"], "US", [{}, {"oil": 50}, {"unemployment": 1, "abroad": 2}])
    assert np.array_equal(rows[0], base)
    assert rows[1, 0] == pytest.approx(1 + 100 * math.log(1.5)) and rows[1, 1:].tolist() == [2.0, 3.0]
    assert rows[2].tolist() == [1.0, 2.0, 4.0]
    assert base.tolist() == [1.0, 2.0, 3.0]


def test_responses_match_the_latest_forecast_and_are_linear_for_ridge():
    frame, groups = build_frame(_synthetic(), "US")
    spec = Spec("ridge", "ridge", refit_every=1)
    r = model_responses(frame, groups, "US", 12, spec)
    origin = r["origin"]
    bt = run_backtest(frame, groups, 12, spec, start=origin, end=origin)
    assert r["base"] == pytest.approx(bt["forecast"].iloc[-1])
    oil = dict(zip(LEVER_BY_KEY["oil"].grid, r["curves"]["oil"]))
    assert oil[0] == 0.0
    assert oil[100] / oil[-50] == pytest.approx(math.log(2) / math.log(0.5))
    for p in PRESETS:  # a linear model adds up exactly
        assert r["joint"][p.key] == pytest.approx(r["additive"][p.key], abs=1e-9)
    assert r["active"]["oil"] == ["oil_c1", "oil_c3", "oil_yoy"] and r["active"]["currency"] == []


def test_render_embeds_data_safely():
    html = render("<script>const DATA = /*__DATA__*/null;</script>", {"note": "</script><b>"})
    assert "</script><b>" not in html and '"note":"<\\/script><b>"' in html
    with pytest.raises(ValueError, match="placeholder"):
        render("<p>no data here</p>", {})


def test_previous_data_round_trips_through_the_page():
    from bondpredictor.inflation.site import previous_data

    data = {"asof": "2026-10-18", "paths": {"US": {"12": {"ens:median": {"start": "2001-01", "values": [1.5, None]}}}},
            "note": "</script>"}
    page = render("<p>x</p><script>const DATA = /*__DATA__*/null;\nboot();</script>", data)
    assert previous_data(page) == data


def test_replace_latest_keeps_the_backtest_sections():
    from bondpredictor.inflation.report import replace_latest

    old = "# Report\n\n## 12 months ahead\n\ntable\n\n## Latest forecasts\n\nold numbers\n"
    new = replace_latest(old, ["## Latest forecasts", "", "new numbers", ""])
    assert new == "# Report\n\n## 12 months ahead\n\ntable\n\n## Latest forecasts\n\nnew numbers\n"
    with pytest.raises(ValueError):
        replace_latest("# Report\n", ["x"])


def test_recursive_ols_matches_brute_force_and_ignores_later_rows():
    from bondpredictor.inflation.transform_search import recursive_ols

    rng = np.random.default_rng(3)
    n, h = 300, 12
    Z = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    Z[rng.random(n) < 0.1, 2] = np.nan
    y = Z[:, 1] * 0.5 - np.nan_to_num(Z[:, 2]) + rng.normal(size=n)
    y[rng.random(n) < 0.05] = np.nan
    origins = np.arange(150, n)
    got = recursive_ols(Z, y, origins, h, t0=10, min_train=60, ridge=0.0)
    for i, k in enumerate(origins[::17]):
        rows = np.arange(10, k - h + 1)
        rows = rows[np.isfinite(Z[rows]).all(axis=1) & np.isfinite(y[rows])]
        if not np.isfinite(Z[k]).all():
            continue
        beta, *_ = np.linalg.lstsq(Z[rows], y[rows], rcond=None)
        assert got[np.searchsorted(origins, k)] == pytest.approx(Z[k] @ beta)
    later = y.copy()
    later[200 - h + 1:] = 99.0  # targets not yet published at origin 200
    assert recursive_ols(Z, later, np.array([200]), h, t0=10)[0] == pytest.approx(recursive_ols(Z, y, np.array([200]), h, t0=10)[0])


def test_candidate_transforms_use_only_the_past():
    from bondpredictor.inflation.transform_search import base_transforms

    rng = np.random.default_rng(4)
    idx = pd.period_range("1990-01", periods=240, freq="M")
    price = pd.Series(100 * np.exp(np.cumsum(0.002 + 0.01 * rng.normal(size=240))), index=idx)
    rate = pd.Series(np.cumsum(rng.normal(size=240)), index=idx)
    k = 180
    for s, kind in ((price, "price"), (rate, "level")):
        full = base_transforms(s, kind)
        cut = base_transforms(s.iloc[: k + 1], kind)
        for name, x in full.items():
            pd.testing.assert_series_equal(x.iloc[: k + 1], cut[name], check_names=False)


def test_convergence_weights_use_only_published_targets():
    from bondpredictor.inflation.transform_search import convergence_weights

    frame, _ = build_frame(_synthetic(), "US")
    h, k = 12, 300
    w = convergence_weights(frame, h, min_rows=60)
    later = frame.copy()
    later.iloc[k - h + 1:, later.columns.get_loc(f"tgt{h}")] = 50.0  # targets not yet published at origin k
    assert convergence_weights(later, h, min_rows=60).iloc[k] == pytest.approx(w.iloc[k])
    assert w.dropna().between(0, 1).all()


def test_partial_anchor_offset_reaches_the_backtest():
    from dataclasses import replace

    from bondpredictor.inflation.transform_search import feature_sets

    frame, groups = build_frame(_synthetic(), "US")
    sets = feature_sets(frame, groups, {}, pd.DataFrame(columns=["name", "var", "select_ratio"]), 12)
    f, g, off = sets["partial_anchor"]
    assert off == "_anchor12" and off not in g
    spec = Spec("ridge", "ridge", refit_every=1, train_start="1985-01")
    base = run_backtest(frame, groups, 12, spec, start="2010-01", end="2010-06", min_train=60, min_hist=24)
    anch = run_backtest(f, g, 12, replace(spec, offset=off), start="2010-01", end="2010-06", min_train=60, min_hist=24)
    assert len(anch) == len(base) and not np.allclose(anch["forecast"], base["forecast"])
    # the anchor must not shrink the training sample: the offset exists wherever the target and core do
    usable = frame[f"tgt12"].notna() & frame["core"].notna() & (frame.index >= pd.Period("1985-01", "M"))
    assert f.loc[usable, off].notna().all()


def test_frame_carries_a_real_time_24_month_anchor_that_is_not_a_feature():
    data = _synthetic()
    frame, groups = build_frame(data, "US")
    assert "anchor_h24" in frame and "anchor_h24" not in groups
    k = 300
    later = frame.copy()
    later.iloc[k - 24 + 1:, later.columns.get_loc("tgt24")] = 40.0  # targets not yet published at origin k
    from bondpredictor.inflation.features import anchor_column

    assert anchor_column(later, 24).iloc[k] == pytest.approx(frame["anchor_h24"].iloc[k])


def test_ml_models_are_anchored_at_24_months_only():
    from dataclasses import replace

    from bondpredictor.inflation.backtest import model_zoo

    specs = {s.name: s for s in model_zoo(1)}
    assert specs["ridge"].anchor_horizons == (24,) and specs["ucsv"].anchor_horizons == ()
    frame, groups = build_frame(_synthetic(), "US")
    ridge = replace(specs["ridge"], train_start="1985-01")
    plain = replace(ridge, anchor_horizons=())
    kw = dict(start="2012-01", end="2012-03", min_train=60, min_hist=24)
    a12, p12 = run_backtest(frame, groups, 12, ridge, **kw), run_backtest(frame, groups, 12, plain, **kw)
    a24, p24 = run_backtest(frame, groups, 24, ridge, **kw), run_backtest(frame, groups, 24, plain, **kw)
    assert np.allclose(a12["forecast"], p12["forecast"])
    assert not np.allclose(a24["forecast"], p24["forecast"])


def test_results_round_trip_through_csv(tmp_path):
    from bondpredictor.inflation.evaluation import pack_results, unpack_results

    idx = pd.period_range("2020-01", periods=4, freq="M", name="date")
    a = pd.DataFrame({"forecast": [1.0, 2.0, 3.0, 4.0], "rw": [1.5] * 4, "actual": [1.1, np.nan, np.nan, np.nan],
                      "refit": [True, False, False, True]}, index=idx)
    b = a.assign(q10=0.5).drop(columns="refit")
    path = tmp_path / "r.csv.gz"
    assert pack_results({("US", 12, "ridge"): a, ("UK", 3, "panel:bart"): b, ("JP", 3, "x"): pd.DataFrame()}, path) == 2
    back = unpack_results([path])
    assert set(back) == {("US", 12, "ridge"), ("UK", 3, "panel:bart")}
    pd.testing.assert_frame_equal(back[("US", 12, "ridge")], a, check_freq=False)
    assert "q10" in back[("UK", 3, "panel:bart")] and "q10" not in back[("US", 12, "ridge")]


def test_pooled_models_use_the_24_month_anchor():
    from dataclasses import replace

    from bondpredictor.inflation.panel import panel_zoo, run_panel_backtest

    a, b = build_frame(_synthetic(seed=0), "US"), build_frame(_synthetic(seed=1), "US")
    frames = {"US": a, "UK": b}
    spec = replace(next(s for s in panel_zoo() if s.name == "panel:ridge"), train_start="1985-01")
    assert spec.anchor_horizons == (24,)
    kw = dict(start="2012-01", end="2012-03", min_train=60, min_hist=24)
    with_a = run_panel_backtest(frames, 24, spec, **kw)["US"]["forecast"]
    without = run_panel_backtest(frames, 24, replace(spec, anchor_horizons=()), **kw)["US"]["forecast"]
    same12 = run_panel_backtest(frames, 12, spec, **kw)["US"]["forecast"]
    assert not np.allclose(with_a, without)
    assert np.allclose(same12, run_panel_backtest(frames, 12, replace(spec, anchor_horizons=()), **kw)["US"]["forecast"])


def test_us_survey_path_chains_quarterly_growth_onto_published_levels():
    from bondpredictor.inflation.surveys import us_spf_yoy

    months = pd.date_range("2018-01-01", "2020-12-01", freq="MS")
    cpi = pd.Series(100.0 * 1.002 ** np.arange(len(months)), index=months)  # 0.2% a month
    spf = pd.DataFrame({"CPI2": [4.0], "CPI3": [4.0], "CPI4": [4.0], "CPI5": [4.0], "CPI6": [4.0]},
                       index=pd.PeriodIndex(["2020Q1"], freq="Q", name="survey"))
    out = us_spf_yoy(spf, cpi).set_index("j")["forecast"]
    q = cpi.resample("QS").mean()
    # one forecast quarter at 4% annualised on top of three published quarters
    expect0 = 100 * (q["2019-10-01"] * 1.04 ** 0.25 / q["2019-01-01"] - 1)
    assert out[0] == pytest.approx(expect0)
    assert out[4] == pytest.approx(4.0)  # four forecast quarters over a forecast base: 4% a year


def test_anchor_uses_a_prior_before_its_weight_can_be_estimated_and_zero_without_core():
    from bondpredictor.inflation.features import anchor_column, convergence_weights

    frame, _ = build_frame(_synthetic(), "US")
    w = convergence_weights(frame, 24)
    first = w.first_valid_index()
    early = frame.index[frame.index.get_loc(first) - 30]
    gap = frame["core"] - frame["pi"]
    a = anchor_column(frame, 24)
    assert np.isfinite(gap[early]) and a[early] == pytest.approx(gap[early])  # prior: full convergence
    assert a[first] == pytest.approx(w[first] * gap[first])
    no_core = frame.copy()
    no_core.loc[: frame.index[150], "core"] = np.nan
    b = anchor_column(no_core, 24)
    rows = no_core.index[:150][no_core["pi"].iloc[:150].notna()]
    assert (b[rows] == 0.0).all()
