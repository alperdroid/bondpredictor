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
