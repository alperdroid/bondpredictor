"""Pilot extensions: nowcast, survey path and bottom-up aggregation use only what was known at the origin."""

import numpy as np
import pandas as pd
import pytest

from bondpredictor.inflation import pilot
from bondpredictor.inflation.features import build_frame
from test_inflation import _synthetic


def _frame():
    return build_frame(_synthetic(), "US")


def test_nowcast_ignores_data_after_the_origin():
    frame, _ = _frame()
    rng = np.random.default_rng(0)
    gas = pd.Series(np.cumsum(rng.normal(size=len(frame))), index=frame.index)
    k = 300
    base = pilot.nowcast(frame, gas, min_train=60)
    later = frame.copy()
    cols = ["pi", "pi_d1", "oil_yoy"]
    later.iloc[k + 1:, [later.columns.get_loc(c) for c in cols]] = 99.0
    gas2 = gas.copy()
    gas2.iloc[k + 1:] = -50.0
    again = pilot.nowcast(later, gas2, min_train=60)
    pd.testing.assert_series_equal(base.iloc[: k + 1], again.iloc[: k + 1])
    assert base.iloc[k:-1].notna().all()


def test_nowcast_recovers_a_gasoline_pass_through():
    frame, _ = _frame()
    rng = np.random.default_rng(1)
    gas = pd.Series(rng.normal(size=len(frame)), index=frame.index).cumsum()
    f = frame.copy()
    # inflation's next change is 0.3 x this month's change in gasoline's y/y rate, plus a little noise
    step = 0.3 * gas.diff().fillna(0.0) + 0.01 * rng.normal(size=len(f))
    f["pi"] = step.shift(1).fillna(0.0).cumsum()
    f["pi_d1"] = f["pi"].diff()
    now = pilot.nowcast(f, gas, min_train=60)
    actual = f["pi"].shift(-1) - f["pi"]
    err = (actual - now).iloc[200:].dropna()
    assert err.std() < 0.05 < actual.iloc[200:].std()


def test_survey_gap_starts_when_the_survey_is_published(monkeypatch):
    frame, _ = _frame()
    a = frame.index[200]
    monkeypatch.setitem(pilot.SURVEY_POINTS, "US", lambda dl: [(a, [(a + 11, 5.0), (a + 23, 3.0)])])
    g = pilot.survey_gap(None, "US", frame, (3, 12, 24))
    assert g.loc[: a - 1].isna().all().all() and g.loc[a: a + 23].notna().all().all()
    assert g.loc[a + 24:].isna().all().all()  # every survey point lies behind the latest print: no path
    pi = frame["pi"]
    # at the publication month: the 12-month target (a + 11) is the first survey point
    assert g.loc[a, "svy_gap_h12"] == pytest.approx(5.0 - pi[a])
    assert g.loc[a, "svy_gap_h24"] == pytest.approx(3.0 - pi[a])
    # 3 months: on the line from the latest print (month a - 1) to the first point (month a + 11)
    assert g.loc[a, "svy_gap_h3"] == pytest.approx((5.0 - pi[a]) * 3 / 12)
    # beyond the survey's last point the path is flat at it
    assert g.loc[a + 20, "svy_gap_h12"] == pytest.approx(3.0 - pi[a + 20])


def test_bottom_up_adds_component_changes_with_real_time_weights():
    idx = pd.period_range("2000-01", periods=200, freq="M")
    rng = np.random.default_rng(2)
    A = pd.Series(2 + np.cumsum(0.1 * rng.normal(size=200)), index=idx)
    B = pd.Series(1 + np.cumsum(0.1 * rng.normal(size=200)), index=idx)
    head = 0.3 * A + 0.7 * B
    W = pilot.aggregation_weights(head, pd.DataFrame({"a": A, "b": B}))
    assert W.iloc[100].tolist() == pytest.approx([0.3, 0.7], abs=1e-6)
    assert W.iloc[:20].isna().all().all()  # needs half a window of published data
    h = 12
    frame = pd.DataFrame({"pi": head, f"tgt{h}": head.shift(-h)})
    comp_frames = {"a": (pd.DataFrame({"pi": A}), {}), "b": (pd.DataFrame({"pi": B}), {})}
    origins = idx[60:150]
    perfect = {n: pd.DataFrame({"forecast": s.shift(-h).loc[origins]}) for n, s in (("a", A), ("b", B))}
    out = pilot.bottom_up(frame, h, comp_frames, perfect)
    assert np.allclose(out["forecast"], out["actual"], atol=1e-6)
    assert np.allclose(out["rw"], head.loc[out.index])
