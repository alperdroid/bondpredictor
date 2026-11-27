"""Pre-registered test: arms as specified, the pooled statistic and the forward record."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from bondpredictor.inflation import prereg as pr
from bondpredictor.inflation.backtest import model_zoo
from bondpredictor.inflation.pistar import state_zoo


def test_arms_change_only_the_training_start():
    a, members, rw = pr.arm_specs()
    zoo = {s.name: s for s in model_zoo(1)}
    assert [m.name for m in members] == list(pr.MEMBERS) and len(members) == 13
    for m in members:
        diff = {k for k, v in asdict(m).items() if asdict(zoo[m.name])[k] != v}
        assert diff == {"train_start"} and m.train_start == "1985-01"
    base = {s.name: s for s in state_zoo()}["pistar_fwd_nostate"]
    assert a.theta == 0.0 and a.loss == "fwd" and a.partial and a.train_start == "1985-01"
    assert {k for k, v in asdict(a).items() if asdict(base)[k] != v} == {"train_start"}
    assert rw.model == "rw"


def _arms(n, seed, scale, a_noise, b_noise):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000-01", periods=n, freq="M")
    act = scale * rng.normal(size=n)
    return pd.DataFrame({"A": act + a_noise * scale * rng.normal(size=n), "B": act + b_noise * scale * rng.normal(size=n),
                         "rw": act + scale * rng.normal(size=n), "actual": act}, index=idx)


def test_pooled_test_favours_the_more_accurate_arm_and_weights_economies_equally():
    good = {"X": _arms(200, 0, 1.0, 0.3, 1.0), "Y": _arms(200, 1, 1.0, 0.3, 1.0)}
    t = pr.pooled_test(good, 12)
    assert t["dm"] > 0 and t["p"] < 0.05 and pr.verdict(t) == "A beats B"
    assert t["mean_rel_A"] < t["mean_rel_B"]
    flipped = pr.pooled_test({c: f.rename(columns={"A": "B", "B": "A"}) for c, f in good.items()}, 12)
    assert flipped["dm"] == pytest.approx(-t["dm"]) and pr.verdict(flipped) == "A does not beat B"
    # Scaling one economy's inflation (and so all its errors) leaves its weight, and the statistic, unchanged.
    big = dict(good)
    big["Y"] = good["Y"] * 10.0
    assert pr.pooled_test(big, 12)["dm"] == pytest.approx(t["dm"])


def test_pooled_test_ignores_unpublished_outcomes():
    f = _arms(200, 2, 1.0, 0.5, 1.0)
    g = f.copy()
    g.loc[g.index[-30:], "actual"] = np.nan
    assert pr.pooled_test({"X": g}, 12)["T"] == 170


def test_holm_is_monotone_and_capped():
    adj = pr.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert adj["a"] == pytest.approx(0.04) and adj["c"] == pytest.approx(0.09)
    assert adj["b"] == pytest.approx(0.09) and adj["d"] == pytest.approx(0.5)


def test_forward_record_holds_only_targets_after_the_last_seen_print():
    idx = pd.period_range("2025-01", "2026-11", freq="M")
    fr = lambda v: pd.DataFrame({"forecast": v, "rw": 2.0, "actual": np.nan}, index=idx)
    res = {}
    for h in pr.HORIZONS:
        res[("US", h, "A")] = fr(1.0)
        for m in pr.MEMBERS:
            res[("US", h, m)] = fr(3.0)
    rec = pr.forward_record(res)
    assert (pd.PeriodIndex(rec["target_print"], freq="M") > pr.LAST_SEEN_PRINT).all()
    r12 = rec[rec["horizon"] == 12]
    # origin k targets print k + 11: the first unseen 12-month target (Nov 2026) comes from origin Dec 2025.
    assert r12["origin"].min() == "2025-12" and r12["origin"].max() == "2026-11"
    assert (rec["A"] == 1.0).all() and (rec["B"] == 3.0).all()
