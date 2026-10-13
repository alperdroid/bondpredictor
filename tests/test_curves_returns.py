import numpy as np
import pandas as pd
import pytest

from bondpredictor.curves import SvenssonCurve, TabulatedCurve, bootstrap_par_yields, monthly_params, svensson
from bondpredictor.returns import par_bond_duration_convexity, par_bond_excess_returns, zero_excess_returns
from bondpredictor.timeseries import to_monthly


def flat_curve(levels: list[float], start="2000-01") -> SvenssonCurve:
    idx = pd.period_range(start, periods=len(levels), freq="M")
    params = pd.DataFrame(
        {"beta0": np.array(levels) * 100, "beta1": 0.0, "beta2": 0.0, "beta3": 0.0, "tau1": 1.0, "tau2": 1.0}, index=idx
    )
    return SvenssonCurve(params)


def test_svensson_matches_bundesbank_publication():
    # Bundesbank parameters for 2026-09; published zero yields 3.04 / 3.36 / 3.64 / 3.99.
    p = dict(beta0=4.18173, beta1=-1.90569, beta2=18.84971, beta3=-19.00801, tau1=1.36147, tau2=1.50953)
    y = svensson(np.array([1, 5, 10, 30]), **p)
    assert np.allclose(y, [3.04, 3.36, 3.64, 3.99], atol=0.01)


def test_svensson_max_maturity_mask():
    c = flat_curve([0.05, 0.05])
    c2 = SvenssonCurve(c.params, max_maturity=pd.Series([5.0, 30.0], index=c.index))
    y = c2.yields([4.0, 10.0])
    assert np.isnan(y.iloc[0, 1]) and y.iloc[1, 1] == pytest.approx(0.05)


def test_forwards_on_flat_curve_equal_yield():
    f = flat_curve([0.04]).forwards(range(1, 6))
    assert np.allclose(f.to_numpy(), 0.04)


def test_tabulated_interpolation_and_bounds():
    idx = pd.period_range("2000-01", periods=1, freq="M")
    c = TabulatedCurve(pd.DataFrame([[0.01, 0.02, np.nan]], index=idx, columns=[1.0, 2.0, 3.0]))
    y = c.yields([0.5, 1.5, 2.0, 2.5])
    assert np.isnan(y.iloc[0, 0]) and y.iloc[0, 1] == pytest.approx(0.015)
    assert y.iloc[0, 2] == pytest.approx(0.02) and np.isnan(y.iloc[0, 3])


def test_bootstrap_flat_par_curve():
    idx = pd.period_range("2000-01", periods=1, freq="M")
    par = pd.DataFrame([[2.0, 2.0, 2.0, 2.0]], index=idx, columns=[1.0, 2.0, 5.0, 10.0])
    z = bootstrap_par_yields(par)
    # Flat 2% semiannual par curve -> flat zero curve at 2*ln(1.01) continuous.
    assert np.allclose(z.iloc[0].dropna().to_numpy(), 2 * np.log(1.01))


def test_zero_excess_return_algebra():
    # Flat 5% today, flat 6% in 12 months: rx(10) = 10*5% - 9*6% - 5% = -9%.
    levels = [0.05] + [0.06] * 12
    rx = zero_excess_returns(flat_curve(levels), [2, 10], 12)
    assert rx["rx12_n10"].iloc[0] == pytest.approx(-0.09)
    assert rx["rx12_n2"].iloc[0] == pytest.approx(2 * 0.05 - 0.06 - 0.05)
    assert rx["rx12_n10"].iloc[1:].isna().all()  # not yet realised
    # Unchanged flat curve earns exactly the funding rate: zero excess return.
    still = zero_excess_returns(flat_curve([0.03] * 13), [5], 12)
    assert still.iloc[0, 0] == pytest.approx(0.0)


def test_par_bond_duration_matches_numerical():
    y, N = 0.05, 10

    def price(r):
        k = 2
        c = y / k
        t = np.arange(1, k * N + 1)
        return np.sum(c / (1 + r / k) ** t) + 1 / (1 + r / k) ** (k * N)

    h = 1e-5
    D_num = -(price(y + h) - price(y - h)) / (2 * h) / price(y)
    C_num = (price(y + h) - 2 * price(y) + price(y - h)) / h**2 / price(y)
    D, C = par_bond_duration_convexity(y, N)
    assert D == pytest.approx(D_num, rel=1e-6)
    assert C == pytest.approx(C_num, rel=1e-4)
    # Negative yields are fine.
    Dn, _ = par_bond_duration_convexity(-0.002, N)
    assert 10 < Dn < 10.2


def test_par_bond_returns_constant_yield_earn_carry():
    idx = pd.period_range("2000-01", periods=24, freq="M")
    par = pd.Series(0.04, index=idx)
    fund = pd.Series(0.04, index=idx)
    rx1 = par_bond_excess_returns(par, 10, fund, 1)
    assert rx1.iloc[0] == pytest.approx(np.log1p(0.04 / 12) - 0.04 / 12)
    rx12 = par_bond_excess_returns(par, 10, fund, 12)
    assert rx12.iloc[0] == pytest.approx(12 * rx1.iloc[0])


def test_to_monthly_quarterly_lag_and_carry_forward():
    q = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01"]))
    m = to_monthly(q, lag=1)
    # Q1 -> March, +1 month lag -> April; held until the next release is due.
    assert m.index[0] == pd.Period("2020-04", "M")
    assert m.loc["2020-06"] == 1.0 and m.loc["2020-07"] == 2.0 and m.loc["2020-12"] == 3.0
    assert m.index[-1] == pd.Period("2020-12", "M")


def test_monthly_params_takes_whole_last_row():
    d = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [10.0, 20.0, 30.0]},
                     index=pd.to_datetime(["2020-01-02", "2020-01-30", "2020-01-31"]))
    m = monthly_params(d)
    assert np.isnan(m.loc["2020-01", "a"]) and m.loc["2020-01", "b"] == 30.0
