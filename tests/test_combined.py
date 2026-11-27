"""The combined forecast, the pump-price loaders and the storage of pilot runs."""

import io

import numpy as np
import pandas as pd
import pytest

from bondpredictor.data import fuel
from bondpredictor.inflation import combined, pilot, surveys
from bondpredictor.inflation.cli import build_parser


def test_page_links_are_absolute_unique_and_in_page_order():
    page = ('<a href="/media/b/prices_2018.csv">new</a> <a href="https://x.org/media/a/prices_2003.csv">old</a>'
            '<a href="/media/b/prices_2018.csv">again</a> <a href="/media/c/notes.pdf">notes</a>')
    assert fuel.page_links(page, "https://x.org/stats/page", r"\.csv$") == [
        "https://x.org/media/b/prices_2018.csv", "https://x.org/media/a/prices_2003.csv"]


def test_uk_pump_prices_are_read_by_fuel_and_date():
    text = ("﻿Date,ULSP (petrol) Pump price in pence/litre,ULSD (diesel) Pump price in pence/litre,"
            "ULSP (petrol) Duty rate in pence/litre\n02/01/2023,150.5,172.25,52.95\n09/01/2023,149.0,171.0,52.95\n,,,\n")
    petrol, diesel = fuel.parse_uk_fuel(text, "petrol"), fuel.parse_uk_fuel(text, "diesel")
    assert petrol.index.tolist() == [pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-09")]
    assert petrol.tolist() == [150.5, 149.0] and diesel.tolist() == [172.25, 171.0]


def test_weekly_workbooks_are_read_by_header_or_position():
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ec = wb.active
    ec.title = "Prices with taxes"
    ec.append(["Consumer prices", "CTR", "EU_price_with_tax_euro95", "CTR", "EUR_price_with_tax_euro95", "EUR_price_with_tax_diesel"])
    ec.append([None, None, "Euro-super 95", None, "Euro-super 95", "Gas oil"])
    ec.append(["Date", None, "1000 l", None, "1000 l", "1000 l"])
    ec.append([pd.Timestamp("2026-11-23").to_pydatetime(), "EU_", 1700.0, "EUR_", 1850.0, 2050.0])
    ec.append([pd.Timestamp("2026-11-16").to_pydatetime(), "EU_", 1710.0, "EUR_", 1900.0, None])
    jp = wb.create_sheet(fuel.METI_SHEETS["regular"])
    jp.append(["price", "date", "national", "Hokkaido"])
    jp.append([None, pd.Timestamp("1990-08-27").to_pydatetime(), 121, 121])
    jp.append([None, pd.Timestamp("1990-09-03").to_pydatetime(), 122, 120])
    raw = io.BytesIO()
    wb.save(raw)
    euro95 = fuel.parse_ec_oil(raw.getvalue(), "euro95")
    assert euro95.tolist() == [1900.0, 1850.0] and euro95.index.is_monotonic_increasing  # the euro-area column
    assert fuel.parse_ec_oil(raw.getvalue(), "diesel").tolist() == [2050.0]  # blanks are skipped
    assert fuel.parse_meti(raw.getvalue(), "regular").tolist() == [121.0, 122.0]


def test_aggregation_weights_are_shares():
    idx = pd.period_range("2000-01", periods=150, freq="M")
    rng = np.random.default_rng(3)
    comps = pd.DataFrame(2 + np.cumsum(0.2 * rng.normal(size=(150, 3)), axis=0), index=idx, columns=list("abc"))
    head = comps @ np.array([0.5, 0.3, 0.2]) + 0.4 + 0.05 * rng.normal(size=150)  # a level shift tempts the fit off the simplex
    W = pilot.aggregation_weights(head, comps).dropna()
    assert (W.to_numpy() >= 0).all()
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-3)


def _pilot_results():
    idx = pd.period_range("2010-01", periods=6, freq="M")
    def frame(values, start=0):
        return pd.DataFrame({"forecast": values, "rw": 2.0, "actual": 3.0}, index=idx[start:start + len(values)])
    return idx, {
        ("US", 12, "nowcast", "ridge"): frame([1.0] * 6), ("US", 12, "nowcast", "rf"): frame([3.0] * 6),
        ("US", 12, "bottom_up", "ridge"): frame([4.0] * 4, start=2), ("US", 12, "bottom_up", "rf"): frame([6.0] * 4, start=2),
        ("US", 12, "current", "ridge"): frame([9.0] * 6),
        ("US", 12, "comp:energy", "ridge"): frame([50.0] * 6),
    }


def test_combination_averages_the_medians_of_the_parts_that_exist():
    idx, res = _pilot_results()
    out = combined.combined_backtest(res, {"US": ("nowcast", "bottom_up", "nowcast+survey_anchor")})
    r = out[("US", 12, combined.KEY)]
    assert r.index.equals(idx)
    assert r["forecast"].tolist() == [2.0, 2.0, 3.5, 3.5, 3.5, 3.5]  # nowcast alone until the bottom-up starts
    assert (r["rw"] == 2.0).all() and (r["actual"] == 3.0).all()
    assert combined.combined_backtest(res, {"UK": ("nowcast",)}) == {}


def test_headline_is_the_combination_only_where_adopted_and_available(monkeypatch):
    monkeypatch.setattr(combined, "PARTS", {"US": ("nowcast",)})
    assert combined.headline_model("US") == combined.KEY
    assert combined.headline_model("US", ["ens:median", combined.KEY]) == combined.KEY
    assert combined.headline_model("US", ["ens:median"]) == combined.MEDIAN
    assert combined.headline_model("JP", ["ens:median", combined.KEY]) == combined.MEDIAN


def test_pilot_runs_round_trip_through_the_results_store(tmp_path):
    _, res = _pilot_results()
    out, store = tmp_path / "out", tmp_path / "store"
    (out / "pilot").mkdir(parents=True)
    pd.to_pickle(res, out / "pilot" / "pilot.pkl")
    parser = build_parser()
    for action in ("pack", "unpack"):
        args = parser.parse_args(["results", action, "--name", "pilot", "--tag", "pilot_US", "--countries", "US",
                                  "--out", str(out), "--store", str(store)])
        if action == "unpack":
            (out / "pilot" / "pilot.pkl").unlink()
        args.func(args)
    back = pd.read_pickle(out / "pilot" / "pilot.pkl")
    assert set(back) == set(res)
    for k, r in res.items():
        pd.testing.assert_frame_equal(back[k], r, check_freq=False, check_names=False)


def test_survey_comparison_scores_each_economy_on_the_models_it_has(monkeypatch):
    seen = {}
    monkeypatch.setattr(surveys, "us_comparison", lambda frame, res, spf, cpi, models: seen.setdefault("US", models) and pd.DataFrame())
    monkeypatch.setattr(surveys, "ea_comparison", lambda frame, res, ecb, models, lag: seen.setdefault("EA", models) and pd.DataFrame())
    r = pd.DataFrame({"forecast": [1.0]})
    results = {("US", 12, "ens:median"): r, ("US", 12, "ens:combined"): r, ("EA", 12, "ens:median"): r}
    store = {"frames": {"US": (None, None), "EA": (None, None)}}
    surveys.compare(store, results, {"us_spf": None, "us_cpi_sa": None, "ecb": None}, models=("ens:median", "ucsv", "ens:combined"))
    assert seen == {"US": ("ens:median", "ens:combined"), "EA": ("ens:median",)}


def test_tokyo_cpi_file_is_found_in_the_listing_and_read_by_item_code():
    from bondpredictor.data import jstat

    page = ('<tr><td>表番号 1</td><td>中分類指数 前月比（1970年2月～最新月）</td>'
            '<a href="/stat-search/file-download?statInfId=000040482970&amp;fileKind=1">CSV</a></tr>'
            '<tr><td>表番号 1</td><td>中分類指数（1970年1月～最新月）</td><td>調査年月 2026年11月</td>'
            '<a href="/stat-search/file-download?statInfId=000040482965&amp;fileKind=1">CSV</a>'
            '<a href="/stat-search/file-download?statInfId=000040482965&amp;fileKind=0">EXCEL</a></tr>')
    assert jstat.table_id(page) == "000040482965"
    with pytest.raises(jstat.DataUnavailable):
        jstat.table_id("<a href='?statInfId=1&fileKind=1'>CSV</a>")
    text = ("類・品目,総合,食料\nGroup/Item,All items,Food\n類・品目符号(Group/Item code),0001,0002\n"
            "ウエイト(Weight),100,25\n197001,27.7,23.6\n197002,27.9,\n202611,103.8,104.9\n")
    s = jstat.parse_cpi_csv(text, "0001")
    assert s.index.tolist() == [pd.Timestamp("1970-01-01"), pd.Timestamp("1970-02-01"), pd.Timestamp("2026-11-01")]
    assert s.tolist() == [27.7, 27.9, 103.8]
    assert jstat.parse_cpi_csv(text, "0002").tolist() == [23.6, 104.9]  # blanks are skipped
    with pytest.raises(jstat.DataUnavailable):
        jstat.parse_cpi_csv(text, "9999")
