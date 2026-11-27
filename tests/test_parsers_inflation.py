"""Parsers for the OECD, ECB, Eurostat and BoJ formats (fixtures are trimmed real responses)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from bondpredictor.data.boj import parse_boj_csv
from bondpredictor.data.eurostat import parse_jsonstat
from bondpredictor.data.sdmx import parse_period, parse_sdmx_csv, parse_sdmx_panel

FIX = Path(__file__).parent / "fixtures"


def test_sdmx_periods():
    idx = parse_period(["2026", "2026-Q3", "2026-08", "2026-08-14"])
    assert list(idx) == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-07-01"),
                         pd.Timestamp("2026-08-01"), pd.Timestamp("2026-08-14")]


def test_oecd_csv_sorted_single_series():
    s = parse_sdmx_csv((FIX / "oecd_cpi_gbr.csv").read_text())
    assert list(s.index) == [pd.Timestamp("1955-05-01"), pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
    assert s.iloc[-1] == pytest.approx(108.6)


def test_oecd_csv_rejects_several_series():
    text = (FIX / "oecd_cpi_gbr.csv").read_text().replace(",GBR,M,N,CPI,IX,_T,N,_Z,2020-01", ",JPN,M,N,CPI,IX,_T,N,_Z,2020-01")
    with pytest.raises(ValueError, match="2 series"):
        parse_sdmx_csv(text)
    wide = parse_sdmx_panel(text)
    assert set(wide.columns) == {"GBR", "JPN"} and wide.loc["2020-01-01", "JPN"] == pytest.approx(108.3)


def test_ecb_csv():
    s = parse_sdmx_csv((FIX / "ecb_m3.csv").read_text())
    assert s.index[0] == pd.Timestamp("1980-01-01") and s.iloc[0] == 1097404


def test_eurostat_jsonstat_skips_missing():
    s = parse_jsonstat(json.loads((FIX / "eurostat_bsco.json").read_text()))
    assert list(s) == [33.7, 36.9, 39.0]  # the fourth month has no value
    assert s.index[0] == pd.Timestamp("1985-01-01")


def test_eurostat_requires_single_series():
    payload = json.loads((FIX / "eurostat_bsco.json").read_text())
    payload["size"][1] = 2
    with pytest.raises(ValueError, match="several values"):
        parse_jsonstat(payload)


def test_boj_monthly_and_quarterly():
    m = parse_boj_csv((FIX / "boj_import_prices.csv").read_text())
    assert list(m.index) == [pd.Timestamp("1960-01-01"), pd.Timestamp("1960-02-01"), pd.Timestamp("2026-09-01")]
    q = parse_boj_csv((FIX / "boj_tankan.csv").read_text())
    # "197402" is Q2 1974; the null Q1 value is dropped.
    assert list(q.index) == [pd.Timestamp("1974-04-01"), pd.Timestamp("1974-07-01"), pd.Timestamp("2026-07-01")]
    assert list(q) == [32, 9, 38]


def test_boj_error_status():
    with pytest.raises(ValueError, match="status 400"):
        parse_boj_csv("STATUS,400\nMESSAGEID,M181005E\nMESSAGE,invalid date\n")


def test_ons_csv_keeps_the_monthly_rows():
    from bondpredictor.data.ons import parse_ons_csv

    s = parse_ons_csv((FIX / "ons_ppi.csv").read_text())
    assert list(s.index) == [pd.Timestamp("1957-01-01"), pd.Timestamp("1957-02-01"), pd.Timestamp("2025-01-01")]
    assert s.iloc[-1] == pytest.approx(136.8)


def test_gscpi_workbook():
    import io

    openpyxl = pytest.importorskip("openpyxl")
    from bondpredictor.data.nyfed import parse_gscpi

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "GSCPI Monthly Data"
    ws.append(["Date", "GSCPI"])
    ws.append([None, None])
    ws.append(["31-Jan-1998", -0.79])
    ws.append(["30-Sep-2026", 1.28])
    buf = io.BytesIO()
    wb.save(buf)
    s = parse_gscpi(buf.getvalue())
    assert list(s.index) == [pd.Timestamp("1998-01-01"), pd.Timestamp("2026-09-01")]
    assert s.iloc[-1] == pytest.approx(1.28)


def test_boe_csv_and_error_page():
    from bondpredictor.data.boe import parse_boe_csv

    s = parse_boe_csv((FIX / "boe_breakeven.csv").read_text())
    assert s.index[0] == pd.Timestamp("1985-01-31") and s.iloc[0] == pytest.approx(7.6138)
    with pytest.raises(ValueError, match="unexpected response"):
        parse_boe_csv("<head><title>Object moved</title></head>")
