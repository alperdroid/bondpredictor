from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bondpredictor.data.bundesbank import parse_bundesbank_csv
from bondpredictor.data.fred import first_release, parse_fredgraph_csv
from bondpredictor.data.fredmd import parse_fredmd, transform
from bondpredictor.data.japan import parse_mof_csv
from bondpredictor.data.us import gsw_parameters, parse_gsw_csv

FIX = Path(__file__).parent / "fixtures"


def test_fred_csv_handles_blank_and_dot_missing():
    s = parse_fredgraph_csv((FIX / "fred_dgs10.csv").read_text(), "DGS10")
    assert list(s.round(2)) == [4.06, 4.03, 3.99, 4.02]
    assert s.index[0] == pd.Timestamp("1962-01-02")


def test_bundesbank_monthly_csv():
    s = parse_bundesbank_csv((FIX / "bundesbank_b0.csv").read_text(encoding="utf-8"))
    assert s.index[0] == pd.Timestamp("1972-09-01")
    assert s.iloc[0] == pytest.approx(8.58387)
    assert len(s) == 7  # 4 head rows + 3 tail rows


def test_bundesbank_daily_csv_skips_missing():
    s = parse_bundesbank_csv((FIX / "bundesbank_daily.csv").read_text(encoding="utf-8"))
    assert len(s) == 1
    assert s.index[0] == pd.Timestamp("2026-10-12") and s.iloc[0] == pytest.approx(3.53)


GSW_SAMPLE = """\
"The Federal Reserve Board"
"Gurkaynak, Sack, and Wright (2006)"
"Series","Compounding"
Date,BETA0,BETA1,BETA2,BETA3,SVENY01,SVENY02,SVENY10,SVENY20,TAU1,TAU2
1971-01-04,6.5,-1.0,-2.0,NA,6.1,6.2,6.4,NA,1.5,NA
2024-01-02,4.5,0.8,-2.0,3.0,4.7,4.3,4.0,4.3,1.8,12.0
"""


def test_gsw_parse_and_max_maturity():
    raw = parse_gsw_csv(GSW_SAMPLE)
    params = gsw_parameters(raw)
    assert list(params.columns) == ["beta0", "beta1", "beta2", "beta3", "tau1", "tau2", "max_maturity"]
    assert params["max_maturity"].tolist() == [10.0, 20.0]
    assert np.isnan(params["beta3"].iloc[0])


MOF_SAMPLE = """\
Interest Rate,,,,,,,,,,,,,,,
Date,1Y,2Y,3Y,4Y,5Y,6Y,7Y,8Y,9Y,10Y,15Y,20Y,25Y,30Y,40Y
1974/9/24,10.327,9.362,8.83,8.515,8.348,8.29,8.24,8.121,8.127,-,-,-,-,-,-
2024/3/29,0.035,0.19,0.27,0.335,0.38,0.46,0.555,0.645,0.68,0.725,1.11,1.55,1.73,1.88,2.075
"""


def test_mof_parse():
    df = parse_mof_csv(MOF_SAMPLE)
    assert df.columns.tolist()[:3] == [1.0, 2.0, 3.0] and 40.0 in df.columns
    assert np.isnan(df.loc["1974-09-24", 10.0])
    assert df.loc["2024-03-29", 10.0] == pytest.approx(0.725)


FREDMD_SAMPLE = """\
sasdate,INDPRO,UNRATE,CPIAUCSL
Transform:,5,2,6
1/1/2020,100,3.5,250
2/1/2020,101,3.6,251
3/1/2020,99,4.4,252
"""


def test_fredmd_parse_and_transform():
    levels, tcodes = parse_fredmd(FREDMD_SAMPLE)
    assert tcodes == {"INDPRO": 5, "UNRATE": 2, "CPIAUCSL": 6}
    assert levels.index[0] == pd.Period("2020-01", "M")
    ip = transform(levels["INDPRO"], 5)
    assert ip.iloc[1] == pytest.approx(np.log(101 / 100))
    assert transform(levels["UNRATE"], 2).iloc[2] == pytest.approx(0.8)


def test_alfred_first_release():
    obs = [
        {"date": "2020-01-01", "realtime_start": "2020-02-14", "value": "1.0"},
        {"date": "2020-01-01", "realtime_start": "2020-03-14", "value": "1.2"},
        {"date": "2020-02-01", "realtime_start": "2020-03-14", "value": "."},
        {"date": "2020-02-01", "realtime_start": "2020-04-14", "value": "2.0"},
    ]
    s = first_release(obs)
    assert s.tolist() == [1.0, 2.0]
