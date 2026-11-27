"""Professional forecasters as a yardstick for the models.

* **United States**: the Survey of Professional Forecasters (Philadelphia
  Fed), median forecasts of quarterly CPI inflation (seasonally adjusted,
  annualised) for the survey quarter and the next four.  The survey goes out
  after the advance GDP release and is returned by the middle of the
  quarter's second month, so respondents know the CPI print for the last
  month of the previous quarter, as the models do at the origin "end of the
  quarter's first month" (the respondents also know a week or two more of
  market data and the jobs report).
* **Euro area**: the ECB Survey of Professional Forecasters, mean point
  forecast of y/y HICP inflation 12 and 24 months after the latest print at
  survey time, which is exactly the models' target.  The survey is run in
  the first weeks of the quarter, so the models' origin with the same latest
  print (the end of that month) sees about three more weeks of market data;
  ``lag=1`` handicaps the models by one print instead.

US forecasts are compared on two bases, because the survey forecasts
quarterly averages and the models a month: "monthly" scores both against y/y
inflation in the quarter's last month (the survey's quarterly figure stands
in for that month), "quarterly" scores both against y/y inflation of the
quarterly average (the models' path is interpolated between their horizons).
"""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd

from ..data import Downloader, fetch_raw

SPF_CPI_URL = ("https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
               "survey-of-professional-forecasters/data-files/files/median_cpi_level.xlsx")
ECB_SPF = {12: "SPF/M.U2.HICP.POINT.P12M.Q.AVG", 24: "SPF/M.U2.HICP.POINT.P24M.Q.AVG"}
US_QUARTERS_AHEAD = {3: 0, 6: 1, 12: 3}  # model horizon in months -> survey quarter relative to the survey quarter


def parse_spf_cpi(raw: bytes) -> pd.DataFrame:
    """Median CPI forecasts by survey quarter: CPI2 is the survey quarter, CPI3..CPI6 the next four (annualised %)."""
    df = pd.read_excel(io.BytesIO(raw))
    df = df.dropna(subset=["YEAR", "QUARTER"])
    idx = pd.PeriodIndex([f"{int(y)}Q{int(q)}" for y, q in zip(df["YEAR"], df["QUARTER"])], freq="Q", name="survey")
    out = df.drop(columns=["YEAR", "QUARTER"]).apply(pd.to_numeric, errors="coerce")
    out.index = idx
    return out.sort_index()


def us_spf_yoy(spf: pd.DataFrame, cpi_sa: pd.Series) -> pd.DataFrame:
    """Survey forecasts of y/y CPI inflation of the quarterly average, ``j`` quarters after the survey quarter.

    The forecast quarterly growth rates are chained onto the last published quarterly level and
    divided by the level four quarters before the target (published, or on the forecast path).
    """
    q = cpi_sa.dropna().resample("QS").mean()
    q.index = q.index.to_period("Q")
    rows = []
    for Q, r in spf.iterrows():
        if Q - 1 not in q.index:
            continue
        level, path = q[Q - 1], {}
        for j, col in enumerate(["CPI2", "CPI3", "CPI4", "CPI5", "CPI6"]):
            if not np.isfinite(r.get(col, np.nan)):
                break
            level = level * (1.0 + r[col] / 100.0) ** 0.25
            path[j] = level
            base = path[j - 4] if j >= 4 else q.get(Q + j - 4, np.nan)
            rows.append({"survey": Q, "j": j, "forecast": 100.0 * (level / base - 1.0)})
    return pd.DataFrame(rows)


def _nw_t(d: np.ndarray, lags: int) -> tuple[float, float]:
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 8:
        return np.nan, np.nan
    x = d - d.mean()
    v = x @ x / n
    for k in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    t = d.mean() / math.sqrt(max(v, 1e-18) / n)
    return float(t), float(math.erfc(abs(t) / math.sqrt(2.0)))


def _score(df: pd.DataFrame, h: int, cols: list[str]) -> dict:
    """RMSE of each forecast column and of no change, on the rows where everything is observed.

    Each model also enters as ``mix:<model>``, the equal-weight average of the model and the survey:
    if the mix beats the survey, the model holds information the forecasters did not use.
    """
    df = df.copy()
    for c in [c for c in cols if c != "survey"]:
        df[f"mix:{c}"] = 0.5 * (df[c] + df["survey"])
    cols = cols + [f"mix:{c}" for c in cols if c != "survey"]
    x = df.dropna(subset=["actual", "rw", *cols])
    out = {"n": len(x), "from": str(x.index.min()) if len(x) else "", "to": str(x.index.max()) if len(x) else ""}
    if x.empty:
        return out
    rw = math.sqrt(np.mean((x["actual"] - x["rw"]) ** 2))
    out["rmse_rw"] = rw
    for c in cols:
        out[f"rel_{c}"] = math.sqrt(np.mean((x["actual"] - x[c]) ** 2)) / rw
    for c in cols:
        if c != "survey":  # Diebold-Mariano against the survey; negative t = the model is better
            d = ((x["actual"] - x[c]) ** 2 - (x["actual"] - x["survey"]) ** 2).to_numpy()
            out[f"dm_t_{c}"], out[f"dm_p_{c}"] = _nw_t(d, max(h // 3 - 1, 0))
    return out


def us_comparison(frame: pd.DataFrame, results: dict, spf: pd.DataFrame, cpi_sa: pd.Series,
                  models=("ens:median",), start: str = "2000Q1") -> pd.DataFrame:
    """Models against the US survey.  ``results``: {(horizon, model): result frame indexed by origin}."""
    sy = us_spf_yoy(spf, cpi_sa)
    pi = frame["pi"]
    pi_ref = pi.set_axis(pi.index - 1)  # reference month of the print
    q = cpi_sa.dropna().resample("QS").mean()
    q.index = q.index.to_period("Q")
    yoy_q = 100.0 * (q / q.shift(4) - 1.0)
    rows = []
    for h, j in US_QUARTERS_AHEAD.items():
        s = sy[sy["j"] == j].set_index("survey")["forecast"]
        s = s[s.index >= pd.Period(start, "Q")]
        for basis in ("monthly", "quarterly"):
            recs = []
            for Q, f in s.items():
                k = Q.asfreq("M", "start")  # origin: end of the quarter's first month; latest print = month before
                m = k - 1
                rec = {"survey_q": Q, "survey": f, "rw": pi.get(k, np.nan)}
                if basis == "monthly":
                    rec["actual"] = pi_ref.get(m + h, np.nan)
                    for name in models:
                        r = results.get((h, name))
                        rec[name] = r["forecast"].get(k, np.nan) if r is not None else np.nan
                else:
                    rec["actual"] = yoy_q.get(Q + j, np.nan)
                    months = [m + 3 * j + i for i in (1, 2, 3)]
                    for name in models:
                        pts = {0: pi.get(k, np.nan)}
                        for hh in (3, 6, 12, 24):
                            r = results.get((hh, name))
                            if r is not None and k in r.index:
                                pts[hh] = r["forecast"][k]
                        xs = sorted(pts)
                        ys = [pts[x] for x in xs]
                        ok = np.all(np.isfinite(ys)) and max(xs) >= 3 * j + 3
                        rec[name] = float(np.mean(np.interp([(mm - m).n for mm in months], xs, ys))) if ok else np.nan
                recs.append(rec)
            df = pd.DataFrame(recs).set_index("survey_q")
            rows.append({"economy": "US", "horizon": h, "basis": basis, **_score(df, h, ["survey", *models])})
    return pd.DataFrame(rows)


def ea_comparison(frame: pd.DataFrame, results: dict, ecb: dict[int, pd.Series], models=("ens:median",),
                  lag: int = 0) -> pd.DataFrame:
    """Models against the ECB survey.  ``ecb``: horizon -> forecasts indexed by target month.

    ``lag=0``: the models' origin has the same latest print as the survey.  ``lag=1``: the models
    forecast one month earlier (one print fewer), still scored against the survey's target month.
    """
    pi = frame["pi"]
    pi_ref = pi.set_axis(pi.index - 1)
    rows = []
    for h, s in ecb.items():
        recs = []
        for T, f in s.dropna().items():
            k = T - h + 1 - lag
            rec = {"target": T, "survey": f, "actual": pi_ref.get(T, np.nan), "rw": pi.get(T - h + 1, np.nan)}
            for name in models:
                r = results.get((h, name))
                rec[name] = r["forecast"].get(k, np.nan) if r is not None else np.nan
            recs.append(rec)
        df = pd.DataFrame(recs).set_index("target")
        rows.append({"economy": "EA", "horizon": h, "basis": "monthly" if lag == 0 else "monthly, one print fewer",
                     **_score(df, h, ["survey", *models])})
    return pd.DataFrame(rows)


def load_surveys(dl: Downloader) -> dict:
    """Download the survey data: {"us_spf", "us_cpi_sa", "ecb": {12: series, 24: series}}."""
    ecb = {}
    for h, key in ECB_SPF.items():
        s = fetch_raw(dl, "ecb", key).dropna()
        ecb[h] = s.set_axis(pd.PeriodIndex(s.index, freq="M"))
    cpi = fetch_raw(dl, "fred", "CPIAUCSL").dropna()
    return {"us_spf": parse_spf_cpi(dl.get(SPF_CPI_URL, name="phil_spf_median_cpi_level.xlsx")),
            "us_cpi_sa": cpi.set_axis(pd.DatetimeIndex(cpi.index)), "ecb": ecb}


def compare(store: dict, results: dict, surveys: dict, models=("ens:median", "ucsv")) -> pd.DataFrame:
    """All comparisons.  ``results``: {(economy, horizon, model): frame}, ensembles included.

    Each economy is scored on the models it has forecasts for (the combined forecast exists only where adopted).
    """
    us = {(h, m): r for (c, h, m), r in results.items() if c == "US"}
    ea = {(h, m): r for (c, h, m), r in results.items() if c == "EA"}
    mu = tuple(m for m in models if any(k[1] == m for k in us))
    me = tuple(m for m in models if any(k[1] == m for k in ea))
    parts = [us_comparison(store["frames"]["US"][0], us, surveys["us_spf"], surveys["us_cpi_sa"], mu),
             ea_comparison(store["frames"]["EA"][0], ea, surveys["ecb"], me, lag=0),
             ea_comparison(store["frames"]["EA"][0], ea, surveys["ecb"], me, lag=1)]
    return pd.concat(parts, ignore_index=True)
