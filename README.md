# bondpredictor

Forecasts **government bond excess returns** (holding a long bond instead of
rolling short bills) for the **United States, Germany and Japan**.  Everything
is built from **public data**.

It implements the predictors from the bond-return literature, re-estimates
them in real time, and judges them **only out of sample**: R²_OS against the
prevailing mean, the Clark–West test, and a duration-timing strategy.  In-sample
regressions use Hodrick (1992) standard errors and a Bauer–Hamilton-style
small-sample bootstrap.

```
pip install -e ".[dev]"          # numpy, pandas, requests (+ pytest)
pip install xlrd                 # optional: NY Fed ACM workbook

python -m bondpredictor build                  # download + build data/processed/panel.pkl
python -m bondpredictor insample --country US --bootstrap 500
python -m bondpredictor backtest --out output  # full model zoo, all countries, + pooled panel
python -m bondpredictor forecast --model curve # latest forecast per country
python -m bondpredictor termpremium --country DE
```

Downloads are cached in `data/cache/`.  Re-runs are fast, and `--offline`
works from the cache.

## What is modelled

Target: the 12-month log excess return on a 10-year zero-coupon bond.
Use `--maturity` and `--horizon 1` for other targets.  Forecast origins are
month-ends.  Each predictor block is added in the build order recommended for
this kind of model.

| Block (`pipeline.py`) | Predictors | Implementation |
|---|---|---|
| `slope` | 10y − 1y zero yield (Fama–Bliss, Campbell–Shiller) | `dataset.py` |
| `cp` | Cochrane–Piazzesi factor: average 2–5y excess return regressed on 1–5y forwards | `learned.ReturnFactor`, **re-fitted each month** |
| `global` | Global CP (equal-weighted average of local CP factors, Dahlquist–Hasseltoft); US slope as a spillover for DE/JP | `learned.GlobalAverage` |
| `cycle` | Cieslak–Povala cycle factor: yields minus a discounted moving average of core inflation, then a CP-style regression | `features.trend_inflation`, `learned.CycleFactor` |
| `carry` | Carry + roll-down of the 10y zero over one month (Koijen et al.) | `dataset.py` |
| `macro` | Ludvigson–Ng PCA factors (FRED-MD for the US; curated FRED panels for DE/JP), one-sided-HP output gap, inflation uncertainty | `learned.PCAFactors`, `features.output_gap`, `features.inflation_uncertainty` |
| `value_mom` | Real yield (10y minus trend inflation), 12-month time-series momentum | `dataset.py` |
| `risk` | Realised rate volatility from daily yields (free MOVE substitute), VIX, US credit spread, Italy–Bund spread for Germany | `features.realized_volatility` |
| `supply_policy` | Δ debt/GDP, Δ central-bank holdings/GDP, 3-month change in the short rate, zero-lower-bound flag, regime dummies (QE, NIRP, APP, QQE, YCC) | `config.py` |

Models in the zoo: OLS on `slope`, `cp`, `cp+global`; ridge on `curve`,
`curve+macro` and `all` blocks.  `all` also runs with lasso, PCR (3
components) and an equal-weighted forecast combination.  A **pooled panel**
(`panel:ridge`) shares slopes across countries and keeps country fixed
effects.

`acm.py` estimates the **Adrian–Crump–Moench term premium** for any curve.
There is no official German or Japanese series, so it is needed for both.
On the US curve its 10-year term premium tracks the published Kim–Wright
series with a correlation of 0.96.

## Data sources

All free.  `config.py` is the single place that says which series feeds what.

| | United States | Germany | Japan |
|---|---|---|---|
| Zero curve | GSW Svensson parameters (Fed, 1961–); fallback Kim–Wright zero yields via FRED (1990–) | Bundesbank Svensson parameters, month-end (1972–) | MOF JGB par yields (1974–), bootstrapped to zeros; fallback OECD 10y via FRED with par-bond returns |
| Funding rate | 3m T-bill (DTB3) | Curve's 3m yield (3m interbank as backup) | Overnight call rate |
| Daily yields (realised vol) | DGS10 | Bundesbank daily 10y zero | MOF daily 10y |
| Inflation / trend | CPI, core CPI | OECD CPI → HICP; OECD core → HICP ex energy, food, alcohol & tobacco | OECD CPI/core (ends 2021-06 on FRED) → GDP deflator |
| Output gap | Real GDP, one-sided HP | Real GDP | Real GDP |
| Macro panel | FRED-MD; fallback 28 FRED series | 15 FRED/OECD series | 10 FRED/OECD series |
| CB holdings / GDP | Fed Treasury holdings (FDHBFRBN) / GDP | Eurosystem assets / euro-area GDP | BoJ assets / GDP |
| Debt / GDP | Debt held by public / GDP | IMF debt/GDP (annual) | IMF debt/GDP (annual) |
| Stress | VIX, Baa–10y spread | Italy–Germany 10y spread | VIX |

### Gaps in the free data

* **Implied rate vol (MOVE, swaptions)** is paid.  Realised volatility of
  daily yield changes is used instead.
* **Japanese CPI and IP.**  FRED's OECD copies stop in 2021 and 2024.
  Inflation is then extended with the GDP deflator.  A better fix is to
  download CPI ex fresh food from e-Stat and point `config.JP.core_cpi_extension`
  at it with `Series("csv", "path/to/jp_core_cpi.csv")`.  Any series can be
  overridden this way with a two-column date/value file.
* **Surveys.**  The SPF/Consensus dispersion of inflation forecasts (Wright
  2011) is replaced by the volatility of real-time AR(1) inflation surprises.
* **Real-time vintages.**  Data are the latest revised releases, shifted by
  conservative publication lags (1–2 months for monthly macro, 1–2 months
  after quarter-end for GDP, 4 months after year-end for annual debt).
  `data.fred.fetch_fred_first_release` pulls first-release US data from
  ALFRED if you set a free `FRED_API_KEY`.
* **Curve smoothing.**  Fitted curves (GSW, Kim–Wright, Bundesbank) smooth
  forward rates, which weakens the CP factor.  The fitted CP weights on
  Kim–Wright forwards come out huge and offsetting; that is a property of
  the input, not a bug.

## How look-ahead is prevented

* Every series sits on a month-end grid shifted by its publication lag
  (`config.Series.lag`).  Quarterly and annual data are held only until the
  next release would be due.  The current, incomplete month is dropped.
* At each forecast origin *t*, `backtest.information_set` blanks every
  return that is not realised by *t*.  A 12-month return stored at *s* is
  hidden until *s* + 12.
* Learned factors (CP, global CP, cycle, macro PCs) are re-fitted at every
  origin on that information set.  `tests/test_backtest.py` checks this:
  scrambling all data after an origin must leave its forecast unchanged.
* Ridge and lasso choose their penalty by expanding-window CV inside the
  training window: five validation blocks over its second half, each with
  a horizon-length gap so overlapping returns cannot leak across.  The
  penalty is re-selected every 12 months; coefficients are refitted every
  month.
* Returns that would need data after the last complete month are blanked,
  so a partial month never completes a target.
* Predictors not yet observed at an origin, or with too little history,
  are dropped for that origin rather than back-filled.

## Results from this build

Run on 13 October 2026 from a sandbox that could reach **FRED and the
Bundesbank only**.  Three sources therefore used their fallbacks:

* **US curve**: Kim–Wright zero yields (1990–), not GSW (1961–).  The OOS
  window starts in Dec 2000.
* **US macro panel**: the curated FRED panel, not FRED-MD.
* **Japan**: no MOF zero curve.  The OECD 10-year yield (a *monthly
  average*) with par-bond returns, so there is no CP or cycle factor, and
  inflation is extended with the GDP deflator after mid-2021.

Run `build` + `backtest` on a machine with full internet access to get the
GSW, MOF and FRED-MD versions.

Target: 12-month log excess return on the 10y bond.  Expanding window with
at least 120 training months.  All models for a country are scored on the
same months (US 2000-12..2025-09, DE 1990-01..2025-09, JP 1999-12..2025-08).
The strategy is mean–variance timing (γ = 5, weight clipped to [−1, 2])
rebalanced monthly on one-month returns.  CER gain is against the same
strategy driven by the prevailing mean.

| | Model | R²_OS | Clark–West p | Sharpe: model / mean / hold | CER gain (bp/yr) |
|---|---|---|---|---|---|
| US | slope | +0.14 | 0.001 | 0.44 / 0.31 / 0.36 | +142 |
| US | cp | -0.23 | 0.074 | 0.19 / 0.31 / 0.36 | +26 |
| US | cp+global | -0.27 | 0.083 | 0.12 / 0.31 / 0.36 | -19 |
| US | curve | +0.16 | 0.000 | 0.83 / 0.31 / 0.36 | +540 |
| US | curve+macro | +0.15 | 0.000 | 0.78 / 0.31 / 0.36 | +497 |
| US | all:ridge | -0.17 | 0.020 | 0.38 / 0.31 / 0.36 | +148 |
| US | all:lasso | -0.22 | 0.005 | 0.69 / 0.31 / 0.36 | +421 |
| US | all:pcr | -0.11 | 0.004 | 0.42 / 0.31 / 0.36 | +185 |
| US | all:combo | +0.10 | 0.002 | 0.40 / 0.31 / 0.36 | +154 |
| US | panel:ridge | +0.05 | 0.002 | 0.62 / 0.31 / 0.36 | +363 |
| DE | slope | +0.11 | 0.002 | 0.56 / 0.39 / 0.46 | +159 |
| DE | cp | -0.47 | 0.285 | -0.23 / 0.39 / 0.46 | -370 |
| DE | cp+global | -0.32 | 0.064 | 0.26 / 0.43 / 0.46 | -113 |
| DE | curve | -0.17 | 0.027 | 0.38 / 0.43 / 0.46 | -22 |
| DE | curve+macro | -0.39 | 0.192 | 0.26 / 0.43 / 0.46 | -145 |
| DE | all:ridge | -0.25 | 0.297 | 0.26 / 0.43 / 0.46 | -144 |
| DE | all:lasso | -0.52 | 0.337 | 0.09 / 0.43 / 0.46 | -325 |
| DE | all:pcr | -0.11 | 0.034 | 0.38 / 0.43 / 0.46 | -32 |
| DE | all:combo | +0.06 | 0.014 | 0.44 / 0.43 / 0.46 | +14 |
| DE | panel:ridge | -0.12 | 0.061 | 0.29 / 0.43 / 0.46 | -109 |
| JP | slope | +0.17 | 0.000 | 0.29 / 0.25 / 0.25 | +26 |
| JP | cp+global | +0.21 | 0.000 | -0.04 / 0.25 / 0.25 | -119 |
| JP | curve | +0.32 | 0.000 | 0.17 / 0.25 / 0.25 | -33 |
| JP | curve+macro | +0.13 | 0.006 | 0.26 / 0.25 / 0.25 | +12 |
| JP | all:ridge | +0.19 | 0.014 | 0.34 / 0.25 / 0.25 | +59 |
| JP | all:lasso | +0.00 | 0.010 | 0.15 / 0.25 / 0.25 | -42 |
| JP | all:pcr | +0.12 | 0.012 | 0.02 / 0.25 / 0.25 | -107 |
| JP | all:combo | +0.07 | 0.002 | 0.25 / 0.25 / 0.25 | +0 |
| JP | panel:ridge | +0.20 | 0.007 | 0.27 / 0.25 / 0.25 | +16 |

How to read this:

* **The slope is the most reliable predictor.**  It beats the historical
  mean in all three countries (R²_OS 11–17%, Clark–West p ≤ 0.002) and
  improves the timing Sharpe ratio in each.  This matches the out-of-sample
  literature (Thornton–Valente; Gargano–Pettenuzzo–Timmermann).
* **The CP factor does well in sample and badly out of sample.**  In-sample
  R² is 0.20 (US) and 0.11 (DE), with Hodrick t-stats of 2.4 and 3.0, but
  R²_OS is −0.23 and −0.47.  Fitted curves (Kim–Wright, Bundesbank)
  smooth forwards and make the five-forward regression unstable.  This is
  the Bauer–Hamilton / smoothing problem in practice.
* **Global information helps Japan.**  The global CP factor plus the US
  slope give R²_OS 0.21 for JGBs, consistent with US yields acting as the
  global factor.
* **The US curve block (slope, CP, global CP, cycle, carry under ridge) is
  the best US model.**  R²_OS is 0.16 and the timing Sharpe 0.83 vs 0.31.
  Adding macro factors adds nothing out of sample.
* **Kitchen-sink models overfit.**  All 20+ predictors under ridge or lasso
  give negative R²_OS in the US and Germany, even with cross-validated
  shrinkage.  Forecast combination is the only "all predictors" model that
  stays positive everywhere.  The pooled panel helps Japan (+0.20), is
  small for the US (+0.05) and negative for Germany.
* A significant Clark–West statistic with a negative R²_OS (e.g. US
  `all:lasso`) means the predictors carry information but estimation noise
  costs more than it gains.  That pattern is common at this sample size.
* Japan's R²_OS is flattered by the smoothed monthly-average yields and
  near-zero rates, and it does not turn into trading gains.  Treat it as
  provisional until rerun on MOF data.

In-sample, for comparison (`insample --bootstrap 300`, 12-month return on the
10y bond):

| | Predictor | R² | t (Newey–West) | t (Hodrick 1B) | bootstrap p |
|---|---|---|---|---|---|
| US 1990–2025 | slope | 0.13 | 3.5 | 1.9 | 0.03 |
| US 1990–2025 | CP | 0.20 | 3.6 | 2.4 | 0.02 |
| DE 1972–2025 | slope | 0.12 | 2.8 | 2.7 | 0.04 |
| DE 1972–2025 | CP | 0.11 | 3.7 | 3.0 | 0.003 |

The bootstrap treats CP as observed rather than estimated, so its p-value
is optimistic.  The out-of-sample table is the one to trust.

## Layout

```
bondpredictor/
  config.py      series and sources per country (edit here)
  data/          FRED/ALFRED, Bundesbank, Fed GSW, NY Fed ACM, MOF, FRED-MD loaders + cache
  curves.py      Svensson & tabulated curves, par→zero bootstrap, forwards
  returns.py     zero-coupon and par-bond excess returns
  features.py    one-sided predictors (trend inflation, HP gap, inflation uncertainty, realised vol)
  learned.py     CP, global CP, cycle and PCA factors re-fitted inside the backtest
  dataset.py     builds the monthly panel
  models.py      OLS, ridge, lasso, PCR, forecast combination (numpy only)
  backtest.py    expanding-window country and pooled-panel backtests
  evaluation.py  R²_OS, Clark–West, timing strategy, CER
  inference.py   Newey–West, Hodrick 1B, null bootstrap
  acm.py         Adrian–Crump–Moench term premium
  cli.py         command line
tests/           offline tests (parsers on real file formats, maths, look-ahead checks)
```

Run the tests with `pytest`.  They need no network access.
