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
* Ridge and lasso choose their penalty on a validation block at the end of
  the training window.  The block is separated by a horizon-length gap so
  overlapping returns cannot leak across.
* Predictors not yet observed at an origin, or with too little history,
  are dropped for that origin rather than back-filled.

## Results from this build

RESULTS_PLACEHOLDER

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
