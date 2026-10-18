# bondpredictor

Forecasts **government bond excess returns** (holding a long bond instead of
rolling short bills) for the **United States, Germany and Japan**, and
**consumer-price inflation** for the **United States, United Kingdom, Japan
and the euro area** (see [Inflation forecasting](#inflation-forecasting-us-uk-japan-and-the-euro-area)).
Everything is built from **public data**.

It implements the predictors from the bond-return literature, re-estimates
them in real time, and judges them **only out of sample**: R²_OS against the
prevailing mean, the Clark–West test, and a duration-timing strategy.  In-sample
regressions use Hodrick (1992) standard errors and a Bauer–Hamilton-style
small-sample bootstrap.

```
pip install -e ".[dev,ml]"       # numpy, pandas, requests, scikit-learn (+ pytest)
pip install xlrd                 # optional: NY Fed ACM workbook

python -m bondpredictor build                  # download + build data/processed/panel.pkl
python -m bondpredictor insample --country US --bootstrap 500
python -m bondpredictor backtest --ml --jobs 4 --out output   # linear + ML zoo, all countries, pooled panel
python -m bondpredictor robustness --forecasts output/forecasts.pkl --jobs 4 --out output
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

## Machine-learning models and robustness

`backtest --ml` adds four ML forecasters (`ml.py`, scikit-learn):

* random forest and gradient boosting on every predictor plus raw forward rates;
* a neural net on the same inputs;
* a neural net on forwards plus macro factors (the Bianchi–Büchner–Tamoni
  2021 specification);
* `ml:ensemble`, the equal-weighted average of the forest, boosting and net.

The ML models share safeguards suited to ~300–400 overlapping monthly
observations:

* standardised inputs and target;
* small hyperparameter grids, tuned by expanding-window CV with a horizon gap;
* nets averaged over 5 initialisations;
* inputs clamped to the training range;
* yearly refits with the fitted factors frozen alongside.

Clamping was added after the first run, which showed the German net
forecasting +13% to +25% just before the 2022 sell-off.  Negative yields lay
outside its training range and a ReLU net extrapolates linearly.  The
unclamped version is kept as a robustness variant so the effect of that
choice is visible.

`robustness` then runs the following (full output in
[`docs/robustness_report.md`](docs/robustness_report.md)):

* **Diebold–Mariano** (Harvey–Leybourne–Newbold corrected) of every model
  against the slope-only model.
* **Hansen SPA / White Reality Check** across all 12 main models.  This asks
  whether *any* model beats the benchmark once the number of models tried
  is accounted for.
* R²_OS by **sub-period** (1990s, 2000–07, GFC/QE, 2016–21, 2022–), by
  **zero-lower-bound regime**, and by **evaluation start date**.
* About 75 **re-estimation variants**:
  * a rolling 15-year window;
  * other random seeds;
  * each hyperparameter setting fixed instead of CV-tuned;
  * the net without input clamping;
  * training that excludes QE (US), negative rates (DE) or YCC (JP) months.
* Timing-strategy **Sharpe net of 5 bp** per unit of turnover.

### What the robustness tests say (12-month return on the 10y bond)

| | Best ML model, R²_OS | Any model beats the mean? (SPA p) | Any ML model beats the mean? | Any model beats the slope? |
|---|---|---|---|---|
| US 2000–25 | boosting −0.03, forest −0.03 | **yes, 0.02** | no, 0.60 | no, 0.85 |
| DE 1990–25 | forest −0.16 | no, 0.30 | no, 1.00 | no, 1.00 |
| JP 2000–25 | forest **+0.23**, boosting +0.16 | borderline, 0.055 | **yes, 0.03** | no, 0.74 |

1. **ML does not beat the slope anywhere** once you account for having
   tried 12 models.  No ML model has a significantly positive
   Diebold–Mariano statistic against the slope; the best ones are
   statistically tied (US |DM| < 1, Japan's forest +0.57, p = 0.57).  In
   Germany, boosting and the nets are significantly *worse* than the slope
   (DM p = 0.03 and 0.002).
2. **Trees are robust; nets are fragile.**
   * Forest and boosting results barely move across seeds (forecast
     correlation 0.95–0.99) or fixed hyperparameters.  A rolling window
     moves them more (US forest −0.12 vs −0.03).
   * Net results depend on the penalty (US R²_OS −0.30 with strong L2,
     −0.71 with weak) and on the seed set (−0.22 to −0.41).  The
     Bianchi-style forwards-plus-macro net is the worst model in every
     country (R²_OS −0.3 to −1.5).
3. **Japan is the one place ML helps.**  The forest (+0.23) and boosting
   (+0.16) beat the mean after the snooping correction.  Boosting's timing
   Sharpe is 0.45 net of costs vs 0.25 for the mean.  This holds across
   seeds, hyperparameters and a rolling window, but it is not
   significantly better than the slope.  It also rests on the FRED
   par-yield fallback (smoothed monthly averages), so treat it as
   provisional until it is rerun on MOF data.
4. **US timing value from ML is real but second-best.**  Boosting, the net
   and the ensemble lift the strategy Sharpe to about 0.55 (net of costs)
   from 0.31 for the mean.  The linear curve model still does better, at
   0.80.
5. **Predictability is regime-dependent.**  In the US and Japan most
   models do better at the zero lower bound than outside it.  In Germany
   most models fail in the 1990s and 2008–15, and only the slope and the
   forecast combination are positive in nearly every sub-period.
6. The SPA test compares mean squared errors without the Clark–West
   adjustment, so it is conservative for nested comparisons.  That is why
   Germany's slope model (Clark–West p = 0.002 on its own) does not survive
   the 12-model SPA (p = 0.30).

Bottom line on this data: the ML models add little that is robust.  Use the
slope, or the forecast combination, as the core signal.  Gradient boosting
is a reasonable diversifying second model for timing, and the neural nets
are not reliable at this sample size.  Rerun with GSW data from 1961 for a
fairer test of the US neural-net results in the literature, whose evidence
rests on a longer sample.

## Inflation forecasting: US, UK, Japan and the euro area

`bondpredictor.inflation` runs a real-time horse race of **econometric
machine-learning models** for consumer-price inflation in the **United
States, United Kingdom, Japan and the euro area**.  Everything is built from
public data.

```
pip install -e ".[dev,ml,excel]"            # + ".[deep,foundation]" for the tier-2 models
python -m bondpredictor.inflation build                  # download + feature frames -> data/processed/inflation.pkl
python -m bondpredictor.inflation backtest --jobs 4      # tier 1: benchmarks + econometric ML, all horizons, pooled models
python -m bondpredictor.inflation robustness --jobs 4    # seeds, fixed hyperparameters, rolling window, post-1985 sample
python -m bondpredictor.inflation forecast --jobs 4      # latest forecasts
python -m bondpredictor.inflation foundation             # tier 2: Chronos / TimesFM zero-shot (downloads weights)
python -m bondpredictor.inflation deep                   # tier 2: global deep time-series nets (slow; optional)
python -m bondpredictor.inflation report                 # docs/inflation_report.md (+ .csv)
```

### Target and timing

* **Target**: y/y CPI inflation (HICP for the euro area) *h* = 3, 6, 12
  and 24 months after the latest published print.
* **Origins** are month-ends from January 2000.  At the end of month *k*
  the latest CPI print is for *k − 1*.  Every predictor is shifted by its
  publication lag (`inflation/config.py`), so a row holds only what had
  been published by then.
* **Models forecast the change** in the y/y rate.  A zero forecast is the
  random walk, and tree models are not capped at the inflation levels in
  their training data.
* **Seasonality**: y/y rates need no adjustment.  Month-on-month features
  use a one-sided seasonal filter estimated from past data only
  (`features.sa_onesided`), because official SA factors are revised with
  hindsight.  On US CPI it tracks the official SA series closely (0.96
  correlation of 3-month annualised rates).
* **Base effects and taxes**: for *h* < 12 part of the target is already
  known.  `drop` is the annualised rate of the months leaving the 12-month
  window, and `mech` is the change implied if the next *h* months ran at
  the recent trend.  Legislated consumption-tax/VAT changes (Japan 1989,
  1997, 2014, 2019; UK VAT; German VAT in the euro area) enter from the
  month they were announced.
* **Training**: expanding window from 1960 where data exist (euro area
  from 1991), at least ten years.  Linear models are refitted monthly; ML
  models yearly, with hyperparameters tuned by expanding-window CV with a
  horizon-length gap.  `tests/test_inflation.py` checks that scrambling
  all data after an origin leaves its forecast unchanged.

### Data (all reachable without keys)

| | United States | United Kingdom | Japan | Euro area |
|---|---|---|---|---|
| CPI target | CPI-U NSA (FRED, 1913–) | CPI (OECD, 1955–) | CPI (OECD COICOP 2018 table, 1955–) | HICP (Eurostat ECOICOP 2, 1996–), OECD euro-area CPI 1990–95 |
| Core / components | core, energy, food, services, goods, shelter | core, energy, food, services, goods | ex food & energy, energy, food | ex energy & food, energy, food, services, NEIG |
| Expectations / price surveys | Michigan, Cleveland Fed, 5y breakeven, Philly Fed prices paid | 10y gilt breakeven (BoE) | Tankan output/input price DIs and firms' inflation outlook (BoJ) | EC consumer and selling-price expectations |
| Slack | unemployment (level, change, gap), payrolls, claims, wages, IP, capacity utilisation, retail, housing | unemployment, average weekly earnings (ONS), IP, retail, confidence | unemployment, employment, earnings, IP, retail, confidence | unemployment, IP, retail, ESI, confidence |
| Costs | oil, PPI, import prices, commodities, dollar | oil in GBP, output PPI (ONS, to Jan 2025), REER, GBP/USD | oil in JPY, PPI, import prices, services PPI (BoJ), REER, USD/JPY | oil in EUR, PPI, import prices, NEER, EUR/USD |
| Money / finance | Fed funds, 10y, term and credit spreads, M2, loans, stocks, house prices | Bank Rate (BoE), 10y gilt, 10y real yield (BoE), money, stocks | call rate, 10y JGB, money, stocks | Euribor, 10y, M3 |
| Global (all) | median y/y inflation across 20 advanced economies (Ciccarelli–Mojon), its change and dispersion; IMF commodity and food prices; New York Fed supply-chain pressure index (GSCPI); the other three economies' inflation |

Gaps: the UK has no inflation-expectations *survey* (the EC survey dropped
the UK after Brexit), so the gilt breakeven stands in; ONS suspended its
producer price index after January 2025; US CPI for October 2025 was never
collected (federal shutdown) and is interpolated for use as an input, never
as a target.  Every series is kept on a contiguous calendar, so a missing
month cannot shift later 12-month changes.

### Models

**Tier 1 – benchmarks and econometric ML** (run first, full robustness):

| | Model | Reference |
|---|---|---|
| Benchmarks | random walk; UCSV (particle filter, γ fixed); AR-type regression on own-inflation terms | Atkeson–Ohanian 2001; Stock–Watson 2007 |
| Shrinkage | ridge; adaptive lasso; elastic net | Medeiros et al. 2021 (JBES) |
| Factors / combinations | diffusion index (PCA); three-pass regression filter; complete subset regression | Stock–Watson 2002; Kelly–Pruitt 2015; Elliott–Gargano–Timmermann 2013 |
| Non-linear | kernel ridge; random forest; gradient boosting (LightGBM) | Goulet Coulombe et al. 2022 (JAE); Medeiros et al. 2021 |
| Macro-specific ML | **Macroeconomic random forest** (generalised time-varying Phillips-curve coefficients); **BART** (Bayesian trees, predictive densities); **hemisphere neural net** ("neural Phillips curve": one sub-network per block, additive, decomposable) and a plain MLP for comparison | Goulet Coulombe 2024 (JAE); Chipman–George–McCulloch 2010, Clark et al. 2023; Goulet Coulombe 2025 (JBES) |
| Pooled | ridge, boosting, MRF, BART and the hemisphere net estimated on all four economies at once (country-standardised common predictors, country effects) | |
| Ensembles (fixed in advance) | mean of the six non-linear ML models; half that, half UCSV; median of all tier-1 models | |

**Tier 2 – generic deep learning**:

* pretrained time-series *foundation models*, zero-shot: Chronos-Bolt,
  Chronos-2, TimesFM 2.5 and TimesFM 3.0 on the inflation history alone, and
  Chronos-2 and TimesFM 3.0 with five macro series as past covariates (core,
  energy and oil inflation, the global inflation factor, unemployment);
* N-HiTS, PatchTST, TiDE and an LSTM (Nixtla `neuralforecast`), univariate
  but trained globally on ~40 economies' inflation, refitted every two years.

Foundation models may have seen these CPI series in pretraining (their
release dates run from November 2024 to August 2026), so the report also
scores them on post-release origins only.

### Evaluation

RMSE relative to the random walk and to UCSV; Diebold–Mariano tests against
the random walk (HLN-corrected, HAC); the 75% **model confidence set**
(Hansen–Lunde–Nason, T_max, stationary bootstrap); RMSE by the period being
forecast (2000–07, 2008–20, the 2021–22 surge, 2023–); 80% interval
forecasts from rolling conformal intervals (each model's own past errors)
and BART's posterior, scored by coverage and interval score; and
re-estimation variants (seeds, fixed hyperparameters, rolling 20-year
window, training only from 1985).

### Results (October 2026 build)

Full tables (per economy, Diebold–Mariano tests, model confidence sets,
sub-periods, intervals, robustness, forecast decompositions, latest
forecasts): [`docs/inflation_report.md`](docs/inflation_report.md).
Origins run from January 2000 (euro area 2002) to the last month with a
realised target; all sources above were reachable for this build.

Mean RMSE relative to the random walk over the four economies (below 1 beats
"no change"):

| Model | 3m | 6m | 12m | 24m |
|---|---|---|---|---|
| UCSV | 0.80 | 0.82 | 0.92 | 0.86 |
| AR-type, own inflation only | 0.81 | 0.85 | 0.96 | 0.89 |
| Ridge | 0.75 | 0.82 | 0.94 | 0.95 |
| Complete subset regression | 0.74 | 0.80 | 0.92 | 0.87 |
| Kernel ridge | 0.77 | 0.85 | 0.91 | 0.97 |
| Random forest | 0.82 | 0.90 | 0.95 | 0.91 |
| LightGBM | 0.83 | 0.91 | 0.97 | 0.96 |
| Macroeconomic random forest | 0.81 | 0.85 | 0.95 | 0.89 |
| BART | 1.02 | 1.13 | 1.03 | 1.01 |
| Hemisphere neural net | 0.82 | 0.89 | 0.97 | 0.93 |
| Plain neural net (MLP) | 0.78 | 0.82 | 0.89 | 0.93 |
| Pooled ridge | 0.74 | 0.81 | 0.92 | 0.91 |
| Pooled MRF | 0.77 | 0.84 | 0.97 | 0.90 |
| ½ non-linear ML + ½ UCSV | 0.75 | 0.81 | **0.88** | 0.86 |
| Median of all tier-1 models | **0.73** | **0.80** | **0.88** | 0.89 |
| TimesFM 3.0, zero-shot | 0.93 | 0.93 | 0.94 | 0.91 |
| Chronos-2 + macro covariates | 0.89 | 0.90 | 0.88 | 0.83 |
| TimesFM 3.0 + macro covariates | 0.83 | 0.83 | **0.84** | **0.81** |

What it says:

* **Short horizons belong to the econometric models.**  At 3 and 6 months
  the median combination, complete subset regression and pooled ridge beat
  the random walk by 20–27% and every foundation model; the
  Diebold–Mariano test favours the median over TimesFM 3 + covariates in
  all four economies at 3 months (p ≤ 0.02).  Base effects and recent
  momentum do most of the work.
* **A year or more out, gains are modest and statistically thin.**  The best
  models are 12–19% better than no change, and almost every model stays in
  the 75% model confidence set.  TimesFM 3 with macro covariates has the
  lowest RMSE at 12 and 24 months, but its edge over the econometric
  combinations is significant only borderline, in the euro area.  UCSV is
  still hard to beat at 24 months.
* **Foundation models need the macro data.**  On inflation history alone
  Chronos and TimesFM barely beat the random walk (0.91–1.08); with five
  macro covariates they move to the top at long horizons.  Since TimesFM
  2.5 and 3.0 score the same without covariates, and covariate versions
  also beat UCSV on post-release origins, the gain looks real rather than
  memorised pretraining data — but that clean window is short.
* **Among the econometric ML models**, the macroeconomic random forest is
  the steadiest across horizons and robustness checks, the MLP is good at
  12 months, and BART is the weakest everywhere.  BART's ranking is not
  reliable either: its forecasts change markedly with the random seed
  (variant correlation ~0.65), a sign that the MCMC is not converging with
  ~70 predictors.
* **Nobody forecast the 2021–22 surge** 12 months ahead (relative RMSE
  ~0.8–1.1 for target months 2021–22; Japan is the partial exception).
  The models earn their keep in the 2023– disinflation (US hemisphere net
  0.40, euro-area TimesFM 3 + covariates 0.50), while Japan's exit from
  deflation breaks most of them after 2023.
* **Robustness**: dropping the 1970s from training slightly *improves* most
  models; seeds and hyperparameters barely move MRF, random forest,
  LightGBM and kernel ridge.  Letting trees forecast the deviation from the
  mechanical base-effect path cuts their 3-month error sharply (random
  forest 0.82 → 0.74, LightGBM 0.83 → 0.75), so their short-horizon
  weakness is about representing that arithmetic, not about the data.
* **More data did not help.**  Adding the supply-chain index, gilt
  breakevens, Bank Rate and ONS series changed the same models' RMSE by
  ±1–2% on average.
* **Intervals**: rolling conformal 80% intervals cover ~70% of outcomes
  (the surge broke past error patterns); the foundation models' own
  quantiles are the best calibrated, BART's posterior covers only ~30%.

Latest forecasts (median of all tier-1 models / TimesFM 3 + covariates):

| | Latest y/y | 3 months | 12 months | 24 months |
|---|---|---|---|---|
| US (Sep 2026) | 3.7% | 4.2 / 4.2 (Dec 2026) | 2.7 / 3.4 (Sep 2027) | 2.2 / 2.9 (Sep 2028) |
| UK (Aug 2026) | 3.2% | 3.5 / 3.4 (Nov 2026) | 3.9 / 3.7 (Aug 2027) | 3.1 / 3.5 (Aug 2028) |
| Japan (Aug 2026) | 1.9% | 2.4 / 2.0 (Nov 2026) | 3.0 / 2.5 (Aug 2027) | 2.1 / 2.1 (Aug 2028) |
| Euro area (Sep 2026) | 3.7% | 4.3 / 4.0 (Dec 2026) | 3.8 / 3.7 (Sep 2027) | 2.9 / 2.9 (Sep 2028) |

The typical 12-month error is 1.2–1.8 pp, and TimesFM's 80% bands at 12
months are about ±1.5 pp, so treat these as central scenarios.

Caveats: data are the latest releases shifted by publication lags, not
real-time vintages (CPI itself is barely revised); foundation-model scores
before each model's release may benefit from pretraining exposure; and the
pooled BART remains unreliable.

## Layout

```
bondpredictor/
  config.py      series and sources per country (edit here)
  data/          FRED/ALFRED, Bundesbank, Fed GSW, NY Fed, MOF, FRED-MD, OECD, ECB, Eurostat, BoJ, BoE, ONS loaders + cache
  curves.py      Svensson & tabulated curves, par→zero bootstrap, forwards
  returns.py     zero-coupon and par-bond excess returns
  features.py    one-sided predictors (trend inflation, HP gap, inflation uncertainty, realised vol)
  learned.py     CP, global CP, cycle and PCA factors re-fitted inside the backtest
  dataset.py     builds the monthly panel
  models.py      OLS, ridge, lasso, PCR, forecast combination (numpy only)
  ml.py          random forest, gradient boosting, neural-net ensembles (scikit-learn)
  robustness.py  Diebold–Mariano, SPA / Reality Check, sub-period and re-estimation checks
  backtest.py    expanding-window country and pooled-panel backtests
  evaluation.py  R²_OS, Clark–West, timing strategy, CER
  inference.py   Newey–West, Hodrick 1B, null bootstrap
  acm.py         Adrian–Crump–Moench term premium
  cli.py         command line
  inflation/     inflation forecasting (python -m bondpredictor.inflation)
    config.py      series and publication lags per economy
    data.py        loading, splicing, gap handling, cross-country CPI panel
    features.py    real-time features, one-sided seasonal adjustment, base effects, targets
    econ.py        shrinkage, factor, subset, kernel-ridge, forest and boosting models
    mrf.py         macroeconomic random forest
    bart.py        Bayesian additive regression trees
    hnn.py         hemisphere neural net and MLP (PyTorch)
    ucsv.py        Stock-Watson UCSV particle filter
    backtest.py    real-time expanding-window engine and model zoo
    panel.py       pooled cross-country models
    foundation.py  Chronos / TimesFM zero-shot forecasts
    deep.py        N-HiTS, PatchTST, TiDE, LSTM (neuralforecast)
    evaluation.py  DM tests, model confidence set, sub-periods, conformal intervals
    robustness.py  re-estimation variants
    report.py      markdown report
tests/           offline tests (parsers on real file formats, maths, look-ahead checks)
```

Run the tests with `pytest`.  They need no network access.
