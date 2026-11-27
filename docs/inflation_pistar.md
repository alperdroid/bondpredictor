# Partial horizons and a persistent trend state (ECB Working Paper 3307)

Two ideas from Montes-Galdón (2026), *Deep-π\*: a neural network estimate of trend inflation*, ECB Working Paper 3307, tested on this repository's real-time inflation backtest (`python -m bondpredictor.inflation pistar run`, then `pistar report`).

Forecast origins 2000-01 to 2026-10; every model re-estimated at each origin on data published by then. RMSE is relative to the random walk (below 1 beats "no change"); stars mark Harvey-Leybourne-Newbold Diebold-Mariano p-values below 0.10/0.05/0.01 (`(-)` when the variant is the less accurate one).

## Findings (November 2026 build)

* **Partial horizons do not help the direct models.**  Over 2000-2025 the stacked versions match or trail the
  standard ones 12 months ahead (mean RMSE relative to the random walk: ridge 0.93 -> 0.93, elastic net
  0.99 -> 0.99, LightGBM 0.96 -> 0.99, random forest 0.95 -> 1.00, AR-type 0.96 -> 0.98).  The partial rows
  move results by about 0.01 against the complete-origin control, in either direction.  On the paper's US
  2021-23 window ridge gains (0.69 as it is, 0.74 with complete origins, 0.61 with partial ones; its
  January 2021 forecast rises from 2.3% to 4.0%), but the elastic net and the trees lose there too.
* **The linear persistent-state model does not beat the existing models.**  The paper's design averages
  0.94 against 0.88 for the repository's median and 0.92 for UCSV; it is best in Japan (0.81).  Removing
  the memory (theta = 0) does as well (0.92), and neither the state nor the partial horizons is
  significant in any economy at the 5% level.  Validation shrinks the predictors' part of the trend hard
  (the penalty sits at the top of its grid at 89% of US origins and every UK origin), so the trend is close
  to a re-estimated mean: the predictors moved it by 0.3-0.5 pp between January 2019 and 2022 in real time,
  against about 5 pp for the paper's neural network.  The paper's mechanism needs the non-linear
  innovation; a linear one does not reproduce it.
* **On the paper's window several simple models do as well as Deep-pi\*.**  At the 12 SPF dates the state
  models score 2.14-2.37 (paper: 2.14-2.25; SPF 3.44; the repository's median 2.61), but so do the
  no-state variant (1.95), ridge with partial horizons (1.89) and the elastic net as it is (2.06).  Doing
  well in 2021-23 does not single out the paper's ingredients.
* **Adding the state model to the median helps by about 0.01** (DM p < 0.10 in the UK and Japan).
* **Diagnostic, run after the main results:** trained from 1985, the no-state variant averages 0.84, the best
  12-month number here.  The median was not retrained from 1985, so this is a lead for a pre-specified test,
  not a result.
* Two estimation lessons: the forward loss has two local optima in the AR coefficients, so the fit is
  multi-started (a single warm start left the UK at a trend-led optimum, RMSE 1.46); and taxes need a
  pass-through, estimated here from earlier events.

## What was tested

**Partial-horizon training** (`partial.py`).  A direct 12-month model normally learns only from origins whose 12-month outcome is published, so in February 2021 its newest training row was February 2020.  The stacked version trains on horizons 1-12 at once, each origin entering with every horizon already out, on a common scale (average inflation over the next *m* months minus today's y/y rate, which is exactly the usual target at *m* = 12), weighted by *m*/12.  *Complete* uses the shorter horizons only for origins whose 12 months are all out; *partial* adds the origins of the last 11 months, scored on the months already published, as in the paper.  The gap between the two is the effect of the partial rows alone.

**Persistent-state trend** (`pistar.py`).  Monthly seasonally adjusted inflation follows an AR(12) around a trend π\*[k] = θ π\*[k-1] + (1-θ)(α + x[k]'β), the paper's recursion with a linear innovation instead of its neural network; θ = 0.995 (half-life 138 months), x = every predictor except headline CPI terms.  The trend is an exponentially weighted average of past predictors; forecasts iterate the AR with the trend decaying towards α.  The forward-looking loss scores the iterated forecasts 1-12 months ahead from every training origin (with or without partial paths), the one-month loss only the next month.  θ = 0 removes the memory.  The ridge penalty on β (grid 0.0001, 0.000562341, 0.00316228, 0.0177828, 0.1, 0.562341, 3.16228) is chosen every 12 months by expanding-window validation on the model's own loss.  Not reproduced: the paper's neural network and attention layer, time-varying volatility and Student-t errors.

## Partial-horizon training, 12 months ahead

Stars compare each stacked version with the same model as it is.

| model | US | UK | Japan | Euro area | mean |
|---|---|---|---|---|---|
| AR-type, own inflation only: as is | 0.99 | 0.97 | 0.94 | 0.95 | 0.96 |
| AR-type, own inflation only: horizons 1-12, complete origins | 1.01(-)** | 0.97 | 0.96(-)** | 0.97(-)** | 0.98 |
| AR-type, own inflation only: horizons 1-12 + partial origins | 1.01(-)* | 0.97 | 0.96(-)** | 0.98(-)*** | 0.98 |
| Elastic net: as is | 0.97 | 1.00 | 1.00 | 0.98 | 0.99 |
| Elastic net: horizons 1-12, complete origins | 0.95 | 1.02 | 0.95 | 0.98 | 0.98 |
| Elastic net: horizons 1-12 + partial origins | 1.06(-)* | 0.98 | 0.99 | 0.94 | 0.99 |
| Gradient boosting (LightGBM): as is | 0.87 | 1.03 | 0.99 | 0.96 | 0.96 |
| Gradient boosting (LightGBM): horizons 1-12, complete origins | 0.85 | 1.05 | 1.01 | 1.03 | 0.98 |
| Gradient boosting (LightGBM): horizons 1-12 + partial origins | 0.93(-)* | 1.01 | 0.98 | 1.03 | 0.99 |
| Random forest: as is | 0.92 | 0.99 | 0.92 | 0.98 | 0.95 |
| Random forest: horizons 1-12, complete origins | 0.92 | 1.03 | 0.94 | 1.07 | 0.99 |
| Random forest: horizons 1-12 + partial origins | 1.02 | 1.01 | 0.95 | 1.02 | 1.00 |
| Ridge: as is | 0.96 | 0.89 | 0.89 | 0.99 | 0.93 |
| Ridge: horizons 1-12, complete origins | 0.99 | 0.91 | 0.85* | 1.01 | 0.94 |
| Ridge: horizons 1-12 + partial origins | 0.95 | 0.92 | 0.85 | 1.00 | 0.93 |

## Persistent-state model, 12 months ahead

Stars: DM test against the random walk. `Median of all models` is the repository's fixed-in-advance median of the 13 tier-1 country models.

| model | US | UK | Japan | Euro area | mean |
|---|---|---|---|---|---|
| Random walk (no change) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| UCSV (Stock-Watson) | 0.89 | 0.94 | 0.85* | 1.00 | 0.92 |
| AR-type, own inflation only | 0.99 | 0.97 | 0.94 | 0.95 | 0.96 |
| Median of all models | 0.88** | 0.90 | 0.84** | 0.90 | 0.88 |
| Plain neural net (MLP) | 0.86 | 0.83 | 0.90 | 0.93 | 0.88 |
| Gradient boosting (LightGBM) | 0.87** | 1.03 | 0.99 | 0.96 | 0.96 |
| State + forward loss, partial horizons (the paper's design) | 1.07 | 0.92 | 0.81** | 0.97 | 0.94 |
| State + forward loss, complete paths only | 1.00 | 0.92 | 0.82* | 1.09 | 0.96 |
| State + one-month loss | 1.01 | 1.09 | 0.82** | 0.98 | 0.98 |
| No state (theta = 0) + forward loss, partial horizons | 0.93 | 0.95 | 0.93 | 0.88 | 0.92 |

## Persistent-state model, 3 months ahead

Stars: DM test against the random walk.

| model | US | UK | Japan | Euro area | mean |
|---|---|---|---|---|---|
| Random walk (no change) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| UCSV (Stock-Watson) | 0.75*** | 0.81*** | 0.79*** | 0.83** | 0.80 |
| AR-type, own inflation only | 0.80*** | 0.81*** | 0.79*** | 0.83** | 0.81 |
| State + forward loss, partial horizons (the paper's design) | 0.82** | 0.78*** | 0.76*** | 0.90 | 0.81 |
| State + forward loss, complete paths only | 0.80** | 0.79*** | 0.76*** | 0.98 | 0.83 |
| State + one-month loss | 0.80** | 0.83** | 0.75*** | 0.82*** | 0.80 |
| No state (theta = 0) + forward loss, partial horizons | 0.71*** | 0.79*** | 0.78*** | 0.85** | 0.78 |

## Persistent-state model, 6 months ahead

Stars: DM test against the random walk.

| model | US | UK | Japan | Euro area | mean |
|---|---|---|---|---|---|
| Random walk (no change) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| UCSV (Stock-Watson) | 0.81** | 0.81*** | 0.79*** | 0.85 | 0.81 |
| AR-type, own inflation only | 0.88** | 0.81** | 0.81** | 0.87 | 0.85 |
| State + forward loss, partial horizons (the paper's design) | 0.94 | 0.77*** | 0.74*** | 0.87 | 0.83 |
| State + forward loss, complete paths only | 0.90 | 0.77*** | 0.75*** | 0.97 | 0.85 |
| State + one-month loss | 0.90 | 0.85 | 0.73*** | 0.82** | 0.82 |
| No state (theta = 0) + forward loss, partial horizons | 0.83** | 0.78** | 0.78*** | 0.82* | 0.80 |

## Persistent-state model, 24 months ahead

Stars: DM test against the random walk.

| model | US | UK | Japan | Euro area | mean |
|---|---|---|---|---|---|
| Random walk (no change) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| UCSV (Stock-Watson) | 0.84** | 0.89 | 0.82* | 0.87* | 0.86 |
| AR-type, own inflation only | 0.89 | 0.94 | 0.99 | 0.75 | 0.89 |
| State + forward loss, partial horizons (the paper's design) | 0.97 | 0.91 | 0.81* | 0.87 | 0.89 |
| State + forward loss, complete paths only | 0.90 | 0.91 | 0.82 | 0.97 | 0.90 |
| State + one-month loss | 0.97 | 1.19 | 0.88 | 0.98 | 1.01 |
| No state (theta = 0) + forward loss, partial horizons | 0.89 | 0.97 | 1.05 | 0.73* | 0.91 |

## Which ingredient does the work (12 months ahead)

RMSE of the paper's design relative to the variant without one ingredient (below 1: the ingredient helps), with the DM p-value.

| ingredient | US | UK | Japan | Euro area |
|---|---|---|---|---|
| partial horizons (vs complete paths only) | 1.07 (p 0.39) | 1.00 (p 0.36) | 0.98 (p 0.39) | 0.89 (p 0.30) |
| persistent state (vs theta = 0) | 1.15 (p 0.10) | 0.97 (p 0.27) | 0.87 (p 0.11) | 1.11 (p 0.07) |
| forward loss (vs one-month loss) | 1.05 (p 0.42) | 0.84 (p 0.01) | 0.98 (p 0.68) | 1.00 (p 0.96) |

## Robustness: state models trained from 1985

The repository's robustness suite also trains from 1985 (`robustness.py`).  Run after the main results, as a diagnostic of the long training sample.  Same origins; RMSE relative to the random walk.

| model | US | UK | Japan | Euro area | mean |
|---|---|---|---|---|---|
| UCSV (Stock-Watson) | 0.89 | 0.94 | 0.85 | 1.00 | 0.92 |
| Median of all models | 0.88 | 0.90 | 0.84 | 0.90 | 0.88 |
| State + forward loss, partial horizons (the paper's design), trained from 1960 | 1.07 | 0.92 | 0.81 | 0.97 | 0.94 |
| State + forward loss, partial horizons (the paper's design), trained from 1985 | 0.92 | 0.94 | 0.90 | 0.97 | 0.93 |
| State + forward loss, complete paths only, trained from 1960 | 1.00 | 0.92 | 0.82 | 1.09 | 0.96 |
| State + forward loss, complete paths only, trained from 1985 | 0.95 | 0.99 | 0.88 | 1.09 | 0.98 |
| State + one-month loss, trained from 1960 | 1.01 | 1.09 | 0.82 | 0.98 | 0.98 |
| State + one-month loss, trained from 1985 | 0.93 | 0.93 | 0.81 | 0.98 | 0.91 |
| No state (theta = 0) + forward loss, partial horizons, trained from 1960 | 0.93 | 0.95 | 0.93 | 0.88 | 0.92 |
| No state (theta = 0) + forward loss, partial horizons, trained from 1985 | 0.85 | 0.86 | 0.77 | 0.88 | 0.84 |

## Adding the state model to the median

| combination | US | US, 2021-23 origins | UK | Japan | Euro area |
|---|---|---|---|---|---|
| Median of all models (as is) | 0.88 | 0.81 | 0.90 | 0.84 | 0.90 |
| Median including the state model | 0.88 | 0.80 | 0.90* | 0.83* | 0.90 |
| 1/2 median + 1/2 state model | 0.94 | 0.78 | 0.88 | 0.79 | 0.91 |
| State model alone | 1.07 | 0.75 | 0.92 | 0.81 | 0.97 |

## By the month being forecast, 12 months ahead

| economy | model | 2000-07 | 2008-20 | 2021-22 surge | 2023- |
|---|---|---|---|---|---|
| US | UCSV (Stock-Watson) | 0.83 | 0.76 | 1.12 | 0.82 |
| US | AR-type, own inflation only | 1.15 | 1.07 | 0.89 | 0.77 |
| US | Median of all models | 0.87 | 0.92 | 0.94 | 0.61 |
| US | State + forward loss, partial horizons (the paper's design) | 1.38 | 1.20 | 0.88 | 0.67 |
| US | State + one-month loss | 1.32 | 1.13 | 0.83 | 0.67 |
| US | State + forward loss, complete paths only | 1.50 | 1.03 | 0.88 | 0.65 |
| US | No state (theta = 0) + forward loss, partial horizons | 1.05 | 1.08 | 0.79 | 0.55 |
| US | AR-type, own inflation only (horizons 1-12 + partial origins) | 1.21 | 1.10 | 0.88 | 0.77 |
| US | AR-type, own inflation only (as is) | 1.15 | 1.07 | 0.89 | 0.77 |
| US | Elastic net (horizons 1-12 + partial origins) | 1.12 | 1.08 | 1.13 | 0.80 |
| US | Elastic net (as is) | 1.07 | 1.06 | 1.00 | 0.47 |
| US | Gradient boosting (LightGBM) (horizons 1-12 + partial origins) | 0.97 | 0.91 | 0.96 | 0.94 |
| US | Gradient boosting (LightGBM) (as is) | 0.87 | 0.83 | 0.90 | 0.88 |
| US | Random forest (horizons 1-12 + partial origins) | 1.02 | 1.02 | 1.06 | 0.95 |
| US | Random forest (as is) | 0.95 | 0.89 | 0.97 | 0.91 |
| US | Ridge (horizons 1-12 + partial origins) | 0.99 | 1.07 | 0.98 | 0.34 |
| US | Ridge (as is) | 0.96 | 1.06 | 1.03 | 0.36 |
| UK | UCSV (Stock-Watson) | 1.08 | 0.93 | 1.02 | 0.81 |
| UK | AR-type, own inflation only | 1.83 | 1.24 | 0.84 | 0.83 |
| UK | Median of all models | 1.80 | 1.03 | 0.89 | 0.71 |
| UK | State + forward loss, partial horizons (the paper's design) | 1.82 | 1.05 | 0.88 | 0.76 |
| UK | State + forward loss, complete paths only | 1.86 | 1.05 | 0.89 | 0.76 |
| UK | State + one-month loss | 3.69 | 1.29 | 0.78 | 0.88 |
| UK | No state (theta = 0) + forward loss, partial horizons | 2.00 | 1.16 | 0.82 | 0.82 |
| UK | AR-type, own inflation only (horizons 1-12 + partial origins) | 1.99 | 1.19 | 0.83 | 0.85 |
| UK | AR-type, own inflation only (as is) | 1.83 | 1.24 | 0.84 | 0.83 |
| UK | Elastic net (horizons 1-12 + partial origins) | 2.51 | 1.01 | 0.95 | 0.78 |
| UK | Elastic net (as is) | 2.62 | 1.14 | 0.92 | 0.77 |
| UK | Gradient boosting (LightGBM) (horizons 1-12 + partial origins) | 1.89 | 1.09 | 1.02 | 0.85 |
| UK | Gradient boosting (LightGBM) (as is) | 2.47 | 1.02 | 1.02 | 0.88 |
| UK | Random forest (horizons 1-12 + partial origins) | 1.29 | 0.97 | 0.97 | 1.08 |
| UK | Random forest (as is) | 1.48 | 1.05 | 0.95 | 0.95 |
| UK | Ridge (horizons 1-12 + partial origins) | 2.38 | 0.99 | 0.85 | 0.77 |
| UK | Ridge (as is) | 2.01 | 0.98 | 0.90 | 0.67 |
| Japan | UCSV (Stock-Watson) | 0.92 | 0.78 | 0.85 | 1.11 |
| Japan | AR-type, own inflation only | 1.88 | 0.88 | 0.73 | 0.87 |
| Japan | Median of all models | 1.08 | 0.74 | 0.81 | 1.13 |
| Japan | State + forward loss, partial horizons (the paper's design) | 1.01 | 0.73 | 0.87 | 0.98 |
| Japan | State + one-month loss | 1.12 | 0.78 | 0.83 | 0.85 |
| Japan | State + forward loss, complete paths only | 0.98 | 0.74 | 0.87 | 1.06 |
| Japan | No state (theta = 0) + forward loss, partial horizons | 1.73 | 0.88 | 0.77 | 0.86 |
| Japan | AR-type, own inflation only (horizons 1-12 + partial origins) | 1.94 | 0.91 | 0.72 | 0.87 |
| Japan | AR-type, own inflation only (as is) | 1.88 | 0.88 | 0.73 | 0.87 |
| Japan | Elastic net (horizons 1-12 + partial origins) | 1.56 | 0.96 | 0.83 | 0.99 |
| Japan | Elastic net (as is) | 1.61 | 0.93 | 0.96 | 1.06 |
| Japan | Gradient boosting (LightGBM) (horizons 1-12 + partial origins) | 0.95 | 0.96 | 0.77 | 1.30 |
| Japan | Gradient boosting (LightGBM) (as is) | 1.16 | 0.86 | 0.84 | 1.57 |
| Japan | Random forest (horizons 1-12 + partial origins) | 0.80 | 0.92 | 0.78 | 1.27 |
| Japan | Random forest (as is) | 0.87 | 0.84 | 0.86 | 1.32 |
| Japan | Ridge (horizons 1-12 + partial origins) | 1.22 | 0.80 | 0.80 | 0.96 |
| Japan | Ridge (as is) | 1.32 | 0.80 | 0.87 | 1.11 |
| Euro area | UCSV (Stock-Watson) | 0.81 | 0.90 | 1.14 | 0.86 |
| Euro area | AR-type, own inflation only | 0.81 | 0.99 | 1.07 | 0.74 |
| Euro area | Median of all models | 0.94 | 0.99 | 1.03 | 0.60 |
| Euro area | State + forward loss, partial horizons (the paper's design) | 1.40 | 0.98 | 1.06 | 0.84 |
| Euro area | State + one-month loss | 1.49 | 1.02 | 1.00 | 0.91 |
| Euro area | State + forward loss, complete paths only | 1.60 | 1.00 | 1.09 | 1.14 |
| Euro area | No state (theta = 0) + forward loss, partial horizons | 1.16 | 0.91 | 1.05 | 0.58 |
| Euro area | AR-type, own inflation only (horizons 1-12 + partial origins) | 0.85 | 1.02 | 1.09 | 0.79 |
| Euro area | AR-type, own inflation only (as is) | 0.81 | 0.99 | 1.07 | 0.74 |
| Euro area | Elastic net (horizons 1-12 + partial origins) | 0.98 | 1.11 | 1.03 | 0.65 |
| Euro area | Elastic net (as is) | 1.05 | 1.03 | 1.00 | 0.93 |
| Euro area | Gradient boosting (LightGBM) (horizons 1-12 + partial origins) | 0.95 | 1.13 | 1.17 | 0.73 |
| Euro area | Gradient boosting (LightGBM) (as is) | 0.81 | 1.05 | 1.03 | 0.79 |
| Euro area | Random forest (horizons 1-12 + partial origins) | 0.92 | 1.06 | 1.18 | 0.73 |
| Euro area | Random forest (as is) | 0.81 | 1.04 | 1.07 | 0.82 |
| Euro area | Ridge (horizons 1-12 + partial origins) | 0.91 | 1.00 | 1.09 | 0.87 |
| Euro area | Ridge (as is) | 1.06 | 1.04 | 1.02 | 0.90 |

## The paper's test window (US)

Origins January 2021 to December 2023, whose latest print runs from December 2020 to November 2023, as in the paper's section 7.  The y/y rate 12 months after the latest print is the paper's "average inflation over the next twelve months".  Rise and fall use the paper's 12 quarterly SPF dates (latest print January, April, July, October; rise = 2021Q1-2022Q2); bias is outcome minus forecast.  The paper's own numbers (its Table 6) are listed for reference.

| model | rel. RW (36 origins) | RMSE (36) | rise bias / RMSE | fall bias / RMSE | RMSE (12 SPF dates) |
|---|---|---|---|---|---|
| Hemisphere neural net | 0.86 | 2.87 | +3.15 / 3.68 | -1.25 / 1.39 | 2.79 |
| Plain neural net (MLP) | 0.83 | 2.77 | +2.89 / 3.32 | -1.74 / 1.81 | 2.68 |
| Gradient boosting (LightGBM) | 0.90 | 3.01 | +2.06 / 3.33 | -2.14 / 2.65 | 3.01 |
| Random walk (no change) | 1.00 | 3.33 | +1.66 / 3.42 | -2.48 / 3.07 | 3.25 |
| UCSV (Stock-Watson) | 1.07 | 3.57 | +3.08 / 4.28 | -1.75 / 2.12 | 3.38 |
| AR-type, own inflation only | 0.83 | 2.76 | +1.66 / 3.02 | -1.70 / 2.30 | 2.68 |
| Ridge | 0.69 | 2.31 | +2.33 / 2.98 | -1.02 / 1.12 | 2.25 |
| Elastic net | 0.64 | 2.15 | +1.44 / 2.55 | -1.31 / 1.41 | 2.06 |
| Median of all models | 0.81 | 2.71 | +2.20 / 3.16 | -1.70 / 1.90 | 2.61 |
| State + forward loss, partial horizons (the paper's design) | 0.75 | 2.51 | +1.83 / 2.75 | -1.40 / 1.83 | 2.34 |
| State + one-month loss | 0.72 | 2.40 | +1.64 / 2.53 | -1.03 / 1.65 | 2.14 |
| State + forward loss, complete paths only | 0.76 | 2.53 | +1.95 / 2.82 | -1.43 / 1.82 | 2.37 |
| No state (theta = 0) + forward loss, partial horizons | 0.64 | 2.14 | +1.69 / 2.37 | -1.18 / 1.42 | 1.95 |
| Random forest (horizons 1-12, complete origins) | 1.02 | 3.40 | +2.58 / 3.86 | -2.24 / 2.78 | 3.37 |
| Gradient boosting (LightGBM) (horizons 1-12, complete origins) | 0.92 | 3.06 | +1.96 / 3.38 | -2.02 / 2.63 | 3.03 |
| AR-type, own inflation only (horizons 1-12, complete origins) | 0.81 | 2.71 | +1.56 / 2.96 | -1.62 / 2.25 | 2.62 |
| Ridge (horizons 1-12, complete origins) | 0.74 | 2.45 | +2.61 / 3.22 | -0.82 / 0.98 | 2.38 |
| Elastic net (horizons 1-12, complete origins) | 0.68 | 2.25 | +2.25 / 3.02 | -0.55 / 0.77 | 2.20 |
| Random forest (horizons 1-12 + partial origins) | 1.04 | 3.46 | +2.04 / 3.97 | -1.82 / 2.63 | 3.37 |
| AR-type, own inflation only (horizons 1-12 + partial origins) | 0.81 | 2.71 | +1.57 / 2.96 | -1.63 / 2.25 | 2.63 |
| Gradient boosting (LightGBM) (horizons 1-12 + partial origins) | 1.00 | 3.34 | +2.05 / 3.79 | -2.02 / 2.70 | 3.29 |
| Ridge (horizons 1-12 + partial origins) | 0.61 | 2.04 | +1.82 / 2.49 | -0.84 / 0.97 | 1.89 |
| Elastic net (horizons 1-12 + partial origins) | 0.93 | 3.11 | +1.99 / 3.47 | -1.51 / 2.48 | 3.01 |
| SPF median (paper) |  |  | +4.53 / 4.86 | -0.06 / 0.33 | 3.44 |
| Deep-pi*, forward-looking, encoder (paper) |  |  | -0.04 / 2.25 | -1.23 / 2.25 | 2.25 |
| Deep-pi*, forward-looking, attention (paper) |  |  | +0.45 / 2.15 | -1.15 / 2.13 | 2.14 |
| AR(12) (paper) |  |  | +1.46 / 2.47 | -1.33 / 1.69 | 2.12 |
| UC-SV (paper) |  |  | +1.19 / 2.82 | -1.93 / 2.53 | 2.68 |
| Random walk (paper) |  |  | +1.66 / 3.46 | -2.46 / 3.02 | 3.25 |

Real-time forecasts of y/y inflation 12 months after the print shown:

| latest print | random walk | UCSV | state, complete | state, partial | no state, partial | ridge | ridge, partial | your median | outcome |
|---|---|---|---|---|---|---|---|---|---|
| 2021-01 print | 1.4 | 1.8 | 2.9 | 2.9 | 4.3 | 2.3 | 4.0 | 2.2 | 7.2 |
| 2021-04 print | 4.1 | 2.2 | 4.2 | 4.3 | 4.4 | 4.8 | 4.5 | 4.1 | 7.9 |
| 2021-07 print | 5.2 | 2.9 | 5.0 | 5.1 | 5.3 | 5.2 | 5.6 | 4.9 | 8.2 |
| 2021-10 print | 6.0 | 3.4 | 5.8 | 6.0 | 5.7 | 4.6 | 4.8 | 4.7 | 7.5 |
| 2022-01 print | 7.2 | 6.1 | 6.1 | 6.3 | 6.1 | 5.3 | 6.2 | 6.3 | 6.2 |
| 2022-04 print | 7.9 | 6.9 | 6.1 | 6.3 | 5.8 | 5.8 | 5.8 | 6.4 | 4.8 |
| 2022-07 print | 8.2 | 7.3 | 6.5 | 6.7 | 5.4 | 4.1 | 4.1 | 5.8 | 3.1 |
| 2022-10 print | 7.5 | 5.5 | 5.5 | 5.5 | 4.9 | 4.9 | 4.7 | 6.0 | 3.2 |

## What moved the US trend in real time

Change of the trend (pp, annualised) since January 2019 by predictor group, from the fit at each origin (the trend is linear, so the split is exact).

| origin | trend level | initial state and alpha | cost | expect | global | money | own (core, components) | slack | total change |
|---|---|---|---|---|---|---|---|---|---|
| 2020-06 | 2.41 | -0.01 | 0.15 | 0.11 | -0.01 | -0.0 | 0.08 | 0.04 | 0.36 |
| 2021-01 | 2.72 | -0.01 | 0.13 | 0.02 | -0.0 | 0.03 | 0.05 | 0.06 | 0.27 |
| 2021-03 | 2.84 | -0.01 | 0.16 | -0.01 | 0.0 | 0.04 | 0.04 | 0.06 | 0.28 |
| 2021-06 | 3.1 | -0.01 | 0.19 | -0.02 | 0.01 | 0.05 | -0.0 | 0.08 | 0.29 |
| 2021-12 | 3.6 | -0.01 | 0.14 | -0.04 | 0.02 | 0.07 | 0.12 | 0.07 | 0.38 |
| 2022-06 | 4.04 | -0.01 | 0.03 | -0.07 | 0.0 | 0.1 | 0.4 | 0.01 | 0.46 |
| 2022-12 | 3.7 | -0.01 | -0.1 | -0.08 | -0.02 | 0.12 | 0.43 | -0.04 | 0.29 |
| 2023-06 | 3.22 | -0.01 | -0.14 | -0.1 | -0.03 | 0.12 | 0.3 | -0.07 | 0.08 |

