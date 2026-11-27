# Pre-registered test: results

Plan: [`docs/inflation_prereg_1985.md`](inflation_prereg_1985.md), committed before the median was trained from 1985.  A = the no-memory trend model, B = the median of the 13 tier-1 country models, both trained from 1985.

## Primary comparison (12 months, four economies pooled)

* Diebold-Mariano statistic 0.68 (positive favours A), two-sided p = 0.499, 311 origins.
* Mean RMSE relative to the random walk: A 0.841, B 0.866.
* **Verdict under the pre-registered rule (DM > 0 and p < 0.05): A does not beat B.**

## Secondary: per economy, 12 months

| economy | origins | A rel. RW | B rel. RW | A / B | DM (+ favours A) | p | Holm p |
|---|---|---|---|---|---|---|---|
| US | 309 | 0.85 | 0.82 | 1.03 | -0.52 | 0.604 | 1.000 |
| UK | 311 | 0.86 | 0.86 | 1.00 | -0.07 | 0.947 | 1.000 |
| Japan | 311 | 0.77 | 0.89 | 0.87 | 1.06 | 0.291 | 1.000 |
| Euro area | 256 | 0.88 | 0.90 | 0.98 | 0.57 | 0.569 | 1.000 |

## Secondary: pooled test at every horizon

| horizon | origins | mean rel. RW, A | mean rel. RW, B | DM (+ favours A) | p |
|---|---|---|---|---|---|
| 3 | 320 | 0.74 | 0.74 | 0.19 | 0.851 |
| 6 | 317 | 0.76 | 0.81 | 1.47 | 0.141 |
| 12 | 311 | 0.84 | 0.87 | 0.68 | 0.499 |
| 24 | 299 | 0.78 | 0.84 | 1.85 | 0.064 |

## Secondary: by the month being forecast, 12 months ahead (RMSE relative to the random walk)

| economy | arm | 2000-07 | 2008-20 | 2021-22 surge | 2023- |
|---|---|---|---|---|---|
| US | A | 0.81 | 0.89 | 0.99 | 0.33 |
| US | B | 0.84 | 0.78 | 1.04 | 0.46 |
| UK | A | 0.98 | 0.96 | 0.93 | 0.68 |
| UK | B | 1.00 | 0.94 | 0.95 | 0.64 |
| Japan | A | 0.93 | 0.66 | 0.80 | 1.10 |
| Japan | B | 1.01 | 0.61 | 0.95 | 1.61 |
| Euro area | A | 1.14 | 0.91 | 1.05 | 0.58 |
| Euro area | B | 0.90 | 0.99 | 1.03 | 0.60 |

US origins 2021-01 to 2023-12 (the paper's window): A 0.85, B 0.87 relative to the random walk (36 origins).

## Reference: the median trained from 1960

What the training window alone does to the median (same origins; RMSE relative to the random walk).

| economy | origins | median from 1960 | median from 1985 (B) | A |
|---|---|---|---|---|
| US | 309 | 0.88 | 0.82 | 0.85 |
| UK | 311 | 0.90 | 0.86 | 0.86 |
| Japan | 311 | 0.84 | 0.89 | 0.77 |
| Euro area | 256 | 0.90 | 0.90 | 0.88 |

## Forward record

180 forecasts whose target print is after 2026-10 (`results/inflation/prereg_1985/forward.csv`), by economy and horizon:

| economy | 3 months | 6 months | 12 months | 24 months |
|---|---|---|---|---|
| EA | 3 | 6 | 12 | 24 |
| JP | 3 | 6 | 12 | 24 |
| UK | 3 | 6 | 12 | 24 |
| US | 3 | 6 | 12 | 24 |

Score them after a new `build` with `python -m bondpredictor.inflation prereg forward`.

