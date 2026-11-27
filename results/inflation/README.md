# Backtest results

Real-time forecasts of every model at every origin, one long CSV per group
(`economy, horizon, model, date` = forecast origin, then `forecast`, `rw` =
latest y/y rate, `actual` and diagnostics).  Written by
`python -m bondpredictor.inflation results pack`; `results unpack` turns
them back into `output/inflation/*.pkl` for `report` and `site`.

| file | contents |
|---|---|
| `backtest_US.csv.gz`, `_UK`, `_JP`, `_EA` | benchmarks and econometric ML models, origins 2000-01 on (euro area 2002/2003) |
| `backtest_pooled.csv.gz` | the five pooled four-economy models |
| `foundation.csv.gz` | Chronos and TimesFM, zero-shot |
| `deep.csv.gz` | N-HiTS, PatchTST, TiDE, LSTM (when present) |
| `pilot_US.csv.gz`, `_UK`, `_JP`, `_EA` | six fast models with the nowcast, survey-path and bottom-up variants (`model` is `set\|model`; `comp:<name>` are the component forecasts behind `bottom_up`); the combined forecast is built from these |
| `variants/backtest_JP_h24_noanchor.csv.gz` | Japan, 24 months ahead, without the core anchor (for comparison) |

Run on 27 November 2026 with data to October 2026: 24-month forecasts of the
ML models anchored on core inflation, US models with the underlying-inflation
measures, UK target CPI (the UK, pooled, foundation and pilot files were
rerun after the switch from CPIH; `deep.csv.gz` still holds the CPIH run for
the UK until its rerun lands).  `results unpack` writes the pilot files to
`output/inflation/pilot/pilot.pkl`.
