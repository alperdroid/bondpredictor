# Pre-registered test: the no-memory trend model against the median, both trained from 1985

Written and committed on 27 November 2026, **before** the median's members were trained from 1985.  The code
that runs and scores the test (`bondpredictor/inflation/prereg.py`, `python -m bondpredictor.inflation prereg`)
is committed with this file.  Results go to a separate file, `docs/inflation_prereg_1985_results.md`, so this
plan stays as written; its git history shows whether it changed.

## Why

In `docs/inflation_pistar.md` the persistent-state model *without* memory (theta = 0, forward-looking loss with
partial horizons, `pistar_fwd_nostate`), trained from 1985, averaged 0.84 of the random walk's 12-month RMSE,
against 0.88 for the repository's median combination.  But the median was trained from 1960, so the gap mixes
two things: the model and the training window.  The repository's own robustness suite finds that dropping the
1970s slightly improves most models.  This test gives the median the same window.

**What this test cannot fix.**  The challenger was picked after its 2000-2025 results were seen, out of about
ten variants (four state models, two training windows, stacked direct models).  Its in-sample-of-evaluation
numbers are therefore already known and are biased upwards by that selection; a p-value below 0.05 below
overstates the evidence.  The only clean evidence is on outcomes nobody has seen yet, so the test also commits
a **forward record** (below) to be scored as new CPI prints arrive.

## Arms

* **A (challenger):** `pistar_fwd_nostate` exactly as in `pistar.state_zoo()` (theta = 0, forward loss over
  1-12 months with partial horizons, AR(12), ridge penalty grid `pistar.LAMBDAS` tuned every 12 months,
  refitted monthly, multi-started), with `train_start = "1985-01"`.
* **B (benchmark):** the repository's fixed-in-advance median ("Median of all models",
  `evaluation.add_ensembles`, all members present) of the 13 tier-1 country models: ridge, adalasso, enet, di,
  3prf, csr, krr, rf, lgbm, mrf, bart, hnn, mlp.  Each member is `backtest.model_zoo(1)`'s spec with only
  `train_start = "1985-01"` changed, as in `robustness.py`'s "train from 1985" variants (refit cadence,
  hyperparameter grids and the 24-month core anchor as in the zoo).  Pooled panel models are not members.
* For the euro area the data start in 1990, so "from 1985" changes nothing there for either arm.

## Data and sample

* Data build `data/processed/inflation.pkl` of 27 November 2026 (sha256
  `ef6b80137d896f4db07e0ea60ee6a0fa22afe201d497544d81a440e607a926db`), latest print October 2026.
* Forecast origins from 2000-01 to 2026-11 for both arms (`min_train` = 120, as everywhere else).
* **Scored sample:** for each economy, the origins with a published outcome in this build where both arms have
  a forecast.

## Primary comparison (decides the test)

* Horizon: **12 months** (y/y inflation 12 months after the latest print), all four economies pooled.
* Loss differential for economy *c* at origin *t*: `d[c,t] = (e_B[c,t]^2 - e_A[c,t]^2) / s_c`, where `e` is
  outcome minus forecast and `s_c` is the random walk's mean squared error on economy *c*'s scored sample
  (so each economy counts equally, as in the repository's mean of relative RMSEs).
* Pooled series: `d[t]` = mean of `d[c,t]` over the economies scored at *t*.
* Test: Diebold-Mariano on `d[t]`, Newey-West variance with 11 lags, Harvey-Leybourne-Newbold correction with
  h = 12, normal p-value (as in `robustness.diebold_mariano`).
* **Decision rule: A beats B if the statistic is positive and the two-sided p-value is below 0.05.**  Otherwise
  the 1985 lead is attributed to the training window and/or the selection, and the zoo stays as it is.
* Reported with it: each arm's mean (over economies) of RMSE relative to the random walk.

## Secondary results (reported, not decisive)

* Per economy, 12 months: RMSE of A relative to B and the HLN Diebold-Mariano p-value, Holm-adjusted over the
  four economies.
* The same pooled test at 3, 6 and 24 months.
* 12 months by the period being forecast (`evaluation.SUBPERIODS`) and on the US 2021-23 origins.
* Reference only: the median trained from 1960 (from `output/inflation/pistar`), to show what the training
  window alone does to the median.

## Forward record (the clean test)

* `results/inflation/prereg_1985/forward.csv`: both arms' forecasts for every economy, horizon (3, 6, 12, 24)
  and origin whose target print is **later than October 2026**, i.e. not yet published.  The selection above saw
  no such outcome.
* Scoring later: `python -m bondpredictor.inflation prereg forward` after a new `build` compares the recorded
  forecasts (not re-estimated ones) with the new prints, with the same pooled statistic per horizon.  A verdict
  on the forward record should wait until it holds at least 24 monthly 12-month outcomes per economy (about
  November 2028).

## Integrity rules

* No change to model code, specs, grids, data or the scoring code after this commit.  If a bug forces a change,
  both the original and the corrected results will be reported, with the reason.
* All outcomes are reported whichever arm wins.
