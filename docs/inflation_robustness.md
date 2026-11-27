# Robustness of the econometric ML models (October 2026 configuration)

Re-estimation variants (seeds, fixed hyperparameters, rolling windows, post-1985 sample, target offsets) run on the October 2026 configuration, before the 24-month core anchor and the US underlying-inflation measures.  They have not been rerun on the current configuration; the main report's tables are in `inflation_report.md`.

## Robustness of the econometric ML models, 12 months ahead

Each variant re-runs one model with one change, scored with its baseline on the same months (RMSE relative to the random walk).

### United States

| variant | baseline | variant | corr. with baseline |
|---|---|---|---|
| bart \| 100 trees | 0.89 | 0.82 | 0.53 |
| bart \| k=3 (more shrinkage) | 0.89 | 0.81 | 0.48 |
| bart \| offset mech | 0.89 | 0.99 | 0.53 |
| bart \| rolling 20y | 0.89 | 0.99 | 0.38 |
| bart \| seed 1 | 0.89 | 0.92 | 0.54 |
| bart \| seed 2 | 0.89 | 0.98 | 0.59 |
| bart \| train from 1985 | 0.89 | 0.93 | 0.50 |
| hnn \| fixed width=16,depth=2,dropout=0.2,weight_decay=0.001 | 0.88 | 0.89 | 0.98 |
| hnn \| fixed width=32,depth=2,dropout=0.0,weight_decay=0.001 | 0.88 | 0.91 | 0.97 |
| hnn \| fixed width=32,depth=2,dropout=0.4,weight_decay=0.01 | 0.88 | 0.89 | 0.98 |
| hnn \| fixed width=64,depth=2,dropout=0.2,weight_decay=0.001 | 0.88 | 0.87 | 0.99 |
| hnn \| rolling 20y | 0.88 | 0.90 | 0.80 |
| hnn \| seed 1 | 0.88 | 0.90 | 0.98 |
| hnn \| seed 2 | 0.88 | 0.85 | 0.98 |
| hnn \| train from 1985 | 0.88 | 0.83 | 0.90 |
| krr \| fixed alpha=1.0,gamma=0.3 | 0.94 | 0.90 | 0.99 |
| lgbm \| fixed n_estimators=300,num_leaves=4 | 0.85 | 0.86 | 0.99 |
| lgbm \| fixed n_estimators=300,num_leaves=8 | 0.85 | 0.87 | 0.98 |
| lgbm \| fixed n_estimators=800,num_leaves=4 | 0.85 | 0.81 | 0.99 |
| lgbm \| fixed n_estimators=800,num_leaves=8 | 0.85 | 0.85 | 0.98 |
| lgbm \| offset mech | 0.85 | 0.90 | 0.85 |
| lgbm \| rolling 20y | 0.85 | 0.88 | 0.74 |
| lgbm \| seed 1 | 0.85 | 0.86 | 1.00 |
| lgbm \| seed 2 | 0.85 | 0.85 | 1.00 |
| lgbm \| train from 1985 | 0.85 | 0.84 | 0.85 |
| mrf \| fixed min_leaf=20,lam=0.1 | 0.91 | 0.93 | 0.99 |
| mrf \| fixed min_leaf=40,lam=0.1 | 0.91 | 0.92 | 0.99 |
| mrf \| no time smoothing | 0.91 | 0.92 | 0.99 |
| mrf \| rolling 20y | 0.91 | 0.89 | 0.10 |
| mrf \| seed 1 | 0.91 | 0.91 | 0.97 |
| mrf \| seed 2 | 0.91 | 0.92 | 0.98 |
| mrf \| train from 1985 | 0.91 | 0.85 | 0.78 |
| rf \| fixed min_samples_leaf=15 | 0.93 | 0.95 | 0.99 |
| rf \| offset mech | 0.93 | 0.95 | 0.81 |
| rf \| rolling 20y | 0.93 | 0.88 | 0.84 |
| rf \| seed 1 | 0.93 | 0.94 | 1.00 |
| rf \| seed 2 | 0.93 | 0.94 | 1.00 |
| rf \| train from 1985 | 0.93 | 0.87 | 0.91 |
| ridge \| rolling 20y | 1.00 | 0.98 | 0.52 |
| ridge \| train from 1985 | 1.00 | 0.92 | 0.75 |

### United Kingdom

| variant | baseline | variant | corr. with baseline |
|---|---|---|---|
| bart \| 100 trees | 1.07 | 1.05 | 0.60 |
| bart \| k=3 (more shrinkage) | 1.07 | 1.19 | 0.75 |
| bart \| offset mech | 1.07 | 1.09 | 0.60 |
| bart \| rolling 20y | 1.07 | 1.05 | 0.63 |
| bart \| seed 1 | 1.07 | 1.27 | 0.70 |
| bart \| seed 2 | 1.07 | 1.14 | 0.75 |
| bart \| train from 1985 | 1.07 | 0.99 | 0.67 |
| hnn \| fixed width=16,depth=2,dropout=0.2,weight_decay=0.001 | 0.97 | 0.93 | 0.84 |
| hnn \| fixed width=32,depth=2,dropout=0.0,weight_decay=0.001 | 0.97 | 0.87 | 0.93 |
| hnn \| fixed width=32,depth=2,dropout=0.4,weight_decay=0.01 | 0.97 | 0.94 | 0.92 |
| hnn \| fixed width=64,depth=2,dropout=0.2,weight_decay=0.001 | 0.97 | 0.90 | 0.86 |
| hnn \| rolling 20y | 0.97 | 0.89 | 0.89 |
| hnn \| seed 1 | 0.97 | 0.87 | 0.90 |
| hnn \| seed 2 | 0.97 | 0.97 | 0.91 |
| hnn \| train from 1985 | 0.97 | 0.90 | 0.82 |
| krr \| fixed alpha=1.0,gamma=0.3 | 0.89 | 0.89 | 0.95 |
| lgbm \| fixed n_estimators=300,num_leaves=4 | 1.07 | 1.06 | 0.99 |
| lgbm \| fixed n_estimators=300,num_leaves=8 | 1.07 | 1.01 | 0.98 |
| lgbm \| fixed n_estimators=800,num_leaves=4 | 1.07 | 1.05 | 0.98 |
| lgbm \| fixed n_estimators=800,num_leaves=8 | 1.07 | 1.02 | 0.98 |
| lgbm \| offset mech | 1.07 | 0.99 | 0.94 |
| lgbm \| rolling 20y | 1.07 | 0.97 | 0.87 |
| lgbm \| seed 1 | 1.07 | 1.03 | 0.98 |
| lgbm \| seed 2 | 1.07 | 1.05 | 0.99 |
| lgbm \| train from 1985 | 1.07 | 0.95 | 0.84 |
| mrf \| fixed min_leaf=20,lam=0.1 | 0.96 | 0.97 | 0.98 |
| mrf \| fixed min_leaf=40,lam=0.1 | 0.96 | 0.96 | 1.00 |
| mrf \| no time smoothing | 0.96 | 0.96 | 1.00 |
| mrf \| rolling 20y | 0.96 | 0.97 | 0.62 |
| mrf \| seed 1 | 0.96 | 0.90 | 0.98 |
| mrf \| seed 2 | 0.96 | 0.94 | 0.98 |
| mrf \| train from 1985 | 0.96 | 0.90 | 0.91 |
| rf \| fixed min_samples_leaf=15 | 0.98 | 1.00 | 0.99 |
| rf \| offset mech | 0.98 | 0.90 | 0.95 |
| rf \| rolling 20y | 0.98 | 0.95 | 0.95 |
| rf \| seed 1 | 0.98 | 0.98 | 1.00 |
| rf \| seed 2 | 0.98 | 0.96 | 1.00 |
| rf \| train from 1985 | 0.98 | 0.93 | 0.95 |
| ridge \| rolling 20y | 0.89 | 0.88 | 0.69 |
| ridge \| train from 1985 | 0.89 | 0.84 | 0.77 |

### Japan

| variant | baseline | variant | corr. with baseline |
|---|---|---|---|
| bart \| 100 trees | 1.01 | 1.08 | 0.54 |
| bart \| k=3 (more shrinkage) | 1.01 | 1.01 | 0.66 |
| bart \| offset mech | 1.01 | 1.13 | 0.65 |
| bart \| rolling 20y | 1.01 | 1.11 | 0.53 |
| bart \| seed 1 | 1.01 | 1.21 | 0.49 |
| bart \| seed 2 | 1.01 | 1.09 | 0.51 |
| bart \| train from 1985 | 1.01 | 0.99 | 0.54 |
| hnn \| fixed width=16,depth=2,dropout=0.2,weight_decay=0.001 | 1.10 | 1.01 | 0.75 |
| hnn \| fixed width=32,depth=2,dropout=0.0,weight_decay=0.001 | 1.10 | 0.98 | 0.89 |
| hnn \| fixed width=32,depth=2,dropout=0.4,weight_decay=0.01 | 1.10 | 1.08 | 0.82 |
| hnn \| fixed width=64,depth=2,dropout=0.2,weight_decay=0.001 | 1.10 | 1.58 | 0.11 |
| hnn \| rolling 20y | 1.10 | 1.04 | 0.68 |
| hnn \| seed 1 | 1.10 | 1.30 | 0.49 |
| hnn \| seed 2 | 1.10 | 1.04 | 0.75 |
| hnn \| train from 1985 | 1.10 | 1.07 | 0.52 |
| krr \| fixed alpha=1.0,gamma=0.3 | 0.87 | 0.89 | 0.98 |
| lgbm \| fixed n_estimators=300,num_leaves=4 | 0.98 | 0.99 | 0.97 |
| lgbm \| fixed n_estimators=300,num_leaves=8 | 0.98 | 1.02 | 0.96 |
| lgbm \| fixed n_estimators=800,num_leaves=4 | 0.98 | 0.96 | 0.98 |
| lgbm \| fixed n_estimators=800,num_leaves=8 | 0.98 | 1.02 | 0.95 |
| lgbm \| offset mech | 0.98 | 1.03 | 0.92 |
| lgbm \| rolling 20y | 0.98 | 1.05 | 0.64 |
| lgbm \| seed 1 | 0.98 | 1.02 | 0.98 |
| lgbm \| seed 2 | 0.98 | 0.97 | 0.98 |
| lgbm \| train from 1985 | 0.98 | 0.98 | 0.65 |
| mrf \| fixed min_leaf=20,lam=0.1 | 0.97 | 0.95 | 0.99 |
| mrf \| fixed min_leaf=40,lam=0.1 | 0.97 | 0.93 | 0.98 |
| mrf \| no time smoothing | 0.97 | 0.97 | 0.99 |
| mrf \| rolling 20y | 0.97 | 0.94 | 0.59 |
| mrf \| seed 1 | 0.97 | 0.91 | 0.96 |
| mrf \| seed 2 | 0.97 | 0.90 | 0.96 |
| mrf \| train from 1985 | 0.97 | 0.89 | 0.73 |
| rf \| fixed min_samples_leaf=15 | 0.92 | 0.95 | 0.96 |
| rf \| offset mech | 0.92 | 0.97 | 0.86 |
| rf \| rolling 20y | 0.92 | 1.04 | 0.81 |
| rf \| seed 1 | 0.92 | 0.92 | 1.00 |
| rf \| seed 2 | 0.92 | 0.92 | 0.99 |
| rf \| train from 1985 | 0.92 | 0.97 | 0.87 |
| ridge \| rolling 20y | 0.89 | 1.09 | 0.68 |
| ridge \| train from 1985 | 0.89 | 0.95 | 0.78 |

### Euro area

| variant | baseline | variant | corr. with baseline |
|---|---|---|---|
| bart \| 100 trees | 1.15 | 1.05 | 0.85 |
| bart \| k=3 (more shrinkage) | 1.15 | 1.00 | 0.80 |
| bart \| offset mech | 1.15 | 1.01 | 0.56 |
| bart \| rolling 20y | 1.15 | 1.07 | 0.78 |
| bart \| seed 1 | 1.15 | 0.95 | 0.67 |
| bart \| seed 2 | 1.15 | 1.14 | 0.88 |
| bart \| train from 1985 | 1.15 | 1.15 | 1.00 |
| hnn \| fixed width=16,depth=2,dropout=0.2,weight_decay=0.001 | 0.92 | 0.93 | 0.98 |
| hnn \| fixed width=32,depth=2,dropout=0.0,weight_decay=0.001 | 0.92 | 0.99 | 0.98 |
| hnn \| fixed width=32,depth=2,dropout=0.4,weight_decay=0.01 | 0.92 | 0.94 | 0.99 |
| hnn \| fixed width=64,depth=2,dropout=0.2,weight_decay=0.001 | 0.92 | 0.92 | 0.98 |
| hnn \| rolling 20y | 0.92 | 0.98 | 0.99 |
| hnn \| seed 1 | 0.92 | 0.94 | 0.97 |
| hnn \| seed 2 | 0.92 | 0.98 | 0.98 |
| hnn \| train from 1985 | 0.92 | 0.92 | 1.00 |
| krr \| fixed alpha=1.0,gamma=0.3 | 0.94 | 0.96 | 0.96 |
| lgbm \| fixed n_estimators=300,num_leaves=4 | 0.96 | 0.96 | 1.00 |
| lgbm \| fixed n_estimators=300,num_leaves=8 | 0.96 | 0.96 | 1.00 |
| lgbm \| fixed n_estimators=800,num_leaves=4 | 0.96 | 0.96 | 0.99 |
| lgbm \| fixed n_estimators=800,num_leaves=8 | 0.96 | 0.96 | 0.99 |
| lgbm \| offset mech | 0.96 | 0.94 | 0.83 |
| lgbm \| rolling 20y | 0.96 | 0.96 | 0.99 |
| lgbm \| seed 1 | 0.96 | 0.97 | 0.99 |
| lgbm \| seed 2 | 0.96 | 0.98 | 0.99 |
| lgbm \| train from 1985 | 0.96 | 0.96 | 1.00 |
| mrf \| fixed min_leaf=20,lam=0.1 | 0.97 | 0.97 | 0.97 |
| mrf \| fixed min_leaf=40,lam=0.1 | 0.97 | 0.96 | 0.99 |
| mrf \| no time smoothing | 0.97 | 0.96 | 0.99 |
| mrf \| rolling 20y | 0.97 | 0.98 | 0.94 |
| mrf \| seed 1 | 0.97 | 0.96 | 0.94 |
| mrf \| seed 2 | 0.97 | 0.98 | 0.89 |
| mrf \| train from 1985 | 0.97 | 0.97 | 1.00 |
| rf \| fixed min_samples_leaf=15 | 0.98 | 0.98 | 0.98 |
| rf \| offset mech | 0.98 | 0.91 | 0.79 |
| rf \| rolling 20y | 0.98 | 0.98 | 0.99 |
| rf \| seed 1 | 0.98 | 0.97 | 1.00 |
| rf \| seed 2 | 0.98 | 0.98 | 1.00 |
| rf \| train from 1985 | 0.98 | 0.98 | 1.00 |
| ridge \| rolling 20y | 0.99 | 0.96 | 0.83 |
| ridge \| train from 1985 | 0.99 | 0.99 | 1.00 |

## Robustness of the econometric ML models, 3 months ahead

Each variant re-runs one model with one change, scored with its baseline on the same months (RMSE relative to the random walk).

### United States

| variant | baseline | variant | corr. with baseline |
|---|---|---|---|
| bart \| offset mech | 0.98 | 0.72 | 0.92 |
| lgbm \| offset mech | 0.77 | 0.66 | 0.97 |
| rf \| offset mech | 0.82 | 0.73 | 0.96 |

### United Kingdom

| variant | baseline | variant | corr. with baseline |
|---|---|---|---|
| bart \| offset mech | 1.08 | 0.91 | 0.94 |
| lgbm \| offset mech | 0.81 | 0.80 | 1.00 |
| rf \| offset mech | 0.77 | 0.74 | 0.99 |

### Japan

| variant | baseline | variant | corr. with baseline |
|---|---|---|---|
| bart \| offset mech | 0.96 | 1.05 | 0.93 |
| lgbm \| offset mech | 0.84 | 0.83 | 0.99 |
| rf \| offset mech | 0.78 | 0.76 | 0.99 |

### Euro area

| variant | baseline | variant | corr. with baseline |
|---|---|---|---|
| bart \| offset mech | 1.06 | 0.91 | 0.96 |
| lgbm \| offset mech | 0.89 | 0.73 | 0.98 |
| rf \| offset mech | 0.90 | 0.73 | 0.97 |

