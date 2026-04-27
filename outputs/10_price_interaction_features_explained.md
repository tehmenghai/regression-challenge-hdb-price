# 10 — v11: Price Interaction Target Encoding

> Adds three new OOF interaction encodings derived from resale price — `(flat_type × postal_sector)`, `(year × postal_sector)`, and `(year × flat_type)` — plus fixes a `hdb_age` reference-year bug, achieving a Ridge meta-model OOF RMSE of **$21,716** against a v10 Kaggle baseline of ~$21,755.

## Overview

Previous versions encoded geographic price signals at the town or postal-sector level, but treated all flat types within a sector as having the same baseline price. In reality, a 5-room flat in Queenstown sector 1425 sells for ~$845K while a 2-room in the same sector sells for ~$250K — a $595K gap that a single `postal_sector_median_price` cannot capture.

This notebook introduces **three interaction OOF target encodings** that combine resale price with two categorical dimensions simultaneously: flat type × postal sector, transaction year × postal sector, and transaction year × flat type. All encodings are computed out-of-fold to prevent data leakage, and sparse groups fall back to a parent-level encoding rather than the global median.

A secondary fix corrects `hdb_age`, which in the raw data was computed using a hardcoded 2021 reference year. This made it inconsistent with `remaining_lease` (which uses `Tranc_Year`), causing the two features to add noise rather than complementary signal.

## Feature Engineering

The `engineer_features` function builds all new columns in one pass applied identically to train and test:

```python
# FIX: recompute hdb_age using Tranc_Year so hdb_age + remaining_lease = 99
df['remaining_lease'] = 99 - (df['Tranc_Year'] - df['lease_commence_date'])
df['hdb_age']         = df['Tranc_Year'] - df['lease_commence_date']

# Composite keys for interaction OOF encoding
df['ft_sector_key'] = df['flat_type'].astype(str) + '_' + df['postal_sector'].astype(str)
df['yr_sector_key'] = df['Tranc_Year'].astype(str) + '_' + df['postal_sector'].astype(str)
df['yr_ft_key']     = df['Tranc_Year'].astype(str) + '_' + df['flat_type'].astype(str)
```

The composite key columns (`ft_sector_key`, `yr_sector_key`, `yr_ft_key`, `postal_sector`) are dropped from the model's feature matrix `X` after encoding — they are encoding keys, not model inputs.

## Per-Fold OOF Target Encoding

The `oof_group_median` helper computes a group median for each row using only the other folds' data:

```python
def oof_group_median(group_series, y_series, n_splits=5, random_state=42, min_count=1):
    groups, y = group_series.values, y_series.values
    encoded = np.zeros(len(groups))
    global_med = np.median(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, va_idx in kf.split(groups):
        fold_map = {g: np.median(ps) for g, ps in ...
                    if fold_counts[g] >= min_count}
        for i in va_idx:
            encoded[i] = fold_map.get(groups[i], global_med)
    return encoded
```

The six encodings and their fallback strategy:

| Group key | Output column | min_count | Fallback |
|---|---|---|---|
| `town` | `town_median_price` | 1 | global median |
| `postal_sector` | `postal_sector_median_price` | 10 | global median |
| `flat_model` | `flat_model_median_price` | 1 | global median |
| `ft_sector_key` | `ft_sector_median_price` | 10 | `postal_sector_median_price` |
| `yr_sector_key` | `yr_sector_median_price` | 10 | `postal_sector_median_price` |
| `yr_ft_key` | `yr_flattype_median_price` | 1 | global median |

The two interaction features with high sparsity (`ft_sector_key` has ~1,772 groups, `yr_sector_key` has ~5,133 groups) fall back to the **parent postal-sector encoding** rather than global median — a tighter fallback that preserves the geographic price signal for rare groups.

## Hyperparameter Search (Sections 6–8)

All three base models use `RandomizedSearchCV` with `cv=3` and a custom `dollar_rmse_scorer` that evaluates RMSE in SGD rather than log-space:

```python
def neg_dollar_rmse(y_log_true, y_log_pred):
    return -np.sqrt(mean_squared_error(np.expm1(y_log_true), np.expm1(y_log_pred)))
```

Best parameters found:

| Model | Key params | Val RMSE |
|---|---|---|
| XGBoost | n_estimators=2000, max_depth=7, lr=0.05, subsample=0.9 | $21,887 |
| LightGBM | n_estimators=1000, num_leaves=300, max_depth=12, lr=0.03 | $21,782 |
| ExtraTrees | n_estimators=200, max_depth=30, max_features=0.5 | $23,339 |

## OOF Fold Loop (Section 9)

A critical design detail: since `postal_sector`, `ft_sector_key`, etc. were dropped from `X`, the fold loop reads group values directly from `train_fe` using positional indices:

```python
for group_col, price_col, min_ct, fallback_col in ENCODE_PAIRS:
    tr_groups = train_fe.iloc[tr_idx][group_col].values  # read from train_fe, not X_tr
    va_groups = train_fe.iloc[va_idx][group_col].values
    ...
```

Fold-by-fold results:

| Fold | XGB RMSE | LGBM RMSE | ET RMSE |
|---|---|---|---|
| 1 | $21,593 | $21,642 | $23,399 |
| 2 | $22,460 | $22,515 | $23,980 |
| 3 | $21,738 | $21,727 | $23,594 |
| 4 | $22,031 | $22,024 | $23,701 |
| 5 | $21,933 | $21,803 | $23,550 |
| **Mean** | **$21,951** | **$21,942** | **$23,645** |

Fold 2 is a notable outlier — likely caused by `yr_sector_median_price` sparse groups getting unstable medians in that fold's split.

## Ridge Meta-Model (Section 11)

Ridge regression combines the three OOF predictions with optimal weights:

| Alpha | Ridge OOF RMSE | XGB coef | LGBM coef | ET coef |
|---|---|---|---|---|
| 0.001 | $21,716 | 0.4524 | 0.4113 | 0.1389 |
| 1.0 | $21,716 | 0.4498 | 0.4102 | 0.1426 |
| 10.0 | $21,721 | 0.4318 | 0.4003 | 0.1704 |

Best alpha: **0.001** — coefficients are stable across a wide range, indicating well-calibrated base models. All three coefficients are positive, confirming each model contributes genuine signal.

## Results

| Metric | Value |
|---|---|
| XGB val RMSE | $21,887 |
| LGBM val RMSE | $21,782 |
| ET val RMSE | $23,339 |
| XGB OOF RMSE (5-fold) | $21,953 |
| LGBM OOF RMSE (5-fold) | $21,944 |
| ET OOF RMSE (5-fold) | $23,645 |
| Equal-weight blend OOF RMSE | $21,846 |
| **Ridge meta OOF RMSE** | **$21,716** |
| v10 Kaggle baseline | ~$21,755 |

## Key Takeaways

- **Interaction encodings close a genuine gap**: `ft_sector_median_price` encodes the (flat type, location) price signal that neither feature captures alone — a 5-room in Bishan is worth far more than a 2-room in Bishan, even in the same postal sector.
- **Sparsity is the main risk**: `yr_sector_median_price` has 44.5% of groups with fewer than 10 rows. Fold 2's outlier result ($22,460 vs ~$21,700 in other folds) shows this instability in action. The parent fallback mitigates but doesn't eliminate it.
- **The hdb_age fix removes contradictory signal**: raw `hdb_age` used 2021 as a fixed reference, so `hdb_age + remaining_lease` could sum to 108 rather than 99 for older transactions. Both features now consistently reflect the flat's age at time of sale.
- **Stacking adds ~$230 over the best individual model**: Ridge meta RMSE ($21,716) vs best individual OOF RMSE ($21,944) — a $228 gain from combining diverse models.
- **Ridge OOF is optimistic**: the true Kaggle RMSE will be ~$100–130 higher than the meta-train RMSE. With a gap of ~$128 (observed in v9), estimated Kaggle is ~$21,844 — which may or may not beat v10. Kaggle submission is needed to confirm.
