# HDB Resale Price Prediction — My Approach

**Competition:** HDB Resale Price Prediction (Kaggle)

**Iterations:** v1 → v15

**Objective**: Predict Singapore HDB resale flat prices from a structured tabular dataset (150,634 train / 16,737 test rows). 

## Guiding Principles

- **Baseline fast, then iterate in measured increments** — one hypothesis per version; compound gains beat re-architecting.
- **Trust the leaderboard signal** — the Kaggle test set is i.i.d., so random KFold drove optimisation; walk-forward CV stayed as a diagnostic.
- **Prune ruthlessly** — complexity without measurable RMSE gain is debt.
- **Never re-add what was proven harmful** — LOF, LGBM-DART, ExtraTrees, and block-level TE were each tested, ruled out, and not revisited.

## Pipeline

```
Feature Engineering (~55 features)
  ├── Location:  postal_te, town_te, dist_to_cbd, knn_logprice
  ├── Physical:  floor_area_sqm, storey_pct, flat_type_ord
  ├── Lease:     remaining_lease_at_sale, age_when_sold
  └── Temporal:  Tranc_Year, lag_3m / lag_6m / lag_12m
        │
        ▼
5-Fold KFold CV → Out-of-Fold Predictions
  ├── LGBM-Huber (3-seed avg)
  ├── LGBM-MSE   (3-seed avg)
  ├── XGBoost
  └── Segment-blend LGBM
        │
        ▼
Meta-learner: nnls LinearRegression (non-negative weights)
        │
        ▼
Stack OOF RMSE: $21,832
```

## Highest-Impact Wins

| Win | RMSE Gain | Lesson |
|---|---|---|
| Linear → LightGBM | ~$34,000 | Tree models are not optional for HDB pricing |
| Log1p target transform | Structural | Skew correction is table stakes |
| OOF target encoding | ~$700 | Always encode within CV folds |
| Lag features (3/6/12m rolling medians) | ~$300 | Local price momentum is predictive |
| Feature pruning | ~$300 | Noise hurts tree splits and meta-learner alike |
| Age-when-sold correction | ~$150 | A small data bug fix often beats architectural changes |
| Seeded LGBM averaging | ~$100 | Tail stability on high-value flats |

## What Didn't Work

LOF outlier removal (dropped legitimate premium properties), LGBM-DART (catastrophic collapse), `block_enc` (59% importance but redundant with postal/planning_area), ExtraTrees and HGBT-Quantile (zero meta-weight), and walk-forward as the primary metric (misaligned with the i.i.d. test set).

## Result

Final submission: `20260430_21832_likhong.csv` — CV RMSE **$21,832 SGD** (0.0473 log), pipeline runtime ~104 minutes.

See [post-mortem.md](materials/post-mortem.md) for the full breakdown across all 15 iterations, validation strategy, and learning points.
