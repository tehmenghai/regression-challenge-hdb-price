# 04 — Model Tuning with XGBoost & LightGBM

> This notebook replaces Random Forest with gradient boosting, tunes XGBoost via RandomizedSearchCV, and blends predictions — dropping val RMSE from $25,871 (RF baseline) to $21,828, with a Kaggle score of $22,428 after blending.

## Overview

After a Random Forest baseline, this notebook asks: can gradient boosting do better? The answer is a resounding yes. XGBoost and LightGBM differ from Random Forest in a fundamental way — instead of building hundreds of independent trees and averaging them, they build each new tree specifically to correct the mistakes the previous trees made. This "learn from your errors" strategy, called gradient boosting, is why these models consistently dominate tabular data competitions.

The notebook runs both models with sensible defaults first, then applies RandomizedSearchCV to explore 30 random hyperparameter combinations for XGBoost. The best configuration cuts val RMSE from $25,871 to $21,828 — a ~$4,000 improvement from architecture change alone. The notebook concludes by blending tuned XGBoost and LightGBM predictions, finding that a 45% XGB / 55% LGBM mix improves further, achieving a Kaggle public score of $22,428.

The same 11-feature engineering pipeline from notebook 03 is applied identically to train and test before any modelling, ensuring no data leakage.

## Why Gradient Boosting over Random Forest?

Random Forest builds all its trees independently in parallel — each tree sees a random subset of data and makes its best guess. The final prediction is just the average. This works well, but no tree ever learns from another's mistakes.

Gradient boosting is sequential. You start with a simple guess (say, the mean price of all flats). Then you build a tree to predict the *residual error* — the gap between what you predicted and what actually happened. You add that correction to the model. Repeat 500–1000 times. Each round, the model gets a little less wrong on the hardest cases.

LightGBM adds one more innovation: it grows trees **leaf-wise** (deepest, highest-gain leaf first) rather than **level-wise**. This means it focuses attention on the rows that are still hardest to predict, often giving better accuracy with fewer iterations.

| | Random Forest | XGBoost / LightGBM |
|---|---|---|
| Tree construction | All trees built independently | Each tree corrects previous errors |
| Growth strategy | Level-wise | XGB: level-wise; LGBM: leaf-wise |
| Speed | Slower on large data | Much faster (LGBM especially) |
| Competition performance | Good baseline | Usually top performer on tabular data |

## Preprocessing Pipeline

Both models use a consistent `sklearn` pipeline: median imputation for numerics, most-frequent imputation + OrdinalEncoder for categoricals. XGBoost and LightGBM can handle missing values natively, but keeping the same pipeline makes model swapping trivial.

```python
num_pipe = Pipeline([('imp', SimpleImputer(strategy='median'))])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
preprocessor = ColumnTransformer([('num', num_pipe, num_cols),
                                   ('cat', cat_pipe, cat_cols)])
```

## XGBoost & LightGBM — Default Baselines

Before tuning anything, both models are run with sensible defaults to establish a gradient boosting baseline:

| Model | Val RMSE | 5-Fold CV RMSE |
|---|---|---|
| RF v1 Baseline | $25,871 | $26,389 |
| XGBoost default | $23,649 | $23,932 ± $328 |
| LightGBM default | $24,018 | $24,348 ± $282 |

XGBoost already beats RF by ~$2,200 before a single hyperparameter is touched. XGBoost edges LightGBM on both metrics, so it's chosen for tuning first.

Key hyperparameters to understand:

| Parameter | What it controls | Typical range |
|---|---|---|
| `n_estimators` | Number of boosting rounds (trees) | 300–1000 |
| `max_depth` | How deep each tree grows | 4–8 |
| `learning_rate` | Contribution of each tree (lower = safer) | 0.01–0.1 |
| `subsample` | Fraction of rows each tree sees | 0.7–1.0 |
| `colsample_bytree` | Fraction of columns each tree sees | 0.6–1.0 |
| `min_child_weight` | Minimum samples required in a leaf | 1–10 |

Rule of thumb: lower `learning_rate` + more `n_estimators` = better accuracy but slower training.

## Hyperparameter Tuning — RandomizedSearchCV

Instead of exhaustively trying every combination (Grid Search, often thousands of fits), RandomizedSearchCV randomly samples 30 combinations from the parameter space. This is 10–100× faster and typically finds a near-optimal result.

```python
param_dist = {
    'model__n_estimators'     : [300, 500, 700, 1000],
    'model__max_depth'        : [4, 5, 6, 7, 8],
    'model__learning_rate'    : [0.01, 0.03, 0.05, 0.08, 0.1],
    'model__subsample'        : [0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0],
    'model__min_child_weight' : [1, 3, 5, 10],
}
xgb_search = RandomizedSearchCV(xgb_default, param_distributions=param_dist,
                                 n_iter=30, cv=3, scoring='neg_root_mean_squared_error',
                                 random_state=42, n_jobs=-1)
```

The winning configuration after 90 total fits (30 combinations × 3 folds):

| Parameter | Winning value |
|---|---|
| `n_estimators` | 1000 |
| `max_depth` | 7 |
| `learning_rate` | 0.05 |
| `subsample` | 0.7 |
| `colsample_bytree` | 0.6 |
| `min_child_weight` | 3 |

Best CV RMSE: **$22,748** → Validation RMSE with winning params: **$21,828**

## Feature Importance — Tuned XGBoost

The feature importance chart confirms that engineered features (highlighted in red) are among the most valuable signals:

- **`dist_to_cbd`** and **`remaining_lease`** — the top two engineered features, consistently high-importance across all models in this project
- **`floor_area_sqm`** and **`town_median_price`** — strongest original/derived features
- **`amenity_score`**, **`is_mature_estate`**, and cyclical month encodings also rank in the top 15

This validates the feature engineering work in notebook 03 — the model genuinely relies on these calculated signals, not just the raw columns.

## Model Blending — v5

Even if one model is slightly better alone, blending often improves accuracy because two models make *different* kinds of errors, and averaging partially cancels them out.

The notebook sweeps from 0% XGB / 100% LGBM to 100% XGB / 0% LGBM in 5% steps, measuring val RMSE at each point. The optimal mix on the validation set was **45% XGBoost + 55% LightGBM**, which achieved:

- Val RMSE: ~$21,500
- Kaggle public score: **$22,428** (improvement over v3's $22,801)

## Results

| Version | Model | Val RMSE | Kaggle RMSE |
|---|---|---|---|
| v1 | RF Baseline | $25,871 | $25,943 |
| v2 | RF + Feature Engineering | $26,125 | $27,582 |
| v3 | XGBoost tuned | $21,828 | $22,801 |
| v4 | LightGBM default | $24,018 | $24,952 |
| v5 | 45% XGB + 55% LGBM blend | ~$21,500 | $22,428 |

## Key Takeaways

- Switching from Random Forest to gradient boosting cut val RMSE by ~$4,000 — the single largest model architecture improvement in the project.
- RandomizedSearchCV with just 30 iterations found params that improved XGBoost from $23,649 → $21,828; exhaustive Grid Search wasn't necessary.
- `dist_to_cbd` and `remaining_lease` are the two most important engineered features — they appear at the top of every importance ranking.
- Blending two independently tuned models (XGB + LGBM) improved over either alone, confirming that diverse models make complementary errors.
- Lower `learning_rate` (0.05) + more trees (1000) consistently outperformed higher learning rates with fewer rounds.
