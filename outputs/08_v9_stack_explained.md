# 08 — v9: RandomizedSearchCV (wider grid) + Richer Feature Encoding

> This notebook adds three new features (postal sector price encoding, flat model price encoding, numeric block extraction), widens the RandomizedSearchCV grids to 40 trials, and applies a 3-model Ridge stack — achieving a Kaggle RMSE of **$21,756** (v8 was $21,805).

## Overview

v8 was already a strong 3-model stack (XGBoost + LightGBM + ExtraTrees on OOF predictions, with a Ridge meta-learner). Two weaknesses remained: high-cardinality columns encoded as arbitrary integers, and a hyperparameter search space that was somewhat narrow. v9 fixes both.

The `postal` column had 9,125 unique values — after OrdinalEncoding, those became meaningless integers 0–9,124 with no geographic meaning. Replacing it with a 4-character `postal_sector` (598 levels) and encoding that sector's *median resale price* via out-of-fold encoding gives the model a real price signal at fine geographic resolution. Similarly, `flat_model` encodes a huge price spread: Type S2 flats have a median price roughly $550K above Improved flats. `block_num` extracts the numeric part of the block string (e.g. "123A" → 123), replacing 2,514 ordinal-integer noise with a genuine number.

The RandomizedSearchCV grids are widened to 40 trials each for XGBoost and LightGBM, with `n_estimators` extended to 2000 and explicit regularisation parameters (`reg_alpha`, `reg_lambda`) added to the search. The 5-fold stacking and Ridge meta-learner architecture are unchanged from v8. The result: Kaggle RMSE **$21,756** — a $49 improvement over v8.

## What Changes vs v8

| Change | Why | Expected Gain |
|---|---|---|
| `postal_sector_median_price` OOF | Replaced 9,125-level ordinal noise with real geographic price signal | ~200–400 |
| `flat_model_median_price` OOF | Type S2 vs Improved price spread ~$550K — direct price-tier signal | ~50–150 |
| `block_num` numeric | Replaced 2,514-level ordinal noise with actual number | ~50–100 |
| Wider RandomizedSearchCV (40 trials) | More combinations searched than v8's 30-trial grid | ~50–150 |

## Feature Engineering

All features from v8 carry forward unchanged: `remaining_lease`, `dist_to_cbd`, `is_mature_estate`, cyclical month encodings, `amenity_score`, interaction features, `street_freq`, and more.

Three new features are added:

```python
# 1. Postal sector: first 4 chars (e.g. "520123" → "5201")
df['postal_sector'] = df['postal'].astype(str).str[:4]

# 2. Block number: extract digits from block string ("123A" → 123)
df['block_num'] = pd.to_numeric(
    df['block'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'
).fillna(0)
```

`postal` and `block` (the raw high-cardinality columns) are then **dropped** from the feature matrix. `postal_sector` itself is used only to compute the OOF median price encoding; the final numeric column `postal_sector_median_price` is what the model sees.

```
postal_sector unique (train): 598    ← vs 9,125 for raw postal
postal_sector unique (test):  573    ← 100% test sectors seen in train
```

## Per-Fold OOF Target Encoding

Target encoding is a powerful technique: replace a categorical group (like "postal sector") with the median resale price of all flats in that group. The risk is **leakage** — if you compute the median on the full training set, the model sees its own labels encoded as inputs.

Out-of-fold (OOF) encoding prevents this. The training set is split into 5 folds. When computing the encoding for fold 1, only folds 2–5 are used to compute the median. Each training row's encoding never sees its own label.

```python
def oof_group_median(group_series, y_series, n_splits=5, random_state=42):
    groups = group_series.values
    y      = y_series.values
    encoded    = np.zeros(len(groups))
    global_med = np.median(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, va_idx in kf.split(groups):
        fold_map = {}
        for g, p in zip(groups[tr_idx], y[tr_idx]):
            fold_map.setdefault(g, []).append(p)
        fold_map = {g: np.median(ps) for g, ps in fold_map.items()}
        for i in va_idx:
            encoded[i] = fold_map.get(groups[i], global_med)
    return encoded
```

This function is applied to `town`, `postal_sector`, and `flat_model`. For the test set (which has no labels), a full-training-set median map is used — no leakage there because test labels are unknown.

## Feature Matrix

After dropping `postal`, `block`, and irrelevant columns, the feature matrix grows from ~68 features (v8) to **71 features**:

```
Features: 71  (num=56, cat=15)
postal / block in cat_cols: False / False   ← successfully dropped
block_num, postal_sector_median_price, flat_model_median_price: all present as numeric
```

## RandomizedSearchCV — XGBoost

The XGBoost grid is widened with `n_estimators` up to 2000, explicit regularisation parameters, and a narrower `colsample_bytree` range (0.4–0.5 sometimes helps with many features):

```python
param_dist_xgb = {
    'model__n_estimators'     : [500, 800, 1000, 1500, 2000],
    'model__max_depth'        : [4, 5, 6, 7, 8, 9],
    'model__learning_rate'    : [0.01, 0.03, 0.05, 0.07, 0.1],
    'model__subsample'        : [0.6, 0.7, 0.8, 0.9],
    'model__colsample_bytree' : [0.4, 0.5, 0.6, 0.7, 0.8],
    'model__min_child_weight' : [1, 3, 5, 7, 10],
    'model__reg_alpha'        : [0, 0.01, 0.1, 1.0],
    'model__reg_lambda'       : [0.5, 1.0, 1.5, 2.0, 3.0],
}
```

Winning configuration (40 trials × 3-fold = 120 fits):

| Parameter | Value |
|---|---|
| `n_estimators` | 2000 |
| `max_depth` | 7 |
| `learning_rate` | 0.05 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.4 |
| `min_child_weight` | 7 |
| `reg_alpha` | 0.01 |
| `reg_lambda` | 1.5 |

**XGB best CV RMSE: $22,611 → Val RMSE: $21,730**

## RandomizedSearchCV — LightGBM

LightGBM's grid adds `num_leaves` (up to 300) and `min_child_samples`. LightGBM grows trees leaf-wise, so `num_leaves` is the main capacity control (more important than `max_depth`):

```python
param_dist_lgbm = {
    'model__n_estimators'     : [500, 800, 1000, 1500, 2000],
    'model__max_depth'        : [5, 6, 7, 8, 10, 12],
    'model__num_leaves'       : [60, 80, 100, 127, 200, 300],
    'model__learning_rate'    : [0.01, 0.03, 0.05, 0.07, 0.1],
    ...
}
```

Winning configuration:

| Parameter | Value |
|---|---|
| `n_estimators` | 1000 |
| `max_depth` | 12 |
| `num_leaves` | 300 |
| `learning_rate` | 0.03 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.5 |
| `min_child_samples` | 20 |

**LGBM best CV RMSE: $22,529 → Val RMSE: $21,541**

## RandomizedSearchCV — Extra Trees

ExtraTrees has no learning rate or boosting mechanics, so Bayesian search adds little here. Kept at RandomizedSearchCV with 15 trials. Critically, `max_depth=None` (fully-grown trees) was removed from the grid — it caused the ET search to run for 30+ minutes on 150,634 rows. Capping `max_depth` at 30 reduced ET training time to ~5 minutes.

```python
param_dist_et = {
    'model__n_estimators'     : [200, 300, 500],
    'model__max_depth'        : [15, 20, 30],    # None removed — too slow
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf' : [1, 2, 4],
    'model__max_features'     : [0.5, 0.6, 0.7, 0.8],
}
```

**ET best CV RMSE: $24,460 → Val RMSE: $23,238**

ET is the weakest individual model but still contributes diversity to the stack.

## 5-Fold OOF Generation

Inside each fold of the OOF loop, the three target encodings (`town_median_price`, `postal_sector_median_price`, `flat_model_median_price`) are **recomputed from fold-train rows only**. This is the critical step that prevents leakage from the encoded features into the validation fold:

```python
ENCODE_PAIRS = [
    ('town',          'town_median_price'),
    ('postal_sector', 'postal_sector_median_price'),
    ('flat_model',    'flat_model_median_price'),
]

for fold, (tr_idx, va_idx) in enumerate(kf5.split(X)):
    X_tr = X.iloc[tr_idx].copy()
    X_va = X.iloc[va_idx].copy()

    # Recompute per fold (prevents leakage)
    for group_col, price_col in ENCODE_PAIRS:
        fold_map = {g: np.median(ps) for g, ps in ...}
        X_tr[price_col] = X_tr[group_col].map(fold_map).fillna(global_med_tr)
        X_va[price_col] = X_va[group_col].map(fold_map).fillna(global_med_tr)

    # Fit all 3 models, collect OOF predictions
```

## Ridge Meta-Model

After collecting OOF predictions from all 3 base models, Ridge learns optimal weights:

| Model | OOF RMSE (individual) |
|---|---|
| XGBoost | ~$21,730 |
| LightGBM | ~$21,541 |
| ExtraTrees | ~$23,238 |
| Equal-weight blend | ~$21,757 |

Ridge meta-learner (alpha=0.001):

| Model | Coefficient |
|---|---|
| XGBoost | 0.3854 |
| LightGBM | 0.4824 |
| ExtraTrees | 0.1347 |

All coefficients are positive — confirming that each model contributes unique, complementary signal to the stack.

**Ridge meta OOF RMSE: $21,627**

## Results

| Version | Model | OOF RMSE | Kaggle RMSE |
|---|---|---|---|
| v6 | Log blend + OOF encoding | $21,818 | $22,124 |
| v7 | Stack XGB+LGBM (Ridge) | $21,841 | $21,906 |
| v8 | Stack XGB+LGBM+ET (Ridge) | — | $21,805 |
| **v9** | **Wider grid + richer encoding** | **$21,627** | **$21,756** |

## Key Takeaways

- Replacing high-cardinality ordinal columns (`postal`, `block`) with meaningful numerics (`postal_sector_median_price`, `block_num`) gave the model real geographic signals instead of arbitrary integer noise.
- OOF target encoding must be recomputed **inside** the OOF loop, not precomputed on the full training set — otherwise you leak the target into the encoded features.
- LightGBM (`num_leaves=300`, `learning_rate=0.03`) was the strongest individual model at val RMSE $21,541; XGBoost (`n_estimators=2000`, `colsample_bytree=0.4`) came second at $21,730.
- ExtraTrees contributed a small but genuine signal (coefficient 0.13) — stacking still improved over the best single model.
- Capping `max_depth` for ExtraTrees (removing `None`) is essential — unbounded trees on 150K rows can run for 30+ minutes.
- The OOF→Kaggle gap widened slightly vs v8 ($21,627 OOF vs $21,756 Kaggle, +$129), suggesting mild overfitting from `postal_sector_median_price` in rare sectors with few training rows.
