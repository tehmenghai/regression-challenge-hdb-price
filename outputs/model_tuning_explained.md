# Model Tuning — Full Explanation in Layman Terms

---

## The Big Picture

Our baseline Random Forest scored **$25,871 RMSE**. Feature engineering alone did not help RF (v2 was worse at $27,582). But with gradient boosting models and proper tuning, we can do much better.

> **Tuning is the process of finding the best "settings" (hyperparameters) for a model.**
>
> Think of it like tuning a radio — you tweak the dial until the signal is clearest.
> The model's default settings are a good starting point, but the best settings depend on your data.

This notebook introduces:
1. **XGBoost** — a powerful gradient boosting model
2. **LightGBM** — a faster variant of gradient boosting
3. **RandomizedSearchCV** — automated hyperparameter search
4. **Model blending** — combining two models for a better prediction

---

## Why Gradient Boosting Instead of Random Forest?

| | Random Forest | Gradient Boosting (XGB/LGBM) |
|---|---|---|
| How it builds trees | All 100 trees independently | Each tree corrects the errors of the previous tree |
| Learning style | Parallel (democracy — majority vote) | Sequential (student — learn from mistakes) |
| Speed | Moderate | Faster (especially LightGBM) |
| Accuracy | Good baseline | Usually better with tuning |
| Sensitivity to hyperparameters | Low | High — tuning matters a lot |

### The sequential learning analogy

Random Forest: 100 students each read the textbook independently → average their answers.

Gradient Boosting:
```
Student 1 takes the exam → scores 70% → makes errors on questions 3, 7, 12
Student 2 focuses only on questions 3, 7, 12 → corrects those errors
Student 3 corrects whatever Student 2 still got wrong
... repeat 1000 times
Final answer = sum of all students' contributions
```

Each tree specifically targets where the previous model was wrong — this is called **correcting residuals**.

---

## What is a Residual?

```python
# After fitting model iteration 1:
residuals = y_train - predictions_so_far
# Next tree fits on residuals — it learns what the current model got WRONG
```

Example:
```
Flat 1: Actual $600K, Model predicts $540K → Residual = +$60K (underpredicted)
Flat 2: Actual $400K, Model predicts $430K → Residual = -$30K (overpredicted)

Next tree focuses on: "how do I correct these $60K and $30K errors?"
```

After 1,000 iterations, the model has corrected most errors iteratively.

---

## XGBoost — Extreme Gradient Boosting

```python
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=1000,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.6,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1
)
```

### Hyperparameters explained

#### `n_estimators=1000`
Number of trees (boosting rounds).

- More trees = more iterations of error correction = usually better accuracy
- But too many trees can **overfit** (the model memorises training data, fails on new data)
- We use `early_stopping_rounds` to stop automatically when validation score stops improving

```
Round 1:   RMSE = $45,000
Round 100: RMSE = $28,000
Round 500: RMSE = $23,000
Round 750: RMSE = $22,800  ← best
Round 900: RMSE = $22,900  ← getting worse (overfitting)
Round 1000: RMSE = $23,100 ← still getting worse
```
Early stopping would stop at round 750 and use that model.

---

#### `max_depth=7`
Maximum depth of each tree — how many yes/no questions it can ask.

```
depth=1 (stump):
  Is floor_area > 100?  → done

depth=3:
  Is floor_area > 100?
  ├── YES: Is town = 'BISHAN'?
  │         ├── YES: Is storey > 10? → $600K / $550K
  │         └── NO:  Is dist_to_cbd < 5? → $500K / $450K
  └── NO:  ...

depth=7: 128 possible leaf nodes — very detailed splits
```

- Higher depth = more complex = can capture more patterns
- Too high = overfitting (memorises training data)
- Typical sweet spot: 4–8

---

#### `learning_rate=0.05`
How much each new tree contributes to the final prediction.

```
learning_rate=1.0:  Final = Tree1 + Tree2 + Tree3 + ... (each tree has full weight)
learning_rate=0.1:  Final = 0.1×Tree1 + 0.1×Tree2 + ... (each tree contributes 10%)
learning_rate=0.05: Final = 0.05×Tree1 + 0.05×Tree2 + ... (very small steps)
```

**Lower learning rate = smaller steps = more trees needed = usually better generalisation.**

This is why we use 1,000 trees with lr=0.05 rather than 100 trees with lr=0.5 — smaller steps allow finer correction.

---

#### `subsample=0.7`
For each tree, randomly use only 70% of the training rows.

```
Row usage per tree with subsample=0.7:
Tree 1: uses rows 1,2,4,5,7,8,10... (70% randomly selected)
Tree 2: uses rows 1,3,4,6,8,9,10... (different 70%)
Tree 3: uses rows 2,3,5,6,7,9,10... (different 70% again)
```

Why? **Prevents overfitting.** Each tree sees a slightly different dataset → the forest is more diverse → predictions generalise better to new data.

---

#### `colsample_bytree=0.6`
For each tree, randomly use only 60% of the features (columns).

```
Feature usage per tree with colsample_bytree=0.6:
Tree 1: uses floor_area, dist_to_cbd, town, storey... (60% of columns)
Tree 2: uses flat_type, tranc_year, remaining_lease... (different 60%)
```

Same reason as subsample — forces diversity, prevents overfitting.

---

#### `min_child_weight=3`
A tree cannot split a node if fewer than 3 training samples would go into that branch.

```
Without min_child_weight:
  Is block == 'Block 123A'? → YES: 1 flat, NO: 150,000 flats
  ← overfitting: memorising one specific flat's address

With min_child_weight=3:
  Can't split if the YES branch has fewer than 3 flats
  ← prevents memorising rare specific combinations
```

---

## LightGBM — Light Gradient Boosting Machine

```python
from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(
    n_estimators=1200,
    max_depth=8,
    num_leaves=80,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1
)
```

### How is LightGBM different from XGBoost?

| | XGBoost | LightGBM |
|---|---|---|
| Tree growth | **Level-wise** (grows layer by layer) | **Leaf-wise** (grows the most impactful leaf first) |
| Speed | Slower | 3–10× faster |
| Memory | More | Less |
| Accuracy | Comparable | Comparable (sometimes better) |

### Level-wise vs Leaf-wise tree growth

**XGBoost (level-wise):**
```
Level 1: Split root into 2 nodes
Level 2: Split BOTH nodes into 2 children each (4 total)
Level 3: Split ALL 4 into 2 children each (8 total)
```
Grows evenly, one layer at a time.

**LightGBM (leaf-wise):**
```
Step 1: Split root → leaf A reduces loss by 50, leaf B reduces by 10
Step 2: Split leaf A (biggest gain first) → leaf A1 and A2
Step 3: Split whichever leaf now has biggest gain
```
Always splits the leaf that gives the most improvement — more efficient but can overfit with deep trees.

---

#### `num_leaves=80`
LightGBM's equivalent of max_depth, but controls the total number of leaf nodes directly.

```
max_depth=8  →  2^8 = 256 theoretical max leaves (with level-wise)
num_leaves=80 → exactly 80 leaves (tighter control with leaf-wise)
```

Rule of thumb: `num_leaves < 2^max_depth` to prevent overfitting.
```
2^8 = 256, so num_leaves=80 gives a moderate-complexity tree
```

---

## RandomizedSearchCV — Automated Hyperparameter Search

```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators':    [300, 500, 800, 1000],
    'max_depth':       [4, 5, 6, 7, 8],
    'learning_rate':   [0.01, 0.05, 0.1, 0.2],
    'subsample':       [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree':[0.5, 0.6, 0.7, 0.8],
    'min_child_weight':[1, 3, 5, 7]
}

search = RandomizedSearchCV(
    xgb,
    param_distributions=param_grid,
    n_iter=30,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42
)
```

### Why not try every combination? (GridSearchCV)

The `param_grid` above has:
```
4 × 5 × 4 × 4 × 4 × 4 = 5,120 combinations
Each combination × 3-fold CV = 15,360 model fits
```

At ~2 minutes per fit, that is **512 hours**. Not practical.

### What RandomizedSearchCV does

Instead of all 5,120 combinations, it randomly samples **30** (`n_iter=30`):

```
Trial 1:  n_est=800, depth=6, lr=0.05, sub=0.7, col=0.6, mcw=3  → CV RMSE=23,400
Trial 2:  n_est=300, depth=4, lr=0.1,  sub=0.9, col=0.8, mcw=1  → CV RMSE=24,800
Trial 3:  n_est=1000,depth=7, lr=0.05, sub=0.7, col=0.6, mcw=3  → CV RMSE=22,748 ← best
...
Trial 30: n_est=500, depth=5, lr=0.2,  sub=0.6, col=0.5, mcw=5  → CV RMSE=24,100

Best: Trial 3 → use n_est=1000, depth=7, lr=0.05, sub=0.7, col=0.6, mcw=3
```

30 random trials often find 80–90% of the improvement that 5,120 grid trials would find — in 1/170th of the time.

### `cv=3` — 3-fold cross-validation inside the search

Each trial is evaluated on 3 different train/validation splits:
```
Trial 3, Fold 1: Train on fold 2+3, validate fold 1 → RMSE=22,500
Trial 3, Fold 2: Train on fold 1+3, validate fold 2 → RMSE=23,000
Trial 3, Fold 3: Train on fold 1+2, validate fold 3 → RMSE=22,744
Average: 22,748
```

We use 3 folds (not 5) during the search to save time. 5-fold is used for final evaluation.

---

## Results After Tuning

| Version | Model | Val RMSE | Kaggle RMSE |
|---|---|---|---|
| v1 | RF Baseline | $25,871 | 25,943 |
| v2 | RF + FE | $26,125 | 27,582 |
| v3 | **XGBoost Tuned + FE** | **$21,828** | **22,800** |
| v4 | LightGBM Default + FE | $24,018 | 24,951 |
| v5 (LGBM tuned) | LightGBM Tuned + FE | $21,762 | — |
| **v5 (blend)** | **45% XGB + 55% LGBM** | **$21,570** | **22,428** |

---

## Model Blending

```python
# Final predictions
xgb_pred  = xgb_tuned.predict(X_test_fe)
lgbm_pred = lgbm_tuned.predict(X_test_fe)

# Blend: 45% XGBoost + 55% LightGBM
blend_pred = 0.45 * xgb_pred + 0.55 * lgbm_pred
```

### Why blend two models?

Two different models make **different kinds of errors**. When you average their predictions, errors cancel out.

Analogy: Two doctors with different specialisations give you a diagnosis. Their combined opinion is usually more accurate than either alone.

### Why 45% XGB + 55% LGBM?

We searched across ratios:
```
30% XGB + 70% LGBM: val RMSE = $21,620
40% XGB + 60% LGBM: val RMSE = $21,590
45% XGB + 55% LGBM: val RMSE = $21,570  ← best
50% XGB + 50% LGBM: val RMSE = $21,585
60% XGB + 40% LGBM: val RMSE = $21,650
```

We picked 45/55 because it gave the lowest validation RMSE.

### Why does blending beat either model alone?

```
Flat A: XGB predicts $520K, LGBM predicts $540K → Blend: $530K → Actual: $532K ✓
Flat B: XGB predicts $600K (too high), LGBM predicts $560K → Blend: $578K → Actual: $570K ✓
Flat C: XGB predicts $380K, LGBM predicts $420K (too high) → Blend: $401K → Actual: $400K ✓
```

Neither model is always right. Blending hedges their errors.

### The Kaggle result confirmed this:

```
XGBoost alone:  Kaggle RMSE = 22,800
LightGBM alone: Kaggle RMSE = 24,951  (default — not yet tuned before v5)
Blend 45/55:    Kaggle RMSE = 22,428  ← best, beats XGB alone
```

---

## Understanding Overfitting vs Underfitting

```
                   Underfitting          Good fit          Overfitting
Model complexity:  Too simple     →     Just right   →     Too complex
Training error:    High                 Low               Very low
Validation error:  High                 Low               High (gap from train)

Analogy:
Underfitting: Student memorised only chapter titles — can't answer detailed questions
Overfitting:  Student memorised every word including typos — fails any paraphrase
Good fit:     Student understood the concepts — answers new questions correctly
```

Signs of overfitting in our context:
```
Training RMSE:   $15,000
Validation RMSE: $25,000   ← gap = $10,000 → overfitting
```

How we prevent it:
- `max_depth` limit: prevents trees from being too complex
- `subsample` / `colsample_bytree`: adds randomness, reduces variance
- `min_child_weight` / `min_child_samples`: prevents splits on tiny groups
- `learning_rate`: small steps = less risk of overfitting one batch
- Cross-validation: evaluates on held-out data, not training data

---

## The Submission Pipeline

```python
# Step 1: Retrain on ALL training data (not just 80%)
xgb_tuned.fit(X_fe, y)
lgbm_tuned.fit(X_fe, y)

# Step 2: Predict test set
xgb_test  = xgb_tuned.predict(X_test_fe)
lgbm_test = lgbm_tuned.predict(X_test_fe)

# Step 3: Blend
blend_test = 0.45 * xgb_test + 0.55 * lgbm_test

# Step 4: Save
submission = pd.DataFrame({'Id': test['id'], 'resale_price': blend_test.round(0)})
submission.to_csv('../../outputs/submissions/sub_v5_blend.csv', index=False)
```

Same principle as baseline: we retrain on full data before final submission to use every data point the model can learn from.

---

## Summary — Score Progression

```
Baseline RF (v1)         → $25,871 RMSE   (simple trees, no tuning)
     ↓  added feature engineering
RF + FE (v2)             → $26,125 RMSE   (RF doesn't benefit from FE much)
     ↓  switched to gradient boosting
XGBoost default          → $23,649 RMSE   (just changing the model type helps!)
     ↓  tuned with RandomizedSearchCV
XGBoost tuned (v3)       → $21,828 RMSE   (hyperparameter tuning = -$1,821)
     ↓  added LightGBM tuning + blending
Blend 45% XGB + 55% LGBM → $21,570 RMSE   (blending = -$258)

Total improvement: $25,871 → $21,570 = -$4,301 RMSE
Kaggle confirmed:  $25,943 → $22,428 = -$3,515 RMSE
```

The biggest gain came from **switching models** (RF → XGBoost). Tuning and blending provided meaningful but smaller gains on top.

---

## What's Next

To push below $20,000 RMSE, the likely improvements are:

1. **Log-transform the target** (`np.log1p(resale_price)`) — expensive flats ($1M+) create large absolute errors that dominate RMSE; log transform reduces their outsized impact
2. **Better target encoding** — compute `town_median_price` per CV fold (current implementation has slight data leakage)
3. **More features** — `cutoff_point` ranking (school proximity signal), `dist_to_nearest_mrt_type`
4. **Stacking** — train a meta-model on XGB+LGBM predictions (more powerful than simple blending)

→ Continue in `04_model_tuning.ipynb`
