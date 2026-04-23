# v7 — Stacking & Street Frequency Encoding
## A Layman's Guide to How We Got Our Best Score

| Metric | Value |
|---|---|
| v7 Kaggle RMSE | **$21,906** (↓ $218 vs v6) |
| CV–Kaggle gap | **$64** (↓ from $306 in v6) |
| Training data used | 100% — all 150,633 rows |

---

## What Changed in v7?

| # | Improvement | What it does |
|---|---|---|
| 1 | **Stacking** | Ridge meta-model learns the best blend of XGB + LGBM predictions |
| 2 | **street_freq encoding** | Converts 500+ street names into a single "how popular is this street" number |
| 3 | **Re-tuned base models** | RandomizedSearchCV found better hyperparameters than v6 guesses |

---

## Part 1 — street_name Frequency Encoding

### The problem: 500+ street names confuse the model

Our dataset has over 500 unique street names. Some streets appear only 5 times — the model barely learns from them.

### The fix: count how often each street appears

Replace the raw street name with how many transactions happened on that street in training data:

```
street_name          → street_freq
ANG MO KIO AVE 3     → 3,842   (very popular)
HOLLAND RD           → 412     (premium area)
BUONA VISTA RD       → 89      (smaller estate)
COMPASSVALE BOW      → 12      (newer, fewer transactions)
```

> **Analogy:** Imagine predicting a restaurant bill. "This restaurant served 5,000 customers" vs "this one served 20" tells you something about its location and pricing. Same logic — high-frequency streets tend to be in well-established estates.

### Why is this safe? (No data leakage)

`street_freq` only counts *how many transactions* happened — it never looks at price. It's purely a volume signal.

```python
STREET_FREQ = train['street_name'].value_counts().to_dict()
df['street_freq'] = df['street_name'].map(STREET_FREQ).fillna(0)
```

---

## Part 2 — What is Stacking?

### The problem with blending (v5, v6)

In v5 and v6, we used a **fixed ratio**:
```
v6 blend: 57% × XGB prediction + 43% × LGBM prediction = final prediction
```

This same ratio applies to every flat — a 3-room flat in Jurong, an Executive in Queenstown, a 5-room in Bishan. But what if XGBoost is better for mature estates and LightGBM is better for newer ones?

> **Analogy:** Two chefs — Chef A is great at Asian food, Chef B at Western food. Fixed blending says: "Always serve 57% Chef A + 43% Chef B regardless of the order." Stacking says: "For Asian orders, trust Chef A more. For Western orders, trust Chef B more."

### Step 1: Generate Out-of-Fold (OOF) predictions

Split training into 5 folds. For each fold, train XGB and LGBM on the other 4, predict the held-out fold:

```
Fold 1 (20K rows): Train on folds 2+3+4+5 → predict fold 1 → xgb_oof[fold1]
Fold 2 (20K rows): Train on folds 1+3+4+5 → predict fold 2 → xgb_oof[fold2]
...repeat 5 times for both XGB and LGBM...
```

Result: every training row has 2 honest OOF predictions.

> **Why out-of-fold?** If we trained XGB on all data then predicted those same rows, it would be "too good" — the model memorised them. OOF forces every row to be predicted by a model that has never seen it.

### Step 2: Train the Ridge meta-model

```python
meta_X_train = [[xgb_oof[0], lgbm_oof[0]],   # row 0
                [xgb_oof[1], lgbm_oof[1]],   # row 1
                ...150,633 rows...]

ridge.fit(meta_X_train, y_log)
```

Ridge learned:
```
final = 0.5704 × xgb_prediction + 0.4305 × lgbm_prediction – 0.012
         ↑ trust XGB 57%           ↑ trust LGBM 43%           ↑ small bias correction
```

### Step 3: Predict test data

```python
xgb_test_avg  = average of 5 fold XGB models predicting test rows
lgbm_test_avg = average of 5 fold LGBM models predicting test rows
final_pred = expm1( Ridge.predict([[xgb_test_avg, lgbm_test_avg]]) )
```

---

## Part 3 — Why Ridge and Not Something More Complex?

The meta-model has only **2 input features**. A complex model would massively overfit.

> **Analogy:** Two weather forecasters give you forecasts. You can hire a junior meteorologist (Ridge — simple, reliable) or a full research team (neural network) to combine them. With only 2 inputs, the research team would start reading tea leaves. The junior meteorologist does it fine.

Ridge adds a **regularisation penalty** to prevent extreme coefficients when XGB and LGBM predictions are highly correlated.

```
Without regularisation: coefficients might explode to 2.3 × XGB – 1.8 × LGBM
Ridge (alpha=0.001):    stable weights of 0.57 × XGB + 0.43 × LGBM
```

Alpha barely matters here — all values 0.001 to 1.0 give the same RMSE. This confirms the stacking is very stable.

---

## Part 4 — OOF Results

| Fold | XGB RMSE | LGBM RMSE |
|---|---|---|
| 1 | $21,609 | $21,832 |
| 2 | $22,502 | $22,567 |
| 3 | $21,823 | $22,019 |
| 4 | $22,124 | $22,225 |
| 5 | $21,879 | $21,953 |
| **Mean** | **$21,987** | **$22,119** |

Fold 2 is noticeably harder — it happened to contain more expensive/unusual flats. The mean across all 5 folds is a better estimate.

OOF stacking only improved **$3** over fixed blend ($21,841 vs $21,844). This tells us XGB and LGBM make similar types of errors — a meta-model can't easily decide when to trust one over the other.

---

## Part 5 — The CV–Kaggle Gap Collapsed

| Version | CV/OOF RMSE | Kaggle RMSE | Gap | Meaning |
|---|---|---|---|---|
| v5 | $21,570 | $22,428 | $858 | Large — target leakage |
| v6 | $21,818 | $22,124 | $306 | Smaller — OOF encoding helped |
| **v7** | **$21,841** | **$21,906** | **$64** | **Tiny — excellent generalisation** |

A small gap means the model performs almost as well on new data as on training data.

Why did v7's gap collapse to $64?
1. OOF encoding (from v6) — `town_median_price` no longer leaks target information
2. OOF stacking — meta-model also built without seeing the test set
3. Averaging test predictions across 5 folds — reduces variance

---

## Part 6 — Full Journey

| Version | What we tried | Kaggle RMSE | Key lesson |
|---|---|---|---|
| v1 | Random Forest, raw features | $25,943 | Good starting point |
| v2 | RF + engineered features | $27,582 | RF already captures these implicitly |
| v3 | XGBoost + feature engineering | $22,801 | Gradient boosting benefits from FE |
| v4 | LightGBM (default params) | $24,952 | LGBM needs tuning to beat XGB |
| v5 | Blend 45% XGB + 55% LGBM | $22,428 | Blending two models reduces variance |
| v6 | Log target + OOF encoding | $22,124 | Log transform helps with expensive flats |
| **v7** | **Stacking + street_freq** | **$21,906** | **OOF stacking = best generalisation** |

**Total improvement: –$4,037 RMSE (–15.6%)**

---

## Part 7 — Key Concepts Summary

| Concept | Simple Definition |
|---|---|
| **OOF prediction** | Predicting a row using a model trained without that row |
| **Stacking** | Training a 3rd model to combine predictions from 2+ base models |
| **Blending** | Fixed weighted average of models (used in v5, v6) |
| **Meta-model** | The model on top of base models that makes the final decision |
| **Ridge Regression** | Linear model with regularisation to prevent extreme weights |
| **street_freq** | Count of transactions per street — proxy for street desirability |
| **CV–Kaggle gap** | Difference between training CV RMSE and Kaggle test RMSE |
| **Regularisation (alpha)** | Penalty that stops model weights from going extreme |

---

## What Could We Try Next?

- **CatBoost as third base model** — handles categoricals natively; stack becomes Ridge on [XGB, LGBM, CAT] OOF
- **Optuna hyperparameter tuning** — Bayesian optimisation, 100 trials ≈ 300 random trials in quality
- **block_num encoding** — extract numeric part of block number; corner/landmark blocks may carry premiums
