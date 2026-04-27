# HDB Resale Price — Model Training Analysis Report

> **Purpose:** Consolidated analysis of the full model development journey (v1–v9).
> Covers features tested, pipeline design, training methods, and validation strategy.

---

## 1. Project Overview

### Dataset
- **Train:** 150,634 HDB resale transactions | **Test:** 16,737 transactions
- **Features (raw):** 77 columns — flat attributes, geographic data, estate statistics, nearby amenities
- **Target variable:** `resale_price` (SGD)
- **Competition metric:** RMSE (Root Mean Squared Error, in SGD)

### Journey in numbers
| Metric | Value |
|---|---|
| Best Kaggle RMSE | **$21,756 (v9)** |
| Starting Kaggle RMSE | $25,943 (v1) |
| Total improvement | –$4,187 (–16.1%) |
| Submissions made | 9 (v1–v9) |
| 1st place gap | $289 (1st = $21,466) |

---

## 2. Data Pipeline Steps

The end-to-end pipeline from raw CSV to Kaggle submission:

```
Step 1: Raw data ingestion
   train.csv (150,634 × 77) + test.csv (16,737 × 76)

Step 2: Drop redundant columns
   floor_area_sqft (duplicate), lower/upper/mid (duplicate of mid_storey),
   full_flat_type, address, Tranc_YearMonth, residential, year_completed

Step 3: Fill missing values
   - Amenity count columns (e.g. Hawker_Within_500m) → 0 (no amenity present)
   - Numeric columns → median imputation (inside sklearn Pipeline)
   - Categorical columns → most_frequent imputation + OrdinalEncoder

Step 4: Feature engineering (16 features created — see Section 3)

Step 5: OOF target encoding (v6+, prevents leakage)
   - town_median_price (OOF, 5-fold)
   - postal_sector_median_price (OOF, 5-fold) [v9+]
   - flat_model_median_price (OOF, 5-fold) [v9+]

Step 6: Drop high-cardinality raw columns replaced by encodings
   - postal (9,125 levels) → replaced by postal_sector_median_price
   - block (2,514 levels) → replaced by block_num (numeric)

Step 7: Log-transform target (v6+)
   y_log = np.log1p(resale_price)
   → converts RMSE from absolute to percentage-error-like loss
   → reduces the outsized influence of $1M+ flats on the loss function

Step 8: 5-Fold KFold split (shuffle=True, random_state=42)
   Per fold: train on 4 folds, validate on 1 fold

Step 9: Hyperparameter tuning (RandomizedSearchCV, inner 3-fold CV)
   Tunes XGBoost, LightGBM, ExtraTrees on 80% train / 20% holdout

Step 10: OOF prediction generation (stacking, v7+)
   Each base model trained 5× (once per fold)
   Validation predictions collected → OOF vector of length 150,634

Step 11: Ridge meta-learner
   Trained on [xgb_oof, lgbm_oof, et_oof] → learns optimal blend coefficients

Step 12: Test predictions
   Each base model averaged over 5 folds → meta-model predicts → expm1 inversion

Step 13: Submission CSV
   Kaggle format: Id, Predicted
```

---

## 3. Feature Analysis

### 3.1 Complete Feature Inventory (v9 model — 71 features)

#### Raw Features from Dataset (kept after dropping redundancies)
| Category | Example features |
|---|---|
| Flat attributes | `flat_type`, `flat_model`, `floor_area_sqm`, `mid_storey`, `max_floor_lvl`, `storey_range` |
| Location (geographic) | `Latitude`, `Longitude`, `town`, `street_name`, `planning_area` |
| Lease | `lease_commence_date`, `Tranc_Year`, `Tranc_Month` |
| Estate stats | `total_dwelling_units`, `1room_sold` – `studio_apartment_sold`, `1room_rental` – `other_room_rental` |
| Nearby amenities | `mrt_nearest_distance`, `Mall_Nearest_Distance`, `Hawker_Nearest_Distance`, `mrt_name`, `Hawker_Within_500m/1km/2km`, `Mall_Within_500m/1km/2km` |
| Admin | `market_hawker`, `miscellaneous`, `multistorey_carpark`, `precinct_pavilion` |

#### Engineered Features (16 features created)
| Feature | Formula / Source | What it captures | Version added |
|---|---|---|---|
| `remaining_lease` | `99 - (Tranc_Year - lease_commence_date)` | Years left on lease; directly tied to flat value | v2 |
| `dist_to_cbd` | Haversine km to Raffles Place (1.2847°N, 103.851°E) | Distance premium — closer to CBD = higher price | v2 |
| `is_mature_estate` | Binary: 1 if town in 15 HDB-designated mature towns | ~10–20% price premium for mature estates | v2 |
| `tranc_month_sin` | `sin(2π × Tranc_Month / 12)` | Cyclical month encoding — fixes Dec→Jan discontinuity | v2 |
| `tranc_month_cos` | `cos(2π × Tranc_Month / 12)` | Cyclical month encoding (pair with sin) | v2 |
| `total_sold` | Sum of 8 `xroom_sold` columns | Total units sold in the block | v2 |
| `total_rental` | Sum of 4 `xroom_rental` columns | Total rental units — supply signal | v2 |
| `rental_ratio` | `total_rental / total_dwelling_units` | Estate desirability proxy | v2 |
| `floor_area_per_room` | `floor_area_sqm / num_rooms` | Spaciousness signal independent of flat type | v2 |
| `amenity_score` | Normalised inverse sum of MRT/mall/hawker distances | Composite convenience index [0, 1] | v2 |
| `dist_x_storey` | `dist_to_cbd × mid_storey` | Interaction: floor premium is higher nearer CBD | v6 |
| `lease_x_area` | `remaining_lease × floor_area_sqm` | Interaction: larger flats depreciate faster with lease | v6 |
| `log_dist_to_cbd` | `log1p(dist_to_cbd)` | Linearises the curved distance-price relationship | v6 |
| `street_freq` | Count of transactions per `street_name` | Street-level desirability without target leakage | v7 |
| `postal_sector` | First 4 chars of `postal` (598 unique sectors) | Intermediate geography: finer than town, coarser than raw postal | v9 |
| `block_num` | Numeric extraction from `block` string ("123A" → 123) | Block-level signal without 2,514-category noise | v9 |

#### OOF Target-Encoded Features (3 numeric features)
| Feature | Encoding group | Unique groups | Why OOF? |
|---|---|---|---|
| `town_median_price` | `town` | 26 towns | Prevents leakage — val row's encoding from train rows only |
| `postal_sector_median_price` | `postal_sector` | 598 sectors | Same; finer geography than town |
| `flat_model_median_price` | `flat_model` | ~20 models | Type S2 median ~$1M vs Improved ~$450K — huge spread |

---

### 3.2 Feature Importance Analysis

Based on XGBoost and LightGBM feature importance signals from model training (Gain metric — how much each feature reduces prediction error when used in splits):

#### High-Confidence Important Features (strong signal across multiple model versions)

| Rank | Feature | Type | Direction | Evidence / Why it matters |
|---|---|---|---|---|
| 1 | `floor_area_sqm` | Raw | ↑ larger = pricier | PCA PC1 dominant loading; consistently top feature in all tree models |
| 2 | `town_median_price` | OOF-encoded | ↑ higher median town = pricier flat | Direct price-level signal; top target encoding |
| 3 | `postal_sector_median_price` | OOF-encoded (v9+) | ↑ higher sector median = pricier | Replaced 9,125-level noisy ordinal; biggest v9 gain |
| 4 | `mid_storey` | Raw | ↑ higher floor = pricier | Floor level premium; interacts with dist_to_cbd |
| 5 | `remaining_lease` | Engineered | ↑ more years remaining = pricier | Top 2 engineered feature by XGBoost gain (v3); lease decay is fundamental |
| 6 | `dist_to_cbd` | Engineered | ↓ closer to CBD = pricier | Top 2 engineered feature by XGBoost gain (v3); haversine distance |
| 7 | `flat_model_median_price` | OOF-encoded (v9+) | ↑ higher model tier = pricier | ~$550K spread between Type S2 and Improved |
| 8 | `floor_area_per_room` | Engineered | ↑ more space per room = pricier | Spaciousness independent of room count |
| 9 | `is_mature_estate` | Engineered | ↑ mature = +10–20% premium | Verified by consistent positive signal across all models |
| 10 | `amenity_score` | Engineered | ↑ better amenities = pricier | MRT/mall/hawker composite; PCA PC1 contributes amenity density |
| 11 | `dist_x_storey` | Interaction (v6+) | ↑ higher = near-CBD + high floor premium | Captures that floor-level premiums are larger near CBD |
| 12 | `lease_x_area` | Interaction (v6+) | Captures asymmetric depreciation pattern | Larger, older flats decay in value faster |
| 13 | `street_freq` | Behavioral (v7+) | ↑ more transactions = more desirable street | Captures block-level desirability without target leakage |
| 14 | `mrt_nearest_distance` | Raw | ↓ closer MRT = pricier | Direct accessibility signal (captured partly by amenity_score too) |
| 15 | `log_dist_to_cbd` | Engineered (v6+) | ↓ less log distance = pricier | Linearises the distance signal for gradient boosting |

#### Features with Moderate or Contextual Signal
| Feature | Note |
|---|---|
| `total_sold`, `rental_ratio` | Estate turnover / rental mix signals; useful in some folds |
| `block_num` | Captures block-within-street patterns (corner units, heritage blocks) |
| `tranc_month_sin/cos` | Seasonal price variation; smaller effect than structural features |
| `mall_nearest_distance`, `hawker_nearest_distance` | Partially captured by `amenity_score` |
| `Latitude`, `Longitude` | Raw GPS signal; partially superseded by engineered geography features |
| `max_floor_lvl` | Proxy for building height/era; correlated with flat_type |

#### Key Insight: Why Engineered Features Outperform Raw Ones
Random Forest v1 achieved $25,943 with raw features alone. Adding explicit engineered features for v2 *worsened* RF (to $27,582) because RF already captures implicit interactions via coordinate splits. However, the same features gave XGBoost a –$4,781 improvement (v3 vs v1) because gradient boosting benefits from explicit, human-interpretable signals. This confirms engineered features are most valuable for boosting models.

---

## 4. Training Methods & Models

### 4.1 Model Evolution

| Version | Model(s) | Key change | Kaggle RMSE |
|---|---|---|---|
| v1 | Random Forest (RF) | Baseline, raw features | $25,943 |
| v2 | RF + Feature Engineering | 9 engineered features | $27,582 ⚠️ |
| v3 | XGBoost (tuned) | Switch to gradient boosting | $22,801 |
| v4 | LightGBM (default) | Different architecture, untuned | $24,952 ⚠️ |
| v5 | 45% XGB + 55% LGBM blend | Fixed-ratio blending | $22,428 |
| v6 | XGB+LGBM blend, log target, OOF encoding | Log transform + leakage fix | $22,124 |
| v7 | Ridge stack on XGB+LGBM OOF | OOF stacking meta-model | $21,906 |
| v8 | Ridge stack on XGB+LGBM+ET OOF | Added ExtraTrees (bagging) | $21,805 |
| **v9** | **Ridge stack on XGB+LGBM+ET OOF** | **Richer OOF encoding + wider search** | **$21,756** |

### 4.2 Final Stack Architecture (v9)

```
150,634 training rows
│
├── 5-Fold KFold (shuffle=True, random_state=42)
│   │
│   ├── [Fold 1 of 5] — train on folds 2,3,4,5 → predict fold 1
│   ├── [Fold 2 of 5] — train on folds 1,3,4,5 → predict fold 2
│   ├── [Fold 3 of 5] — train on folds 1,2,4,5 → predict fold 3
│   ├── [Fold 4 of 5] — train on folds 1,2,3,5 → predict fold 4
│   └── [Fold 5 of 5] — train on folds 1,2,3,4 → predict fold 5
│
├── Base model 1: XGBoost ──────────── xgb_oof (150,634,)
├── Base model 2: LightGBM ─────────── lgbm_oof (150,634,)
└── Base model 3: ExtraTrees ────────── et_oof (150,634,)
         │
         ▼
Ridge meta-learner trained on [xgb_oof, lgbm_oof, et_oof]
         │
         ▼
v9 Ridge equation: 0.3854×XGB + 0.4824×LGBM + 0.1347×ET − 0.0318
         │
         ▼
Test predictions (averaged over 5 folds) → expm1() inversion → submission
```

### 4.3 Base Model Configurations (v9 best hyperparameters)

#### XGBoost (RandomizedSearchCV, 40 trials, 3-fold inner CV)
| Parameter | Best value | What it controls |
|---|---|---|
| `n_estimators` | 2000 | Number of sequential trees |
| `max_depth` | 7 | Maximum tree depth |
| `learning_rate` | 0.05 | Step size (shrinkage) |
| `subsample` | 0.9 | Row sampling per tree |
| `colsample_bytree` | 0.4 | Feature sampling per tree |
| `min_child_weight` | 7 | Minimum data per leaf |
| `reg_alpha` | 0.01 | L1 regularisation |
| `reg_lambda` | 1.5 | L2 regularisation |
| **Val RMSE** | **$21,730** | |

#### LightGBM (RandomizedSearchCV, 40 trials, 3-fold inner CV)
| Parameter | Best value | What it controls |
|---|---|---|
| `n_estimators` | 1000 | Number of trees |
| `max_depth` | 12 | Maximum tree depth |
| `num_leaves` | 300 | Max leaves per tree (leaf-wise growth) |
| `learning_rate` | 0.03 | Step size |
| `subsample` | 0.8 | Row sampling |
| `colsample_bytree` | 0.5 | Feature sampling |
| `min_child_samples` | 20 | Minimum samples per leaf |
| **Val RMSE** | **$21,541** ← best single model | |

#### ExtraTrees (RandomizedSearchCV, 15 trials, 3-fold inner CV)
| Parameter | Best value | What it controls |
|---|---|---|
| `n_estimators` | 300 | Number of parallel trees |
| `max_depth` | 20 | Maximum tree depth (capped to prevent 30+ min fit times) |
| `min_samples_split` | 10 | Min samples to split a node |
| `min_samples_leaf` | 1 | Min samples per leaf |
| `max_features` | 0.8 | Feature fraction per split |
| **Val RMSE** | **$23,238** (weakest solo; still contributes diversity) | |

### 4.4 Ridge Meta-Model (v9)
Ridge regression (L2 regularisation) trained on the 3 base models' OOF predictions:

| Alpha sweep | RMSE | XGB coef | LGBM coef | ET coef |
|---|---|---|---|---|
| 0.001 | $21,627 | 0.3854 | 0.4824 | 0.1347 |
| 0.01 | $21,627 | 0.3854 | 0.4823 | 0.1348 |
| 0.1 | $21,627 | 0.3858 | 0.4815 | 0.1352 |
| 1.0 | $21,628 | 0.3891 | 0.4744 | 0.1390 |
| 10.0 | $21,632 | 0.3983 | 0.4346 | 0.1696 |
| **Best alpha** | **0.001** | | | |

Learned equation: `0.3854 × XGB + 0.4824 × LGBM + 0.1347 × ET − 0.0318`

All three coefficients are positive → confirms each model contributes genuine, non-redundant signal.

### 4.5 Per-Model OOF RMSE (v9 — individual performance)

| Model | OOF RMSE (5-fold mean) | Fold-by-fold |
|---|---|---|
| XGBoost | $21,904 | $21,606 / $22,362 / $21,767 / $21,929 / $21,848 |
| LightGBM | $21,803 | $21,486 / $22,291 / $21,611 / $21,882 / $21,739 |
| ExtraTrees | $23,587 | $23,204 / $23,970 / $23,518 / $23,692 / $23,553 |
| Equal-weight blend | $21,757 | — |
| **Ridge meta (v9)** | **$21,627** | — |

---

## 5. Validation Strategy

### 5.1 Cross-Validation Design

**5-Fold KFold** with `shuffle=True, random_state=42`

- Dataset is randomly shuffled once, then split into 5 equal folds (~30,127 rows each)
- Each fold serves as validation exactly once; the model trains on the remaining 4 folds
- KFold (not StratifiedKFold) — regression target does not require stratification
- `random_state=42` ensures reproducibility across all experiments

### 5.2 Out-of-Fold (OOF) Predictions

OOF predictions are critical to stacking correctness:

```
For each fold i of 5:
   1. Train base model on rows NOT in fold i
   2. Predict rows IN fold i → stored in oof_predictions[fold_i_indices]

After all 5 folds:
   oof_predictions.shape = (150,634,) — one prediction per training row
   Each row's prediction was made by a model that NEVER saw that row
```

This property — "never trained on the row being predicted" — makes OOF predictions honest. Using them to train the Ridge meta-model prevents it from seeing artificially easy predictions (where the base model memorised the answer).

### 5.3 OOF Target Encoding (Preventing Leakage in Features)

Target encoding `town_median_price = median(resale_price per town)` is only valid if the median is computed without using the row being encoded. The naive approach (compute on full train set) causes leakage: each row's encoding partly reflects its own label.

**OOF fix** (v6+): For each of the 5 folds, the median map is computed from the other 4 folds only. Validation rows are encoded with a median that excludes them.

```python
for tr_idx, va_idx in kf.split(groups):
    fold_map = {g: median(prices[tr_idx] for group g)}
    encoded[va_idx] = fold_map[groups[va_idx]]  # never sees val row's price
```

**Double OOF** (v7+): Inside the stacking OOF loop, the three group encodings are *recomputed* from each fold's training rows — preventing leakage at the stacking stage as well.

**Test set encoding**: The full training set median map is used for test rows (test labels are unknown — no leakage possible).

### 5.4 Hyperparameter Tuning Validation

RandomizedSearchCV uses an **inner 3-fold CV** inside each search trial:
- Outer loop: 5-fold KFold (model evaluation)
- Inner loop: 3-fold CV inside RandomizedSearchCV (hyperparameter selection)

Custom scoring function forces the inner CV to optimise dollar-RMSE (not log-space RMSE):
```python
def neg_dollar_rmse(y_log_true, y_log_pred):
    return -sqrt(MSE(expm1(y_log_true), expm1(y_log_pred)))
```
This ensures the search directly optimises the same metric Kaggle evaluates.

### 5.5 CV–Kaggle Gap Tracking

The gap between OOF RMSE (estimated from CV) and Kaggle RMSE (estimated from test set) is monitored per version:

| Version | OOF RMSE | Kaggle RMSE | Gap | Interpretation |
|---|---|---|---|---|
| v5 | $21,570 | $22,428 | **$858** | Large — target leakage inflated OOF |
| v6 | $21,818 | $22,124 | **$306** | Leakage fixed; gap collapsed |
| v7 | $21,841 | $21,906 | **$65** | Excellent — OOF stacking generalises well |
| v8 | $21,708 | $21,805 | **$97** | Slight increase expected with added complexity |
| v9 | $21,627 | $21,756 | **$129** | Healthy; richer features added some variance |

**Rule of thumb:** Gap < $300 = well-calibrated. Gap < $100 = excellent generalization.

---

## 6. Results Progression

### Full Version History

| Version | Model | OOF RMSE | Kaggle RMSE | Gap | Δ vs prev | Status |
|---|---|---|---|---|---|---|
| v1 | RF Baseline | $25,871 | $25,943 | $72 | — | ✅ |
| v2 | RF + Feature Engineering | $26,564 | $27,582 | $1,018 | +$1,639 ↑ | ⚠️ worse |
| v3 | XGBoost Tuned + FE | $22,748 | $22,801 | $53 | –$4,781 ↓ | ✅ |
| v4 | LightGBM Default + FE | $24,348 | $24,952 | $604 | +$2,151 ↑ | ⚠️ worse |
| v5 | Blend 45% XGB + 55% LGBM | $21,570 | $22,428 | $858 | –$524 ↓ | ✅ |
| v6 | Log target + OOF encoding | $21,818 | $22,124 | $306 | –$304 ↓ | ✅ |
| v7 | Stack XGB+LGBM (Ridge) | $21,841 | $21,906 | $65 | –$218 ↓ | ✅ |
| v8 | Stack XGB+LGBM+ET (Ridge) | $21,708 | $21,805 | $97 | –$101 ↓ | ✅ |
| **v9** | **Wider grid + richer encoding** | **$21,627** | **$21,756** | **$129** | **–$49 ↓** | **🏆 best** |

### Improvement Inflection Points

**1. v2 → v3: Biggest single gain (–$4,781)**
Switched from Random Forest to tuned XGBoost. Gradient boosting leverages engineered features (explicit distance, lease, location signals) far better than RF. The same feature set that hurt RF improved XGBoost dramatically.

**2. v5 → v6: Leakage fix (–$304 Kaggle, gap $858 → $306)**
`town_median_price` was being computed on full training data in v5, leaking target information into validation rows. OOF encoding with `log1p()` target transform fixed this and exposed genuine RMSE gains.

**3. v6 → v7: OOF stacking (–$218, gap $306 → $65)**
Replacing fixed-ratio blending (v5/v6) with Ridge meta-stacking on OOF predictions collapsed the CV–Kaggle gap to $65 — the best generalization in the project. The gap improvement confirms stacking is not just better in CV but genuinely more robust.

**4. v7 → v8: Bagging diversity (–$101)**
Adding ExtraTrees (random split thresholds, parallel construction) to the stack brings a fundamentally different error pattern from boosting models. Ridge learned a 22% weight for ET despite it being weaker individually ($23,478 OOF vs $21,841 for the stack). Model diversity is more valuable than model accuracy alone.

**5. v8 → v9: Richer OOF encoding (–$49)**
Replacing 9,125-level ordinal `postal` with 598-level `postal_sector_median_price` OOF encoding gave trees a clean geographic price signal. Block-level and flat model encodings added further signal.

---

## 7. Key Learnings & Conclusions

### Algorithm Choice Matters More Than Feature Engineering
The single largest improvement (–$4,781) came from switching RF → XGBoost, not from adding features. The same features that made RF *worse* (v2 scored $27,582 vs v1's $25,943) made XGBoost dramatically better. Rule: match the model family to the feature representation.

### Target Leakage Is Measurable and Fixable
The $858 CV–Kaggle gap in v5 was a precise signal of target encoding leakage. After the OOF fix in v6, the gap fell to $306. This demonstrates that gap analysis is a reliable diagnostic tool.

### OOF Stacking Generalises Better Than Fixed Blending
The Ridge meta-learner learns context-sensitive blending weights from OOF predictions, rather than applying a single ratio across all examples. The gap collapse from $306 → $65 (v6→v7) confirms this: the model became more honest with itself.

### Model Diversity Beats Model Strength
ExtraTrees (individual OOF RMSE $23,587) contributes positively to the stack despite being weaker than XGBoost ($21,904) and LightGBM ($21,803). The Ridge coefficient of 0.13 for ET confirms it adds genuine signal. This validates the boosting + bagging diversity hypothesis.

### Geographic Granularity Is High Value
The progression from town-level encoding (26 groups) → postal-sector encoding (598 groups) in v9 was the largest single change in that version. Fine-grained geographic price signals are among the most informative features in the dataset, consistent with Singapore's highly location-sensitive housing market.

### What Remains to Close the $289 Gap to 1st Place
| Approach | Expected gain | Why |
|---|---|---|
| Optuna Bayesian HPO | –$50 to –$150 | 100 Optuna trials ≈ 300 random trials; smarter exploration |
| CatBoost as 4th base model | –$50 to –$100 | Native categorical handling, different architecture family |
| Interaction price encodings (flat type × sector) | –$20 to –$80 | 5-room in Queenstown sector ≠ 2-room in Queenstown sector |
| Temporal features (year × sector) | –$20 to –$50 | Price trends differ by location |
