# HDB Resale Price — Model Training Journey

> **Live document.** Update this file after every Kaggle submission.
> Add a new version card in Section 3, update Sections 0 and 5, then commit.

---

## Section 0 — Current Best

| | |
|---|---|
| **Best Kaggle Public RMSE** | **$21,804.67 (v8)** |
| **Gap to 1st place** | $338 (1st = $21,466.23) |
| **Total improvement from v1** | –$4,138 (–16.0%) |
| **Submissions made** | 8 |
| **Last updated** | 2026-04-23 |

---

## Section 1 — Quick Reference Table

| Version | Model | Notebook | CV/OOF RMSE | Kaggle RMSE | Gap | Δ vs prev | Status |
|---|---|---|---|---|---|---|---|
| v1 | RF Baseline | `02_baseline.ipynb` | $25,871 | $25,943 | $72 | — | ✅ |
| v2 | RF + Feature Engineering | `03_feature_engineering.ipynb` | $26,564 | $27,582 | $1,018 | +$1,639 ↑ | ⚠️ worse |
| v3 | XGBoost Tuned + FE | `04_model_tuning.ipynb` | $22,748 | $22,801 | $53 | –$4,781 ↓ | ✅ |
| v4 | LightGBM Default + FE | `04_model_tuning.ipynb` | $24,348 | $24,952 | $604 | +$2,151 ↑ | ⚠️ worse |
| v5 | Blend 45% XGB + 55% LGBM | `04_model_tuning.ipynb` | $21,570 | $22,428 | $858 | –$524 ↓ | ✅ |
| v6 | Log target + OOF encoding | `05_advanced_tuning.ipynb` | $21,818 | $22,124 | $306 | –$304 ↓ | ✅ |
| v7 | Stack XGB+LGBM (Ridge) | `06_stacking.ipynb` | $21,841 | $21,906 | $65 | –$218 ↓ | ✅ |
| **v8** | **Stack XGB+LGBM+ET (Ridge)** | `07_extra_trees_stack.ipynb` | **$21,708** | **$21,805** | **$97** | **–$101 ↓** | **🏆 best** |

---

## Section 2 — CV vs Kaggle Gap Trend

The gap between CV/OOF RMSE and Kaggle RMSE measures model generalisation.
A smaller gap means the model performs consistently on unseen data.

| Version | CV/OOF RMSE | Kaggle RMSE | Gap | What drove the change |
|---|---|---|---|---|
| v5 | $21,570 | $22,428 | **$858** | Target leakage in `town_median_price` — computed on full train |
| v6 | $21,818 | $22,124 | **$306** | OOF encoding for `town_median_price` eliminated leakage |
| v7 | $21,841 | $21,906 | **$65** | OOF stacking + fold-averaged test predictions |
| v8 | $21,708 | $21,805 | **$97** | 3-model stack; slight gap increase vs v7 (expected with added complexity) |

> **Rule of thumb:** Gap < $300 = well-calibrated. Gap < $100 = excellent.

---

## Section 3 — Version Detail Cards

---

### v1 — Random Forest Baseline
**Date:** 2026-04-21 | **Notebook:** `02_baseline.ipynb` | **Status:** ✅ submitted

**What we tried:**
- Random Forest with default + light tuning (`n_estimators=100, max_depth=15, min_samples_leaf=5`)
- Raw features only — no engineering
- 5-fold cross-validation

**Why:**
Establish a baseline. RF is robust out of the box and provides feature importance signals.

| CV RMSE | Kaggle RMSE | Gap | Δ vs prev |
|---|---|---|---|
| $25,871 | $25,943 | $72 | — (first submission) |

**Key learning:** RF gives a solid baseline ($25,943). The small $72 gap means CV is well-calibrated even without special encoding.

---

### v2 — RF + Feature Engineering
**Date:** 2026-04-21 | **Notebook:** `03_feature_engineering.ipynb` | **Status:** ⚠️ worse than v1

**What we tried:**
- Added 9 engineered features: `remaining_lease`, `dist_to_cbd`, `is_mature_estate`, cyclical month encoding, `total_sold`, `rental_ratio`, `floor_area_per_room`, `town_median_price`, `amenity_score`
- Same RF hyperparameters

**Why:**
Domain knowledge suggests distance to CBD, remaining lease, and estate maturity drive prices.

| CV RMSE | Kaggle RMSE | Gap | Δ vs prev |
|---|---|---|---|
| $26,564 | $27,582 | $1,018 | +$1,639 ↑ worse |

**Key learning:** Random Forest already captures lat/lon and age interactions implicitly through its splits. Explicit feature engineering didn't help RF — but these same features gave XGBoost a significant boost in v3.

---

### v3 — XGBoost Tuned + Feature Engineering
**Date:** 2026-04-21 | **Notebook:** `04_model_tuning.ipynb` | **Status:** ✅ submitted

**What we tried:**
- Switched from RF to XGBoost
- RandomizedSearchCV tuning: `n_estimators=1000, max_depth=7, lr=0.05, subsample=0.7, colsample_bytree=0.6, min_child_weight=3`
- Same 9 engineered features from v2

**Why:**
Gradient boosting (XGBoost) is designed to benefit from explicit feature engineering — unlike RF which builds implicit interactions.

| CV RMSE | Kaggle RMSE | Gap | Δ vs prev |
|---|---|---|---|
| $22,748 | $22,801 | $53 | –$4,781 ↓ better |

**Key learning:** The single biggest jump in the project (–$4,781). Gradient boosting + domain-engineered features is a powerful combination.

---

### v4 — LightGBM Default + Feature Engineering
**Date:** 2026-04-21 | **Notebook:** `04_model_tuning.ipynb` | **Status:** ⚠️ worse than v3

**What we tried:**
- Replaced XGBoost with LightGBM using near-default parameters
- `n_estimators=500, max_depth=6, num_leaves=40, lr=0.05`

**Why:**
LightGBM is typically faster and more accurate than XGBoost — but only when properly tuned.

| CV RMSE | Kaggle RMSE | Gap | Δ vs prev |
|---|---|---|---|
| $24,348 | $24,952 | $604 | +$2,151 ↑ worse |

**Key learning:** LightGBM with default parameters underperforms tuned XGBoost. Default `num_leaves=31` is too simple for 150K rows and 60+ features. Tuning is essential.

---

### v5 — Blend 45% XGB + 55% LGBM Tuned
**Date:** 2026-04-21 | **Notebook:** `04_model_tuning.ipynb` | **Status:** ✅ submitted

**What we tried:**
- Tuned LightGBM with RandomizedSearchCV: `n_estimators=1200, max_depth=8, num_leaves=80, lr=0.08`
- Fixed-ratio blend: 45% XGB + 55% LGBM (ratio selected by grid search on val set)

**Why:**
Two models with different error patterns should average out to better predictions than either alone.

| CV RMSE | Kaggle RMSE | Gap | Δ vs prev |
|---|---|---|---|
| $21,570 | $22,428 | $858 | –$524 ↓ better |

**Key learning:** Blending works (–$524 improvement). However, the $858 CV–Kaggle gap is large — `town_median_price` was computed on all training data, creating target leakage. Fix: use OOF encoding.

---

### v6 — Log Target + OOF Encoding + Interaction Features
**Date:** 2026-04-22 | **Notebook:** `05_advanced_tuning.ipynb` | **Status:** ✅ submitted

**What we tried:**
- `log1p(resale_price)` as training target → convert back with `expm1()` for submission
- OOF `town_median_price` encoding (per-fold, prevents leakage)
- 3 interaction features: `dist_x_storey`, `lease_x_area`, `log_dist_to_cbd`
- Re-tuned both XGB and LGBM with new target

**Why:**
Log transform reduces the disproportionate influence of expensive flats on RMSE. OOF encoding fixes the $858 gap from v5.

| CV RMSE | Kaggle RMSE | Gap | Δ vs prev |
|---|---|---|---|
| $21,818 | $22,124 | $306 | –$304 ↓ better |

**Key learning:** Log transform + OOF encoding collapsed the gap from $858 → $306. The –$304 Kaggle improvement confirms the leakage fix was the main driver.

---

### v7 — Stacking: Ridge Meta on XGB+LGBM OOF + street_freq
**Date:** 2026-04-22 | **Notebook:** `06_stacking.ipynb` | **Status:** ✅ submitted

**What we tried:**
- 5-fold OOF predictions from XGB and LGBM
- Ridge meta-model trained on `[xgb_oof, lgbm_oof]` → learns optimal blend per context
- `street_freq` encoding: count of transactions per street (500+ streets → 1 signal)
- Re-tuned base models (RandomizedSearchCV)
- Learned: `0.5704 × XGB + 0.4305 × LGBM – 0.012`

**Why:**
Fixed blend ratio (v5/v6) applies the same XGB/LGBM weights to every flat. Ridge can learn adaptive weights. `street_freq` captures street-level desirability without target leakage.

| CV/OOF RMSE | Kaggle RMSE | Gap | Δ vs prev |
|---|---|---|---|
| $21,841 | $21,906 | $65 | –$218 ↓ better |

**Key learning:** OOF stacking + fold-averaged test predictions collapsed the gap to $65 — excellent generalisation. The –$218 Kaggle improvement came mostly from better-tuned base models and street_freq, not the Ridge meta-model itself (which only beat fixed blend by $3 OOF).

---

### v8 — 3-Model Stack: Ridge Meta on XGB+LGBM+ET OOF
**Date:** 2026-04-23 | **Notebook:** `07_extra_trees_stack.ipynb` | **Status:** 🏆 current best

**What we tried:**
- Added `ExtraTreesRegressor` as 3rd base model (in syllabus — bagging family)
- Ridge meta now trains on `[xgb_oof, lgbm_oof, et_oof]` — 3 features
- Tuned ET: `n_estimators=700, max_features=0.8, max_depth=None, min_samples_split=10`
- Ridge learned: `0.4522 × XGB + 0.3340 × LGBM + 0.2164 × ET – 0.033`

**Why:**
ET (bagging) makes fundamentally different errors than XGB/LGBM (boosting). Adding a structurally different model gives Ridge more signal to correct systematic boosting errors.

| CV/OOF RMSE | Kaggle RMSE | Gap | Δ vs prev |
|---|---|---|---|
| $21,708 | $21,805 | $97 | –$101 ↓ better |

**Key learning:** Even though ET is weaker individually ($23,478), its 22% weight in the Ridge meta-model improved the stack by –$133 OOF and –$101 Kaggle. Bagging/boosting diversity is real and measurable.

---

## Section 4 — Cumulative Feature Engineering

| Feature | Added in | What it captures | In model? |
|---|---|---|---|
| `remaining_lease` | v2 | 99 – (year – lease_commence). More interpretable than age. | ✅ yes |
| `dist_to_cbd` | v2 | Haversine km to Raffles Place | ✅ yes |
| `is_mature_estate` | v2 | Binary flag for 15 mature towns | ✅ yes |
| `tranc_month_sin/cos` | v2 | Cyclical encoding — fixes Dec→Jan discontinuity | ✅ yes |
| `total_sold` | v2 | Sum of 8 xroom_sold columns | ✅ yes |
| `rental_ratio` | v2 | total_rental / total_dwelling_units | ✅ yes |
| `floor_area_per_room` | v2 | floor_area_sqm / num_rooms | ✅ yes |
| `town_median_price` | v2 (leaky) → v6 (OOF) | Median price per town — powerful target signal | ✅ yes (OOF) |
| `amenity_score` | v2 | Composite: 1/mrt + 1/mall + 1/hawker distance | ✅ yes |
| `dist_x_storey` | v6 | dist_to_cbd × mid_storey interaction | ✅ yes |
| `lease_x_area` | v6 | remaining_lease × floor_area_sqm interaction | ✅ yes |
| `log_dist_to_cbd` | v6 | log1p(dist_to_cbd) — linearises distance signal | ✅ yes |
| `street_freq` | v7 | Count of transactions per street_name | ✅ yes |

---

## Section 5 — What's Next

Current gap to 1st place: **$338**

| Priority | Experiment | Expected gain | Notes |
|---|---|---|---|
| 1 | **Optuna hyperparameter tuning** | –$50 to –$150 | Bayesian search, smarter than RandomizedSearchCV. 100 Optuna trials ≈ 300 random trials. |
| 2 | **block_num encoding** | –$30 to –$80 | Extract numeric part of block number. Some block numbers (corner, heritage) carry premiums. |
| 3 | **More interaction features** | –$20 to –$60 | e.g. `storey × mature_estate`, `floor_area × remaining_lease × town_median` |
| 4 | **Longer training (more n_estimators + lower lr)** | –$20 to –$50 | Current LR=0.05 with 1500 trees. Try LR=0.02 with 3000 trees for more stable convergence. |

---

## TEMPLATE — Copy this block for a new version

```
---

### vN — [Model Description]
**Date:** YYYY-MM-DD | **Notebook:** `NN_notebook_name.ipynb` | **Status:** ✅/⚠️/🏆

**What we tried:**
- [bullet 1]
- [bullet 2]

**Why:**
[1–2 sentences on motivation]

| CV/OOF RMSE | Kaggle RMSE | Gap | Δ vs prev |
|---|---|---|---|
| $XX,XXX | $XX,XXX | $XXX | –/+$XXX ↓/↑ better/worse |

**Key learning:** [One sentence takeaway]
```

> After adding the new card:
> 1. Update the Quick Reference Table in Section 1
> 2. Update the Current Best in Section 0 if it's the new best
> 3. Add the gap to Section 2 if it changed significantly
> 4. Update Section 5 (What's Next) to remove completed experiments
> 5. Update `submission_journey.html` to match
> 6. Commit and push both files
