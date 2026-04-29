# HDB Resale Price — Model Training Journey

> **Live document.** Update this file after every Kaggle submission.
> Add a new version card in Section 3, update Sections 0 and 5, then commit.

---

## Section 0 — Current Best

| | |
|---|---|
| **Best Kaggle Public RMSE** | **$21,350.91 (Team Ensemble Weighted)** |
| **Gap to 1st place** | $108 (1st = $21,242.91) |
| **Total improvement from v1** | –$4,592 (–17.7%) |
| **Submissions made** | 19 |
| **Last updated** | 2026-04-28 |

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
| v8          | Stack XGB+LGBM+ET (Ridge)              | `07_extra_trees_stack.ipynb`    | $21,708      | $21,805    | $97  | –$101 ↓ | ✅ |
| v9          | Stack XGB+LGBM+ET (expanded OOF)       | `08_v9_stack.ipynb`             | ~$21,708 OOF | $21,755.56 | ~$47 | –$49 ↓  | ✅ |
| v10–v13     | HPO + H3/block/CatBoost experiments    | `09–13_*.ipynb`                 | $21,578 OOF  | $21,783.02 | $205 | +$27 ↑  | ⚠️ worse on Kaggle |
| v14b        | Lease + interaction features           | `14b_clean_features.ipynb`      | ~$21,593 OOF | —          | —    | —       | ✅ (team sub only) |
| v15         | town×flat_type + Year OOF              | `15_interaction_oofs.ipynb`     | $21,593 OOF  | —          | —    | —       | ✅ (team sub only) |
| v16         | CatBoost 4th base + full retrain       | `16_catboost_stack.ipynb`       | $21,552 OOF  | —          | —    | —       | 🏆 best solo OOF |
| v17         | School name OOF (rejected)             | `17_school_oof.ipynb`           | $21,556 OOF  | —          | —    | —       | ❌ no gain |

**Team Members** (details not available — external models)

| Version     | Model                   | Notebook | CV/OOF RMSE | Kaggle RMSE | Gap | Δ vs prev | Status |
|---|---|---|---|---|---|---|---|
| Likhong-v1  | Unknown model           | —        | —           | $22,187.95  | —   | —         | ✅ team |
| Lanson-v1   | Unknown model           | —        | —           | $21,932.24  | —   | —         | ✅ team |
| Lanson-v2   | Unknown model           | —        | —           | $21,705.81  | —   | —         | ✅ team |
| Lai-v1      | Unknown model           | —        | —           | $35,111.47  | —   | —         | ✅ team |
| Lanson-full | Unknown (100% retrain?) | —        | —           | $21,548.25  | —   | —         | ✅ team |

**Team Ensemble** (MengHai submitted on behalf of team)

| Version      | Model                        | Notebook             | CV/OOF RMSE | Kaggle RMSE    | Gap | Δ vs prev | Status |
|---|---|---|---|---|---|---|---|
| Ensemble-eq  | Equal-weight 3-model blend   | `ensemble_team.ipynb` | —           | $21,359.82     | —   | —         | ✅ |
| **Ensemble-wt**  | **Weighted 3-model blend**   | **`ensemble_team.ipynb`** | —       | **$21,350.91** | —   | —         | **🏆 BEST** |
| Ensemble-rk  | Rank-average 3-model blend   | `ensemble_team.ipynb` | —           | $21,384.22     | —   | —         | ✅ |

---

## Section 2 — CV vs Kaggle Gap Trend

The gap between CV/OOF RMSE and Kaggle RMSE measures model generalisation.
A smaller gap means the model performs consistently on unseen data.

| Version | CV/OOF RMSE | Kaggle RMSE | Gap | What drove the change |
|---|---|---|---|---|
| v5 | $21,570 | $22,428 | **$858** | Target leakage in `town_median_price` — computed on full train |
| v6 | $21,818 | $22,124 | **$306** | OOF encoding for `town_median_price` eliminated leakage |
| v7 | $21,841 | $21,906 | **$65** | OOF stacking + fold-averaged test predictions |
| v8          | $21,708      | $21,805    | **$97** | 3-model stack; slight gap increase vs v7 (expected with added complexity) |
| v9          | ~$21,708 OOF | $21,755.56 | ~$47    | Expanded OOF encodings (postal_sector, flat_model) + retune |
| v16         | $21,552 OOF  | —          | —       | Not individually submitted; 4-model stack used in team ensemble |
| Ensemble-wt | —            | $21,350.91 | —       | Team blend; individual gap metric not applicable |

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
**Date:** 2026-04-23 | **Notebook:** `07_extra_trees_stack.ipynb` | **Status:** ✅ submitted

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

### v9 — Stack XGB+LGBM+ET (Expanded OOF Encodings)
**Date:** 2026-04-23 | **Notebook:** `08_v9_stack.ipynb` | **Status:** ✅ submitted

**What we tried:**
- Added 2 new OOF target encodings: `postal_sector_median_price` (median per postal sector) and `flat_model_median_price` (median per flat model type)
- Re-tuned base models on expanded feature set
- Same 3-model Ridge stack as v8 (XGB + LGBM + ET)

**Why:**
Postal sector (~500 groups) is finer-grained than town (~26 groups) and captures block-level price variation. Flat model (e.g. Model A, Improved, Premium Apartment) carries independent price signal not captured by flat_type alone.

| CV/OOF RMSE | Kaggle RMSE | Gap | Δ vs prev |
|---|---|---|---|
| ~$21,708 | $21,755.56 | ~$47 | –$49 ↓ better |

**Key learning:** Postal sector and flat model OOF encodings gave another –$49 Kaggle improvement. Gap stayed excellent at ~$47, confirming the OOF encoding approach is sound.

---

### v10–v13 — HPO + Feature Experiments (H3, Block, CatBoost)
**Date:** 2026-04-23–27 | **Notebooks:** `09–13_*.ipynb` | **Status:** ⚠️ one submission worse than v9 on Kaggle

**What we tried:**
- v10/v11: H3 geohash encoding (hex-bin spatial clustering), `block_num` numeric extraction
- v12: Standalone CatBoost as base model experiment
- v13: Optuna HPO (Bayesian search, 100 trials each for XGB/LGBM)
- One submission filed April 27 (OOF ~$21,578) → Kaggle $21,783

**Why:**
H3 geohash captures sub-postal-sector spatial signal. Optuna was expected to outperform RandomizedSearchCV. Block number can carry premium signals (corner units, heritage blocks).

| CV/OOF RMSE | Kaggle RMSE | Gap | Δ vs prev |
|---|---|---|---|
| $21,578 OOF | $21,783.02 | $205 | +$27 ↑ worse |

**Key learning:** Despite better OOF, Kaggle was $27 worse than v9. The large gap ($205) suggests some feature added test distribution drift or subtle leakage. These experiments refined the feature set that fed into v14b.

---

### v14b — Lease Depreciation + Storey Interaction Features
**Date:** 2026-04-28 | **Notebook:** `14b_clean_features.ipynb` | **Status:** ✅ (team sub only)

**What we tried:**
- `log_remaining_lease`: log1p transform — linearises the non-linear price depreciation curve
- `lease_below_60`: binary flag for leases with <60 years remaining (accelerated depreciation zone)
- `lease_x_below60`: interaction between remaining_lease and lease_below_60
- `flat_type_rank`: ordinal size rank of flat type (studio=0 → EA=6)
- `storey_x_flattype`: mid_storey × flat_type_rank interaction

**Why:**
HDB prices don't depreciate linearly with lease. The <60-year threshold is a well-known market discontinuity (buyer financing restrictions). Storey premium differs by flat type — high floors matter more for large flats.

| CV/OOF RMSE | Kaggle RMSE | Gap | Δ vs prev |
|---|---|---|---|
| ~$21,593 OOF | — | — | — (not individually submitted) |

**Key learning:** Lease interaction features proved useful as a building block for v15 and v16. Not submitted to Kaggle individually — folded into v15.

---

### v15 — town×flat_type + Transaction Year OOF Encodings
**Date:** 2026-04-28 | **Notebook:** `15_interaction_oofs.ipynb` | **Status:** ✅ (team sub only)

**What we tried:**
- `town_flat_type_median_price`: OOF encoding of the town×flat_type combination (e.g. "Tampines 4-room") — captures town-level price variation that differs by flat size
- `year_median_price`: OOF encoding of transaction year — captures macro price trend per year

**Why:**
A 4-room flat in Bishan vs Jurong West has a very different price, and that gap isn't fully captured by town alone or flat_type alone. The interaction term captures this joint signal. Year encoding captures market cycles (e.g. 2021 vs 2023 peak).

| CV/OOF RMSE | Kaggle RMSE | Gap | Δ vs prev |
|---|---|---|---|
| $21,593 | — | — | –$115 OOF vs v9 |

**Key learning:** town×flat_type interaction OOF is powerful — it reduced OOF by $115 vs v9. Not individually submitted; used as base for v16.

---

### v16 — CatBoost as 4th Base + 100% Data Retrain
**Date:** 2026-04-28 | **Notebook:** `16_catboost_stack.ipynb` | **Status:** 🏆 best solo OOF

**What we tried:**
- Added `CatBoostRegressor` as 4th base model (symmetric/oblivious trees — structurally different from XGB/LGBM gradient boosting)
- CatBoost params: `iterations=1000, depth=7, lr=0.08, l2_leaf_reg=1, colsample_bylevel=0.7`
- Ridge meta now trains on 4 OOF predictions; learned weights: XGB 0.3431, LGBM 0.3673, ET 0.1198, CAT 0.1725
- Also generated 100% data retrain (`sub_v16_fulldata.csv`) using full-data group maps

**Why:**
CatBoost uses symmetric (oblivious) trees that split the same feature at every level — it makes structurally different errors than gradient boosting. Ridge can exploit that diversity. 100% retrain uses all 150K rows (vs 80% per fold) for the final test predictions.

| CV/OOF RMSE | Kaggle RMSE | Gap | Δ vs prev |
|---|---|---|---|
| $21,552 | — | — | –$41 OOF vs v15 |

**Key learning:** CatBoost's structural diversity added real value — OOF dropped $41 despite CatBoost being individually weaker ($22,607) than XGB/LGBM. Confirms the principle: stack diversity matters more than individual strength.

---

### v17 — Primary School Name OOF Encoding (Rejected)
**Date:** 2026-04-28 | **Notebook:** `17_school_oof.ipynb` | **Status:** ❌ no gain

**What we tried:**
- OOF target encoding for `pri_sch_name` (177 primary schools) → `pri_sch_median_price`
- Also included `log_pri_sch_dist` and `log_sec_sch_dist`

**Why:**
Top primary schools (Raffles, Nan Hua) are known to drive HDB price premiums in their 1km registration zone.

| CV/OOF RMSE | Kaggle RMSE | Gap | Δ vs prev |
|---|---|---|---|
| $21,556 | — | — | +$4 ↑ worse than v16 |

**Key learning:** School signal is already captured by `postal_sector_median_price` (598 postal sectors ≈ 200m grid). Adding school OOF on top was redundant — and the additional noise from 177 groups with small counts hurt slightly. Rejected; reverted to v16.

---

### Team Ensemble — Weighted Blend of 3 Best Models
**Date:** 2026-04-28 | **Notebook:** `ensemble_team.ipynb` | **Status:** 🏆 BEST — #2 on Leaderboard

**What we tried:**
- Combined OOF predictions from MengHai (v16, OOF $21,552), Lanson (fulldataV1, Kaggle $21,548), and Likhong's best model
- Tested 3 ensemble methods: equal-weight average, Ridge-optimised weights, rank averaging
- Weighted blend (Ridge on OOF) scored best

**Why:**
Each team member trained on different features and used different algorithms. Their prediction errors on individual houses are partially uncorrelated — averaging causes error cancellation and reduces overall RMSE.

| Ensemble Method | Kaggle RMSE | Δ vs best individual |
|---|---|---|
| Equal weight | $21,359.82 | –$188 |
| **Weighted (Ridge)** | **$21,350.91** | **–$197** |
| Rank average | $21,384.22 | –$164 |

**Key learning:** Team ensemble delivered the largest single improvement in the project (–$197 vs best individual). Error cancellation across diverse models/teams is more powerful than any single model improvement. Gap to 1st place: **$108**.

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
| `postal_sector_median_price` | v9 | Median price per postal sector (OOF) | ✅ yes |
| `flat_model_median_price` | v9 | Median price per flat model type (OOF) | ✅ yes |
| `log_remaining_lease` | v14b | log1p(remaining_lease) — linearises depreciation curve | ✅ yes |
| `lease_below_60` | v14b | Binary: remaining_lease < 60 years | ✅ yes |
| `lease_x_below60` | v14b | remaining_lease × lease_below_60 interaction | ✅ yes |
| `flat_type_rank` | v14b | Ordinal size rank of flat type (studio → EA) | ✅ yes |
| `storey_x_flattype` | v14b | mid_storey × flat_type_rank interaction | ✅ yes |
| `town_flat_type_median_price` | v15 | Median price per town×flat_type combo (OOF) | ✅ yes |
| `year_median_price` | v15 | Median price per transaction year (OOF) | ✅ yes |

---

## Section 5 — What's Next

Current gap to 1st place: **$108** (Group 9: $21,242.91)

| Priority | Experiment | Expected gain | Notes |
|---|---|---|---|
| 1 | **Submit sub_v16_fulldata.csv** | –$50 to –$100 | 100% data retrain already generated — trains all 4 models on full 150K rows |
| 2 | **Retune XGB/LGBM hyperparameters** | –$50 to –$150 | v9 params used since v9; v16 now has 9 more features — params are stale |
| 3 | **Optimise ensemble weights via OOF** | –$20 to –$50 | Grid-search weights on team OOF predictions instead of fixed Ridge meta |
| 4 | **Add more team member OOF predictions** | –$10 to –$30 | More model diversity = more error cancellation in ensemble |

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
