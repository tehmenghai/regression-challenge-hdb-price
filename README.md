# HDB Price Regression Challenge

## Final Result

**Private Leaderboard RMSE: $21,256.57** — 3-member weighted ensemble (v20)  
Total improvement: –$4,687 (–18.1%) from v1 baseline over 21 submissions.

### Individual Submissions

| Version | Model | OOF RMSE | Kaggle RMSE | Notes |
|---|---|---|---|---|
| v1 | RF Baseline | $25,871 | $25,943 | Benchmark |
| v2 | RF + Feature Engineering | $26,564 | $27,582 | Worse — leakage |
| v3 | XGBoost Tuned + FE | $22,748 | $22,801 | –$4,781 breakthrough |
| v4 | LightGBM Default + FE | $24,348 | $24,952 | Worse |
| v5 | Blend 45% XGB + 55% LGBM | $21,570 | $22,428 | First blend |
| v6 | Log target + OOF encoding + interactions | $21,818 | $22,124 | Log transform wins |
| v7 | Stack XGB+LGBM (Ridge meta) | $21,841 | $21,906 | First stack |
| v8 | Stack XGB+LGBM+ET (Ridge meta) | $21,708 | $21,805 | Add ExtraTrees |
| v9 | Stack + expanded OOF encoding | ~$21,708 | $21,756 | Richer OOF features |
| v10–v13 | HPO + H3 geo + CatBoost experiments | $21,578 | $21,783 | Overfit on Kaggle |
| v14b | Lease + interaction features | ~$21,593 OOF | — | Team blend only |
| v15 | town×flat_type + Year OOF | $21,593 OOF | — | Team blend only |
| v16 | CatBoost as 4th base model | $21,552 OOF | — | Best solo OOF |
| v17 | Primary school OOF encoding | $21,556 OOF | — | No gain vs v16 |
| v19 | LGBM + ET Optuna HPO | $21,533 OOF | — | Used in team blend |
| v20 | All Optuna params (LGBM+ET+CatBoost) | $21,465 OOF | $21,827 | Overfit (gap $362) |

### Team Ensemble Submissions

| Submission | Method | Kaggle Public | Private | Notes |
|---|---|---|---|---|
| Ensemble-eq | Equal-weight (3-member) | $21,360 | — | |
| Ensemble-wt | Weighted 1/RMSE² (3-member) | $21,351 | — | |
| Ensemble-rk | Rank average (3-member) | $21,384 | — | |
| 3member-v19 | Equal-weight with MengHai v19 | $21,366 | — | |
| 4member-wt | Weighted incl. Lai | $21,661 | — | Scale drag |
| 3member-equal-v20 | Equal-weight with MengHai v20 | $21,346 | — | |
| **3member-wt-v20** | **Weighted with MengHai v20** | **$21,336** | **$21,256.57** | **Final submission** |

Full tracker: `outputs/submission_tracker.xlsx`

---

## Project Structure

```
├── data/
│   ├── raw/              # Original train.csv and test.csv (do not modify)
│   ├── processed/        # Cleaned and encoded datasets
│   └── external/         # Any supplementary data (e.g. MRT distances)
│
├── notebooks/
│   ├── eda/              # Exploratory data analysis
│   │   ├── 01_eda.ipynb              ✅ done
│   │   └── 05_pca_exploration.ipynb  ✅ done
│   └── experiments/      # Model experiments (numbered sequentially)
│       ├── 02_baseline.ipynb           ✅ done — RF RMSE $25,871, R²=0.9672
│       ├── 03_feature_engineering.ipynb ✅ done — 11 engineered features
│       └── 04_model_tuning.ipynb        ✅ done — XGBoost + LightGBM + blend
│
├── src/
│   ├── features/         # Feature engineering scripts
│   ├── models/           # Train and predict scripts
│   └── evaluation/       # Metrics and cross-validation helpers
│
├── outputs/
│   ├── models/           # Saved model files
│   │   ├── rf_baseline.pkl
│   │   └── rf_feature_eng.pkl
│   ├── submissions/      # Competition submission CSVs
│   │   ├── sub_v1_rf_baseline.csv
│   │   ├── sub_v2_feature_eng.csv
│   │   ├── sub_v3_xgb_tuned.csv
│   │   ├── sub_v4_lgbm.csv
│   │   ├── sub_v5_lgbm_tuned.csv
│   │   └── sub_v5_blend.csv          ← current best
│   ├── figures/          # Charts and plots
│   ├── column_grouping.xlsx          # 6+1 group EDA reference
│   ├── submission_tracker.xlsx       # Score tracking per submission
│   ├── train_summary.xlsx            # EDA stats (info + describe)
│   └── baseline_explained.md/.html  # Layman explanation of baseline
│
├── members/              # Individual member contributions
│   ├── menghai/          # notebooks/ + materials/ + README
│   ├── lanson/
│   ├── ben/
│   ├── likhong/
│   ├── lai/
│   └── shl/
│
└── predict_my_hdb.py     # Interactive price estimator (run: python predict_my_hdb.py)
```

---

## Workflow
| Step | Notebook | Status |
|---|---|---|
| 1. EDA | `notebooks/eda/01_eda.ipynb` | ✅ done |
| 2. Baseline model | `notebooks/experiments/02_baseline.ipynb` | ✅ done |
| 3. Feature engineering | `notebooks/experiments/03_feature_engineering.ipynb` | ✅ done |
| 4. Model tuning | `notebooks/experiments/04_model_tuning.ipynb` | ✅ done — XGB+LGBM tuned, blend v5 |
| 5. Advanced tuning | `notebooks/experiments/05_advanced_tuning.ipynb` | ✅ done — log target + OOF encoding, v6 = 22,124 |
| 6. Stacking | `notebooks/experiments/06_stacking.ipynb` | ✅ done — stacking + street_freq, v7 = 21,905 |
| 7. Extra Trees stack | `notebooks/experiments/07_extra_trees_stack.ipynb` | ✅ done — 3-model stack, v8 = 21,805 |

---

## Engineered Features (notebook 03)
| Feature | Source | Impact |
|---|---|---|
| `remaining_lease` | 99 - hdb_age | Lease years left — more interpretable than age |
| `dist_to_cbd` | Lat/Lon → Raffles Place | #2 feature importance in XGBoost |
| `is_mature_estate` | town flag | Mature estates carry 10–20% premium |
| `tranc_month_sin/cos` | Tranc_Month | Cyclical encoding — fixes Dec→Jan gap |
| `total_sold` | sum xroom_sold | Collapses 8 cols → 1 |
| `rental_ratio` | total_rental / total_units | Block desirability proxy |
| `floor_area_per_room` | floor_area_sqm / rooms | Spaciousness signal |
| `town_median_price` | town target encoding | Direct price signal per town |
| `amenity_score` | MRT + Mall + Hawker distance | Composite convenience score |
