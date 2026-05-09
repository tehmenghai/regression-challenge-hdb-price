# HDB Price Regression Challenge

## Final Result

**Private Leaderboard RMSE: $21,256.57** — 3-member weighted ensemble (v20)

| Version | Model | Kaggle Public RMSE | Status |
|---|---|---|---|
| v1 | RF Baseline | 25,943.38 | ✅ submitted |
| v2 | RF + Feature Engineering | 27,582.06 | ✅ submitted |
| v3 | XGBoost Tuned + FE | 22,800.92 | ✅ submitted |
| v4 | LightGBM Default + FE | 24,951.72 | ✅ submitted |
| v5 | Blend 45% XGB + 55% LGBM Tuned | 22,428.34 | ✅ submitted |
| v6 | Log target + OOF encoding + interaction features | 22,124.05 | ✅ submitted |
| v7 | Stacking + street_freq (Ridge meta on XGB+LGBM OOF) | 21,905.74 | ✅ submitted |
| v8 | 3-model stack (XGB + LGBM + Extra Trees OOF) | 21,804.67 | ✅ submitted |
| **Team ensemble** | **3-member weighted ensemble (v20)** | **—** | ✅ **final submission → private $21,256.57** |

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

---

## Next Steps
- [ ] Try **Optuna** hyperparameter tuning — Bayesian search, smarter than RandomizedSearchCV
- [ ] Add `block_num` encoding — some block numbers carry premiums
- [ ] More feature engineering — storey band interactions, age × estate maturity
