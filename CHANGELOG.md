# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-21

### Added
- EDA notebook (`01_eda.ipynb`) — missing value analysis, correlation heatmap, geospatial price maps, skewness analysis
- PCA exploration notebook (`05_pca_exploration.ipynb`) — scree plot, 2D/3D scatter, feature loadings
- Baseline model notebook (`02_baseline.ipynb`) — Ridge Regression and Random Forest, 5-fold CV, submission generation
- Feature engineering notebook (`03_feature_engineering.ipynb`) — 11 engineered features: `remaining_lease`, `dist_to_cbd`, `is_mature_estate`, cyclical month encoding, `total_sold`, `rental_ratio`, `floor_area_per_room`, `town_median_price`, `amenity_score`
- Model tuning notebook (`04_model_tuning.ipynb`) — XGBoost and LightGBM with RandomizedSearchCV, model blending
- Interactive price estimator script (`predict_my_hdb.py`) — CLI tool to estimate personal HDB resale price
- Column grouping reference (`outputs/column_grouping.xlsx`) — 6+1 group taxonomy with keep/drop/review/create labels and stats from train_summary
- Submission score tracker (`outputs/submission_tracker.xlsx`) — tracks all Kaggle submissions with CV and public scores
- EDA summary export (`outputs/train_summary.xlsx`) — dtype, null stats, describe stats for all 77 columns

### Submission History
| Version | Model | Kaggle Public RMSE |
|---|---|---|
| v1 | RF Baseline | 25,943.38 |
| v2 | RF + Feature Engineering | 27,582.06 |
| v3 | XGBoost Tuned + FE | 22,800.92 |
| v4 | LightGBM Default + FE | 24,951.72 |
| v5 | Blend 45% XGB + 55% LGBM Tuned | 22,428.34 |
| v6 | Log target + OOF encoding + interaction features | 22,124.05 |
| v7 | Stacking + street_freq (Ridge meta on XGB+LGBM OOF) | **21,905.74** |
