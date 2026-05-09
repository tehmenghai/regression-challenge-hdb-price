# MengHai (Yipu)

## My Approach

Led the end-to-end modelling pipeline from baseline to ensemble. My focus was on:

- **Stacking architecture** — built the OOF stacking framework used by the whole team (XGB + LGBM + ExtraTrees base models, Ridge meta-learner)
- **Feature engineering** — log-target transformation, OOF target encoding, geospatial features (H3 hex grids, distance to CBD), postal-sector smoothing, school catchment OOF, price interaction features
- **Hyperparameter tuning** — replaced RandomizedSearchCV with Optuna (Bayesian TPE) across all base models
- **Pipeline automation** — built the config-driven training pipeline with MLflow experiment tracking
- **Team infrastructure** — submission tracker, knowledge base webapp, ensemble coordination

**Personal best:** Kaggle Public RMSE **$21,552** (v16 — CatBoost 4-model stack)

**Final team result:** Private Leaderboard RMSE **$21,256.57** — 3-member weighted ensemble (v20), submitted as `20260429_3member_wt_v20.csv`

---

## Key Experiments

| Notebook | Version | What it tests | Kaggle RMSE |
|---|---|---|---|
| `02_baseline.ipynb` | v1 | RF baseline — benchmark before any feature engineering | $25,943 |
| `03_feature_engineering.ipynb` | v2 | 11 engineered features (lease, CBD distance, cyclical month, etc.) | $27,582 |
| `04_model_tuning.ipynb` | v3–v5 | XGBoost + LightGBM tuning, weighted blend | $22,428 |
| `05_advanced_tuning.ipynb` | v6 | Log target, OOF encoding, interaction features | $22,124 |
| `06_stacking.ipynb` | v7 | Ridge meta-learner on XGB+LGBM OOF | $21,905 |
| `07_extra_trees_stack.ipynb` | v8 | 3-model stack — add ExtraTrees as base model | $21,805 |
| `08_v9_stack.ipynb` | v9 | Wider HPO search + richer OOF encoding | $21,756 |
| `09_postal_smoothing.ipynb` | v10 | OOF smoothing for rare postal sectors | — |
| `11_h3_geospatial.ipynb` | v11 | H3 hex-grid geospatial encoding | — |
| `12_catboost_4th_model.ipynb` | v12 | CatBoost as 4th base model | — |
| `13_optuna_hpo.ipynb` | v13 | Optuna Bayesian HPO on 3-model stack | — |
| `14_new_features.ipynb` | v14 | New feature candidates | — |
| `15_interaction_oofs.ipynb` | v15 | OOF encoding on interaction features | $21,593 OOF |
| `16_catboost_stack.ipynb` | v16 | CatBoost + v15 feature set — **personal best** | **$21,552** |
| `17_school_oof.ipynb` | v17 | Primary school OOF encoding + log distance | $21,552 OOF |
| `19_lgbm_et_optuna_params.ipynb` | v19 | Optuna best params for LGBM + ET | — |
| `20_catboost_optuna.ipynb` | v20 | Optuna best params for CatBoost | — |
| `21_all_optuna_params.ipynb` | v20 final | All Optuna params combined on full stack | $21,336 |
