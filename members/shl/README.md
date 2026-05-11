# HDB Resale Price Prediction — Kaggle Challenge

End-to-end machine learning pipeline to predict Singapore HDB resale prices. Built for the [Regression Challenge — HDB Price](https://www.kaggle.com/competitions/regression-challenge-hdb-price) Kaggle competition.

**Metric:** RMSE (Root Mean Squared Error, in SGD)  
**Training data:** 150,633 transactions | 80 raw features | 2012–2021  
**Test data:** 16,735 transactions  
**Best Kaggle RMSE: S$21,661** (OOF Stacking — LightGBM + XGBoost → Ridge meta-learner, log-target)

---

## Notebook Sequence

| Notebook | Purpose |
|---|---|
| `00_eda.ipynb` | Exploratory data analysis — null audit, target distribution, correlation heatmap, geospatial price map, export `feature_profile.xlsx` |
| `01_preprocessing.ipynb` | Feature engineering pipeline + LinearRegression baseline RMSE |
| `02_linear_lasso_ridge_elasticnet.ipynb` | Regularised linear models, alpha grids, coefficient plots |
| `03_knn_dt_bias_variance.ipynb` | KNN, Decision Tree, bias-variance decomposition, learning curves |
| `04_gradient_boosting_xgb_lgbm.ipynb` | GBR, XGBoost, LightGBM, SHAP beeswarm feature importance |
| `05_hyperparameter_tuning.ipynb` | GridSearchCV, RandomizedSearchCV, Optuna 100-trial Bayesian optimisation |
| `06_kfold_cv.ipynb` | CV strategy comparison: KFold vs StratifiedKFold vs TimeSeriesSplit |
| `07_ensemble_stacking.ipynb` | Voting, weight-optimised blending, OOF stacking — main experimentation notebook |
| `08_final_submission.ipynb` | Retrain on full training set, generate Kaggle submission CSV |

---

## Data Pipeline

### 1. Null Handling

| Column group | Strategy |
|---|---|
| `Mall_Within_500m/1km/2km`, `Hawker_Within_*` | Fill 0 — NaN means no facility in range |
| `Mall_Nearest_Distance` | Fill median by town |
| All other numerics | `SimpleImputer(strategy='median')` |
| Categoricals | `SimpleImputer(strategy='most_frequent')` |

### 2. Feature Engineering (`src/features.py` — `FeatureEngineer`)

**Columns dropped** (redundant or zero-variance):

| Column(s) | Reason |
|---|---|
| `id`, `Tranc_YearMonth`, `address`, `full_flat_type` | Identifiers / composite duplicates |
| `storey_range`, `lower`, `upper`, `mid` | Parsed into `mid_storey`; range string redundant |
| `floor_area_sqft` | Exact linear transform of `floor_area_sqm` |
| `residential` | Constant 'Y' — zero variance |
| `postal`, `year_completed`, `vacancy` | Dominated by Lat/Lon; collinear with age; sparse |
| `*_name`, `*_latitude`, `*_longitude` (MRT, bus, schools) | Distances already encoded |

**Engineered features** (highlighted):

| Feature | Source | Notes |
|---|---|---|
| `cbd_dist_m` | `Latitude`, `Longitude` | Haversine to Raffles Place (1.2830°N, 103.8513°E) |
| `orchard_dist_m` | `Latitude`, `Longitude` | Haversine to Orchard Road (1.3048°N, 103.8318°E) |
| `remaining_lease` | `hdb_age` | 99 − hdb_age; lease depreciation signal |
| `hdb_tranc_age` | `Tranc_Year`, `lease_commence_date` | Transaction year − lease start; captures vintage effect |
| `floor_area_log` | `floor_area_sqm` | log1p — compresses right skew |
| `mrt_dist_log` | `mrt_nearest_distance` | log1p — diminishing returns on proximity |
| `tranc_quarter` | `Tranc_Month` | ⌈Month / 3⌉ — seasonal grouping |
| `year_quarter` | `Tranc_Year`, `tranc_quarter` | `Year × 10 + Quarter` — continuous time index |
| `rental_density` | `total_rental_units / total_dwelling_units` | Block composition signal |
| `block_flat_psqm` | `resale_price / floor_area_sqm` (fitted) | Block + flat_type + flat_model mean price-per-sqm; fallback chain to flat_type × flat_model → flat_type |

### 3. Preprocessing (`src/pipeline.py` — `build_preprocessor`)

| Feature group | Columns | Transformer |
|---|---|---|
| Numeric | 20 features (see above) | `SimpleImputer(median)` → `StandardScaler` (linear) / passthrough (tree) |
| Low-cardinality categorical | `flat_type`, `flat_model`, `planning_area`, `pri_sch_affiliation` | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')` |
| High-cardinality categorical | `block` (2,514 unique), `street_name` (553 unique) | `TargetEncoder(cv=5, smooth='auto')` — CV prevents target leakage |

---

## Model Training

### Algorithms Covered for Learning Purposes

| Notebook | Algorithms |
|---|---|
| 02 | LinearRegression, Lasso, Ridge, ElasticNet |
| 03 | KNeighborsRegressor, DecisionTreeRegressor |
| 04 | GradientBoostingRegressor, XGBRegressor, LGBMRegressor |
| 05 | Hyperparameter tuning (GridSearchCV, RandomizedSearchCV, Optuna) |
| 06 | CV strategy analysis |
| 07 | VotingRegressor, weight-optimised blending, OOF StackingRegressor |

### Evaluation Setup

- **Cross-validation:** `KFold(n_splits=5, shuffle=True, random_state=42)`
- **Metric:** RMSE on held-out fold predictions in original price scale
- **Log-transform:** Target `y_log = np.log(resale_price)` for all boosting and ensemble models; predictions use `np.exp()` before RMSE computation
- **Temporal integrity:** `Tranc_Year` stratification; `TimeSeriesSplit` explored in nb06

### Hyperparameter Tuning

**LightGBM — Optuna (100 trials, TPE sampler, log-target):**
**XGBoost — RandomizedSearchCV (30 candidates, log-target):**

### Ensemble Strategy

```
OOF Stacking (final model):
  1. KFold(n_splits=5) split training set
  2. Train LightGBM + XGBoost on each fold (log-target)
  3. Generate OOF predictions for each base model (log space)
  4. Stack OOF matrix → Ridge(alpha=1.0) meta-learner trained on y_log
  5. Retrain base models on full training set → stack test predictions
  6. meta.predict(test_stack) → np.exp() → submission CSV
```

---

## Key Experiments & Results
**Metric:** RMSE (SGD)  
**Baseline → Best:** S$24,220 → S$21,576 (val) | S$23,044 → S$21,661 (Kaggle test)  
**Total improvement:** ~11% reduction in val RMSE over 5 experiment versions

---

## Experiment Timeline

| Version | Key Change | LightGBM Val | XGBoost Val | Ensemble Val | Kaggle Test |
|---|---|---|---|---|---|
| v1 | Baseline models (n=300, lr=0.05) | S$25,156 | S$23,868 | — | — |
| v2 | Optimised-weight blend, single holdout | S$25,156 | S$23,868 | S$22,751 | S$24,220 |
| v3 | Log-transform target | S$23,689 | S$23,284 | S$22,565 | S$23,044 |
| v4 | n=500, XGBoost tuned, additional regularisation | S$21,916 | S$22,084 | S$21,549 | — |
| v5 | Optuna 100-trial (LGBM) + RandomizedSearch (XGB) | S$21,874 | S$21,876 | S$21,576 | S$21,661 |

### Ensemble Comparison at v5

| Method | Weights | Val RMSE | Delta vs LightGBM solo |
|---|---|---|---|
| LightGBM only | — | S$21,874 | — |
| XGBoost only | — | S$21,876 | +S$2 |
| Equal-weight average | 0.50 / 0.50 | S$21,586 | −S$288 |
| OOF-Optimised Blend | 0.518 / 0.482 | S$21,586 | −S$288 |
| OOF Stacking (Ridge meta) | — | **S$21,576** | **−S$298** |

---

## What Worked

### 1. Log-transform target — biggest single gain (+7% RMSE, ~S$1,700)

Applying `y_log = np.log(resale_price)` and `np.exp(pred)` improved every model. The resale price distribution is right-skewed with a long tail at luxury flat types. Log-transform made the residuals more homoscedastic, which benefited both the gradient boosting models and the Ridge meta-learner.

**Before:** val RMSE ~S$22,751 (ensemble)  
**After:** val RMSE ~S$22,565 (ensemble, same architecture)

### 2. Block-level price-per-sqm (`block_flat_psqm`) — most impactful engineered feature

Fitted during `FeatureEngineer.fit()` as mean `resale_price / floor_area_sqm` grouped by `(block, flat_type, flat_model)`, with a fallback chain to `(flat_type, flat_model)` then `flat_type`. This encodes localised market rate at a granular level that raw block category encoding cannot express. It is fitted inside each CV fold so there is no target leakage.

### 3. TargetEncoder(cv=5) for high-cardinality columns

`block` (2,514 unique values) and `street_name` (553 unique values) are too high-cardinality for one-hot encoding. Using sklearn's `TargetEncoder` with internal CV prevented target leakage while preserving location-level pricing signal — a meaningful contribution over simply dropping these columns or using raw frequency encoding.

### 4. OOF Stacking with Ridge meta-learner

Generating out-of-fold predictions from both base models and training the meta-learner on that OOF matrix (in log space) is the most principled ensemble approach. It avoids the data leakage that would occur if base model predictions were made on training rows used to fit those models.

### 5. Geographic distance features

Computing Haversine distance to Raffles Place (`cbd_dist_m`) and Orchard Road (`orchard_dist_m`) from raw lat/lon added signal beyond including coordinates directly. Raw lat/lon were dropped to avoid multicollinearity with the distance columns.

---

## What Did Not Work / Underperformed

### 1. Adding KNN, GBR to stacking ensemble

Early stacking runs included Ridge, GBR (n=100), and KNN (k=15) alongside LightGBM and XGBoost. These added significant training time (especially `StackingRegressor` calling `cv=5` internally for each base estimator) without improving val RMSE. KNN val RMSE was ~5× worse than LightGBM. These were dropped in favour of a leaner 2-model stack.

### 2. Weight-optimised blending converging to near-equal weights after log-transform

Before log-transform, OOF optimisation produced strongly asymmetric weights (lgbm=0.313, xgb=0.687 in v2). After log-transform, the two models became highly correlated (Pearson r ≈ 0.936) and the optimiser converged to near-equal weights (0.518 / 0.482 in v5). The OOF Stacking Ridge meta-learner gained only S$10 over the blend, making complex ensemble methods hard to justify on this dataset alone.

### 3. Many engineered features were disabled

Several features designed during planning were disabled after empirical testing showed no improvement or caused noise:
- `amenity_score` (Mall_Within_1km + Hawker_Within_1km) — captured by individual counts
- `sinusoidal_month` / `cosine_month` — `tranc_quarter` was sufficient
- `flat_type_ordinal` — redundant with OHE `flat_type`
- `high_storey` binary flag — `mid_storey` continuous value was better
- `lease_pct_remaining` — near-linear with `remaining_lease`, added no new information
- `total_sold_units` / `sold_ratio` — sparse and noisy

### 4. XGBoost accidentally set to lr=0.3 (default) in one iteration

When updating the cell with Optuna-tuned LightGBM params, XGBoost was left with its default `learning_rate=0.3` instead of a tuned value. This produced a misleadingly low val RMSE (S$14,793) due to data leakage in the RandomizedSearchCV setup — the model evaluated on training rows. The error was caught and corrected, but it cost time.

### 5. LightGBM overfitting at n=500 (early runs)

At n=500 estimators with default regularisation, LightGBM achieved train RMSE S$20,124 but val RMSE S$23,689 — a S$3,565 generalisation gap. The Optuna study resolved this by tuning `min_child_samples`, `reg_alpha`, `reg_lambda`, `num_leaves`, and `feature_fraction` jointly.

---

## Lessons Learned

### On modelling

1. **Transform the target before anything else.** Log-transform on a right-skewed price target should be the first experiment, not a late-stage optimisation. The 7% RMSE gain cost almost no added complexity.

2. **Block-level encodings beat raw categoricals.** For property pricing, a fitted mean-price-per-sqm at the block level carries far more signal than any one-hot or frequency encoding of the block ID. Compute it as a stateful transformer inside the pipeline so it is safe in CV.

3. **Ensemble diversity matters more than adding models.** LightGBM and XGBoost are both gradient boosted trees with correlated errors. Adding KNN or GBR did not reduce the ensemble bias. A truly diverse base set (e.g., a well-tuned Ridge + a tree model) would likely gain more than adding a third tree.

4. **Tune before stacking.** The S$298 gain from OOF Stacking over a single LightGBM came almost entirely from better base model hyperparameters (Optuna), not from the stacking architecture itself. Investing 100 Optuna trials in tuning the base models was higher ROI than building a 5-model stacking ensemble.

5. **OOF weight optimisation converges to near-equal weights when models are highly correlated.** When Pearson r > 0.93 between two base model OOF predictions, the optimiser has little curvature to exploit. In this regime, equal-weight averaging is nearly optimal and simpler to reason about.

---

## What Would Be Done Differently

| Area | Change |
|---|---|
| Target | Log-transform from v1, not v3 |
| Feature selection | Run permutation importance after first full model; disable features with near-zero importance earlier |
| Ensemble base models | Add a well-regularised Ridge on the full feature set as a diverse third base model |
| Hyperparameter tuning | Optuna from the start with n_trials=50 as a quick pass before committing to 100 |
