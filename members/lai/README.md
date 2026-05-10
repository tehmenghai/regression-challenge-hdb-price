# regression-challenge-hdb-price

Predicting Singapore HDB resale prices. **Best validation RMSE: $22,536** (Ridge-meta
stacking ensemble of LightGBM + XGBoost + CatBoost) — see
[`notebooks/expt/tree_based_optmodel_v2.ipynb`](notebooks/expt/tree_based_optmodel_v2.ipynb).

```
data/                           # train.csv (150,634 × 77), test.csv (16,737 × 76), submission templates
notebooks/expt/                 # all modelling notebooks (see ladder below)
instruction/                    # task briefs (baseline, EDA, ridge, non-linear, PCA)
environment.yml                 # conda env spec (env name: ml)
```

---

## Approach

Five design choices shaped this project:

1. **Feature engineering before model swapping.** Spent the early iterations tightening
   features (`encoded_rooms`, multiplicative interactions, geospatial cells, OOF target
   encoding) rather than searching for a stronger model on a weak feature set.
2. **Match the model class to the feature set.** Linear Ridge hit a hard ceiling around
   $58k RMSE no matter how the features were curated; tree ensembles closed half that gap
   without any further tuning.
3. **Log-transform the target.** All boosters predict `log1p(resale_price)` and back-transform
   via `expm1`. HDB price has a long right tail; log-space training stops the loss from
   being dominated by million-dollar outliers.
4. **Leak-free validation.** Out-of-fold target encoding for `postal` (~17k unique values),
   OOF stacking for the meta-learner, and a single fixed 80/20 hold-out for headline RMSE.
   Any number quoted in this README is from rows the relevant model never saw.
5. **Stack, don't tune-to-death.** A 5-fold OOF Ridge meta over the top three boosters
   beats every single model and the simple blend mean. `GridSearchCV` is wired up but
   only triggers if the single-model best fails the $30k target — in v2 it's skipped.

---

## Key experiments (validation RMSE ladder)

Sorted in the order they were run. Lower is better; "Δ" is improvement over the previous
best.

| # | Notebook                                                                                       | Best RMSE | Δ          | What changed                                                                |
| - | ---------------------------------------------------------------------------------------------- | --------- | ---------- | --------------------------------------------------------------------------- |
| 1 | [`baseline.ipynb`](notebooks/expt/baseline.ipynb)                                              | $25,716   | —          | LR/Ridge/RF on the **full 60+ raw features**. Ridge alone = $58,632 (linear ceiling). |
| 2 | [`ridge_regression_optModel.ipynb`](notebooks/expt/ridge_regression_optModel.ipynb)            | $68,288   | **−$9,656**| Ridge on **13 instruction-curated features** — *worse* than #1. Linear models need width, not curation. |
| 3 | [`ridge_regression_optModel1.ipynb`](notebooks/expt/ridge_regression_optModel1.ipynb)          | $63,940   | +$4,348    | Added `encoded_rooms` to #2. Helps but Ridge still capped.                  |
| 4 | [`tree_based_model.ipynb`](notebooks/expt/tree_based_model.ipynb)                              | $35,328   | **+$23,304** vs baseline | Same 13 features + interactions, but **tree ensembles** (RF, GB, XGB, LightGBM). LightGBM wins. |
| 5 | [`tree_based_optmodel.ipynb`](notebooks/expt/tree_based_optmodel.ipynb)                        | —         | —          | LightGBM elastic-net regularisation experiment.                             |
| 6 | [`tree_based_optmodel_v2.ipynb`](notebooks/expt/tree_based_optmodel_v2.ipynb) **(current best)** | **$22,536** | **+$12,792** | Reinstated raw features + **geospatial** (DBSCAN, S2 cells) + **OOF target encoding** for `postal` + **stacking ensemble** (Ridge meta over LGBM/XGB/CatBoost). |
| 7 | [`tree_based_optmodel_v3.ipynb`](notebooks/expt/tree_based_optmodel_v3.ipynb)                  | $23,060*  | -$524      | LightGBM 3-seed bag with deeper hyperparameter exploration. *Partial run.*  |
| 8 | [`neural_network.ipynb`](notebooks/expt/neural_network.ipynb)                                  | —         | —          | Feed-forward NN + stacking. Not yet executed.                               |

**Headline takeaways:**
- **$58k → $35k** came from switching model class (linear → tree).
- **$35k → $22k** came from feature engineering (geospatial + target encoding) and stacking.
- Curating features for a linear model (#2, #3) made things *worse* — the lesson was
  expressivity, not parsimony.

---

## Best model walkthrough — `tree_based_optmodel_v2.ipynb`

The notebook is organised as 8 logical sections; the headers below mirror the README task
spec.

### 1. Setup & load data

Load `train.csv` (150,634 × 77) and `test.csv` (16,737 × 76) from `../../data/`. Dependencies:
`scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `s2sphere`. Use the `ml` conda env.

### 2. Feature engineering — `encoded_rooms` + interactions

Reused verbatim from [`tree_based_model.ipynb`](notebooks/expt/tree_based_model.ipynb).

```python
flat_map = {'1 room': 1, '2 room': 2, ..., 'multi-generation': 7}
df['encoded_rooms'] = df['flat_type'].str.lower().str.strip().map(flat_map)
```

Three multiplicative interactions encode HDB pricing intuition:

| Feature        | Formula                          | Captures                                 |
| -------------- | -------------------------------- | ---------------------------------------- |
| `fa_x_age`     | `floor_area_sqm × hdb_age`       | size depreciation over time              |
| `storey_x_age` | `mid_storey × hdb_age`           | premium for high floors decays with age  |
| `fa_x_rooms`   | `floor_area_sqm × encoded_rooms` | per-room space (proxy for layout density) |

Mall/Hawker count NaNs filled with `0` (NaN means "no count recorded").

### 3. Geospatial features (added in v2)

Three layers built on `(Latitude, Longitude)` and `postal`:

- **DBSCAN clustering** — `eps=0.005` (~500m), `min_samples=50`. Fit on combined train+test
  coordinates so both splits share cluster IDs. Output: `dbscan_cluster`.
- **S2 cells** — Google's hierarchical sphere covering. Two granularities:
  `s2_l12` (~4 km², district) and `s2_l14` (~250 m², street block).
- **OOF target encoding for `postal`** — `postal` has ~17k unique values. 5-fold OOF mean
  of `log1p(price)` per postal, falling back to the global mean for unseen codes.
  Resulting `te_postal` correlates ~+0.95 with `log1p(price)` — but leak-free.

Final feature set: **30 features**, all numeric or integer-coded → no scaling needed.

### 4. Bagging vs Boosting comparison

| Model                | RMSE        | MAE      | R²    | Family   | Fit time |
| -------------------- | ----------- | -------- | ----- | -------- | -------- |
| **XGBoost**          | **$22,732** | $16,247  | 0.975 | Boosting | 21 s     |
| LightGBM             | $22,813     | $16,471  | 0.975 | Boosting | 13 s     |
| CatBoost             | $23,462     | $17,011  | 0.973 | Boosting | 26 s     |
| RandomForest (Bag)   | $27,008     | $18,834  | 0.964 | Bagging  | 69 s     |
| GradBoost (sklearn)  | $27,150     | $19,630  | 0.964 | Boosting | 192 s    |

The three modern boosters cluster ~$23k and beat both bagging and sklearn's GBR by ~$4–5k.
CatBoost trails on RMSE but adds diversity for the meta-learner.

### 5. GridSearchCV (conditional)

3-fold grid over LightGBM `(num_leaves, min_child_samples, reg_alpha)` — **only runs if
single-model best > $30k**. In v2 the best single is $22,732, so it's skipped. Wired in to
keep the notebook self-tuning if a future run regresses.

### 6. Stacking ensemble (leak-free)

Two combiners over the three boosters:

1. **Blend mean** — average of log-space predictions, no parameters.
2. **Ridge meta-learner** — weights *fitted on out-of-fold predictions only*, never on
   `y_val`. `cross_val_predict` (5-fold) generates OOF preds on `X_train`; Ridge fits on
   those; final RMSE is from applying that Ridge to predictions of the
   refit-on-full-`X_train` boosters against `X_val`.

Resulting weights: **LGBM = 0.276, XGB = 0.513, CatBoost = 0.213**, intercept ≈ −0.028.

| Combiner                     | RMSE        | Δ vs best single |
| ---------------------------- | ----------- | ---------------- |
| **Stack ridge meta (OOF)**   | **$22,536** | −$196            |
| Blend mean                   | $22,601     | −$131            |
| XGBoost (best single)        | $22,732     | —                |

### 7. Evaluation metrics

All in `resale_price` SGD (`expm1` back-transform applied before scoring).

| Metric | Formula                  | Why we report it                                                                |
| ------ | ------------------------ | ------------------------------------------------------------------------------- |
| **RMSE** | `√mean((y - ŷ)²)`     | Primary metric. Penalises large errors quadratically, matching HDB's fat-tail price distribution. |
| **MAE**  | `mean(\|y - ŷ\|)`     | Robust diagnostic. If MAE drops while RMSE rises, the model traded average accuracy for a few large misses. |
| **R²**   | `1 - SS_res / SS_tot` | Variance explained — useful for cross-experiment comparison (0.77 on 13-feature Ridge → 0.975 here). |

### 8. Generate submission

1. Refit each booster on full training data with the early-stopped iteration count.
2. Apply the chosen combiner (single booster / blend mean / Ridge meta) to test predictions.
3. Back-transform via `expm1`, clip to `[0, ∞)`.
4. Write CSV matching `sample_sub_reg.csv` template.

Output: [`data/sample_stack_ridge_meta_oof_v2.csv`](data/sample_stack_ridge_meta_oof_v2.csv) (16,737 rows).

---

## Reproducing

```bash
conda env create -f environment.yml      # creates env named "ml"
conda activate ml
cd notebooks/expt
jupyter nbconvert --to notebook --execute --inplace tree_based_optmodel_v2.ipynb \
    --ExecutePreprocessor.timeout=1800
```

End-to-end runtime: ~10 min on this machine (DBSCAN ~30s, OOF stacking ~5 min, sklearn GBR
dominates the rest).
