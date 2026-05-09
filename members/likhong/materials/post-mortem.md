# HDB Resale Price Prediction — Project Post-Mortem

**Competition:** HDB Resale Price Prediction (Kaggle)
**Notebooks:** v1 → v15 (15 iterations)
**Final CV RMSE:** $21,832 SGD (0.0473 log)
**Target:** ≤ $22,000 SGD ✓
**Total pipeline runtime:** ~104 minutes

---

## 1. Strategy

### Problem Framing

Predict Singapore HDB resale flat prices from a structured tabular dataset (150,634 training rows, 16,737 test rows). The target `resale_price` is right-skewed — we applied `log1p` transform throughout and reported RMSE in both log and SGD space.

### Guiding Principles

- **Establish a baseline quickly, then iterate in measured increments.** Every version had one primary hypothesis to test; compound gains accumulate faster than re-architecturing from scratch.
- **Trust the leaderboard signal.** Walk-forward CV exposed time-aware degradation honestly (~$24k–$30k) but the Kaggle test set is i.i.d., not temporal. From v14 onward, random KFold was kept as the primary optimisation metric and walk-forward retained as a diagnostic.
- **Prune ruthlessly.** Complexity without measurable RMSE gain is debt. v15's final pruning pass confirmed that cutting dead-weight bases and redundant features cost nothing.
- **Never re-add what has already been proven harmful.** LOF, LGBM-DART, ExtraTrees, and block-level target encoding were each formally tested and ruled out — they were never revisited.

### Version Strategy Map

```
V1  ───── Baseline (LR → RF → LGBM)
V2  ───── LOF + target encoding + 2-base stack
V3  ───── Walk-forward CV + per-segment blend
V4  ───── Bayesian TE + macro index + 5-base stack
V5  ───── Drop LOF · block lags · DART (failed) · nnls meta
V6  ───── Feature pruning · HGBT-Quantile · interaction features
V7  ───── lease_per_sqm · price_tier_enc · PCA amenities  ← Personal best at v7
V8  ───── Crossed TE · stack pruning · ElasticNet   ← Regression
V9  ───── Fix v8 regression · drop noisy features
V10 ───── Curriculum-aligned · KNN-as-feature · lag features
V11 ───── Stronger bases · relaxed stacking guardrails
V12 ───── Walk-forward primary · native cats · ppsm diversity target
V13 ───── +XGBoost · address-level TE · 100% refit
V14 ───── Return to v7 template + seeded LGBM + lag/KNN features
V15 ───── Prune to 4 essential bases                ← Final submission
```

---

## 2. Data Processing & Feature Engineering

### 2.1 Source Data

| Dataset | Rows | Notes |
|---|---|---|
| Training | 150,634 | Post-clean: 149,127 (after v2 LOF pass, later reverted) |
| Test | 16,737 | 16,735 submitted (IDs 182002, 82198 excluded as data-entry errors) |
| Raw features | 77 | Including `resale_price` (train only) |
| Final engineered features | ~55 | After pruning passes |

### 2.2 Target Transform

```
Train:   y = log1p(resale_price)
Predict: resale_price = expm1(ŷ)
```

Without this, skewness 1.084 → penalises high-value predictions disproportionately. After transform: skewness 0.252.

### 2.3 Feature Engineering Progression

```
Stage        Features Added                          Cumulative   RMSE (SGD)
─────────────────────────────────────────────────────────────────────────────
Raw load     77 raw cols                             77           $57,198
Target enc   town_te, flat_model_te (OOF)            +2           $50,728
Basic FE     remaining_lease, hdb_age, dist_to_cbd   +3           $49,658
Ordinal enc  flat_type_ord, storey_mid               +2           $50,403
LGBM (tuned) ─                                       same         $22,963  ← v1
LOF + age    age_when_sold, remaining_lease_at_sale  +2           ~$22,903 ← v2
Geo cluster  K-Means 50-cluster ID + TE              +1           ~$22,800
Macro index  year-on-year town price % change        +1           ~$22,740 ← v3
Lag features lag_3m / lag_6m / lag_12m (town×type)  +3           ~$22,400
KNN feature  5-NN mean log-price (spatial)           +1           ~$22,300
Interactions storey_pct, lease_value, months_2012    +3           $22,128  ← v6
Final prune  Dropped: price_tier_enc, floor_ratio,   −4           $21,832  ← v15
             block_enc, amenity PCA cols
```

### 2.4 Key Features and Why They Mattered

| Feature | Type | Importance (LGBM gain rank) | Notes |
|---|---|---|---|
| `postal` / `postal_te` | Location | #1 | 9,125 unique postcodes — hyper-local price signal |
| `Tranc_Year` | Time | #2 | Year of transaction (price inflation) |
| `floor_area_sqm` | Physical | #3 | Strongest linear predictor (r=0.65) |
| `planning_area` / `town_te` | Location | #4–5 | Estate-level price premiums |
| `hdb_age` / `age_when_sold` | Lease | #6 | Age at transaction, not 2021 — critical correction |
| `remaining_lease_at_sale` | Lease | #7 | Remaining lease at transaction date |
| `storey_pct` | Physical | #8 | storey_mid / max_floor (0–1 normalised) |
| `lag_6m` | Temporal | #9 | town × flat_type rolling 6-month median log-price |
| `dist_to_cbd` | Location | #10 | Euclidean distance to Raffles Place |
| `knn_logprice` | Spatial | #11 | 5-NN mean log-price (neighbourhood signal) |
| `months_since_2012` | Time | #12 | Linear macro-inflation trend |

### 2.5 Features Tested and Dropped

| Feature | Reason Dropped | Version |
|---|---|---|
| `block_enc` (target-encoded block ID) | Captured 59% of tree importance — over-encoded, hurt generalisation | v9 |
| LGBM-DART | Catastrophic: log RMSE 1.4163 (collapsed completely) | v5 |
| ExtraTrees | Meta-weight 0.000 in v7; $26k RMSE solo | v8 |
| HGBT-Quantile | Meta-weight 0.000 in v15; wasted 442s per run | v15 |
| RandomForest | Meta-weight 0.060; solo $24k RMSE; 2,653s per run | v15 |
| LOF outlier removal | Removed legitimate premium properties; RMSE worsened when ON | v5 |
| `price_tier_enc` | Triple-counted geo_cluster + te_town signal | v15 |
| `floor_ratio` | Literal duplicate of `storey_pct` | v15 |
| Amenity PCA components | Low importance; collinearity from reduction barely captured | v9 |
| `mrt_x_cbd` interaction | Signal already captured by dist_to_cbd + lag features | v9 |
| `macro_yoy_x_dist` | Low importance; redundant with macro index + dist | v9 |

---

## 3. Modelling & Experiments

### 3.1 Algorithm Landscape

```
          Linear         Tree (single)      Ensemble
          ─────────      ─────────────      ──────────────────────────
V1        Ridge/Lasso    Decision Tree      Random Forest · LightGBM
V2–V3     (dropped)      (dropped)          LightGBM + XGBoost stack
V4–V6     (dropped)      (dropped)          5-base → 4-base stacks
V7–V9     meta only      (dropped)          4-base refined stacks
V10–V13   meta only      (dropped)          Walk-forward stacking variants
V14–V15   meta only      (dropped)          4-base seeded stack (FINAL)
```

Linear models plateaued at $50k+ RMSE and were retired after v1. Tree-based ensembles dominate because HDB pricing is deeply non-linear (location, lease, floor, amenity interactions that linear models cannot capture).

### 3.2 Base Model Comparison (Final v15 Stack)

```
Base Model          Loss      Algorithm         Solo CV RMSE   Meta Weight
──────────────────────────────────────────────────────────────────────────
LGBM-Huber (3-seed) Huber α=0.9  Leaf-wise GBDT   $22,042       0.3251  ██████████████
LGBM-MSE   (3-seed) Squared err  Leaf-wise GBDT   $22,012       0.2322  ████████████
XGBoost             Squared err  Level-wise GBDT  $22,472       0.1349  ███████
Segment-blend       Squared err  Per flat_type    $22,379       0.3094  ████████████████
Meta intercept      —            nnls Ridge       —            −0.0196
──────────────────────────────────────────────────────────────────────────
Stack (OOF)                                                     $21,832
```

**Diversity source:** XGBoost uses level-wise tree growth (splits the full tree level by level) whereas LightGBM uses leaf-wise growth (splits the single highest-gain leaf). This structural difference creates genuinely uncorrelated residuals — the reason including XGBoost improved the ensemble despite a worse solo score.

### 3.3 Why Seeded Averaging Works

Three LGBMs with different random seeds (`42, 123, 999`) each overfit slightly to different subsets. Averaging their OOF predictions cancels out individual noise, particularly on high-value ("tail") properties where a single model's tree structure can produce unstable predictions.

```
Single LGBM variance  ───── higher on $600k–$1.2M properties
3-seed average        ───── variance reduced, tail RMSE stable
```

### 3.4 Hyperparameter Tuning

| Parameter | V1 Value | Final V15 Value | Notes |
|---|---|---|---|
| `n_estimators` (LGBM) | 400 | 2,000 | More trees + lower LR |
| `learning_rate` (LGBM) | 0.1 | 0.01–0.05 | Slower learning, better generalisation |
| `num_leaves` (LGBM) | 63 | 63–127 | Controls model complexity |
| `objective` (LGBM) | MSE | Huber (α=0.9) | Robustness to outliers |
| `min_child_samples` | default | 20 | Prevents over-splitting on rare flat types |
| XGBoost `max_depth` | — | 6 | Level-wise depth constraint |
| XGBoost `subsample` | — | 0.8 | Row sampling per tree |
| KFold splits | 5 | 5 | No change — stable throughout |

### 3.5 Stacking Meta-Learner Evolution

```
V1–V2   Ridge (standard)          ── allows negative weights; opaque
V3–V4   Ridge per flat_type       ── segment-specific; complex to debug
V5      nnls Ridge                ── non-negative; interpretable
V6–V7   nnls Ridge (4 bases)      ── stable across versions
V8      ElasticNet tested         ── tied with nnls; nnls retained for simplicity
V14–V15 nnls LinearRegression     ── final choice; all weights positive & interpretable
```

**Why nnls (non-negative least squares)?** Meta-learner weights with standard Ridge can go negative, meaning one base "corrects" another by subtraction. This is mathematically valid but hard to interpret and can amplify errors if a base behaves unexpectedly on the test set. Non-negative constraints ensure each base contributes additively.

---

## 4. Validation

### 4.1 CV Strategy Debate: KFold vs Walk-Forward

This was the most contested design decision across the 15 versions.

```
                    Random KFold           Walk-Forward CV
                    ────────────           ───────────────
What it simulates   i.i.d. test sample     Real deployment (future data)
V1–V9 RMSE          ~$22,000–$23,000       ~$24,000–$25,000 (honest gap)
V12–V13 RMSE        —                      ~$29,814 (strict time-aware)
V14–V15 role        Primary metric         Diagnostic only
Leaderboard match   High (i.i.d. test)     Lower (test is not time-ordered)
```

**Resolution:** The Kaggle test set is sampled i.i.d. (not a future time window), so random KFold is the correct optimisation target. Walk-forward CV was retained as a diagnostic — a sustained gap between the two signals potential time-leakage in features.

### 4.2 RMSE Progression Across All Versions

```
$57,198 ─── Raw baseline (v1, linear models)
      │
$22,963 ─── V1: LightGBM tuned (first GBM milestone)
$22,903 ─── V2: + LOF + stacking
$22,740 ─── V3: + segment blend
$23,040 ─── V4: 5-base stack regression (LOF re-added, Bayesian TE)
$22,434 ─── V5: Drop LOF, block lags
$22,128 ─── V6: Feature pruning + HGBT-Quantile
$22,094 ─── V7: lease_per_sqm + PCA ← previous personal best
$22,733 ─── V8: Regression (block_enc over-encoding)
$22,277 ─── V9: Fixed v8 regression
  ~24k  ─── V10–V13: Walk-forward CV experiments (honest gap)
$21,831 ─── V14: Return to v7 + seeded LGBM + lags
$21,832 ─── V15: Pruned stack ← FINAL SUBMISSION
          └──────────────────────────────────────────
                     Target: ≤ $22,000 ✓
```

### 4.3 Per-Decile RMSE (Final v15)

The model learns **percentage error** well — MAPE is flat at ~3.3–4.4% across all price bands. Absolute RMSE grows naturally with price.

```
Decile   Price Band           N        RMSE (SGD)   MAPE
──────   ─────────────────    ──────   ──────────   ────
  1      $150k – $295k        15,738     $15,544    4.4%  ████████
  2      $295k – $330k        14,995     $16,025    3.8%  ████████
  3      $330k – $362k        14,649     $16,332    3.5%  ████████
  4      $362k – $392k        14,877     $17,357    3.5%  █████████
  5      $392k – $420k        15,110     $17,955    3.3%  █████████
  6      $420k – $452k        15,113     $19,637    3.2%  ██████████
  7      $452k – $493k        15,000     $20,257    3.2%  ██████████
  8      $493k – $550k        15,430     $23,689    3.4%  ████████████
  9      $550k – $650k        15,000     $27,989    3.5%  ██████████████
 10      $650k – $1.26M       14,722     $35,443    3.4%  ██████████████████
```

**Insight:** Decile 10 (luxury/large flats) has $35k absolute RMSE but only 3.4% MAPE — the model is proportionally well-calibrated even for premium units.

### 4.4 What Stuck in Validation

| Practice | Verdict | Why |
|---|---|---|
| 5-fold KFold CV | ✓ Kept throughout | Stable, low variance estimates; matches leaderboard i.i.d. sampling |
| OOF target encoding | ✓ Essential | Prevents target-leakage in high-cardinality encoding (town, postal) |
| Walk-forward CV | ✓ As diagnostic | Reveals temporal drift; not primary because test is i.i.d. |
| Hold-out 20% split | ✓ Early versions only | Replaced by full-CV + 100% refit in v14–v15 |
| Per-decile RMSE analysis | ✓ Kept | Identifies segment-wise weaknesses not visible in aggregate RMSE |
| LOF outlier filtering | ✗ Abandoned v5 | Worsened RMSE — removed legitimate premium transactions |

---

## 5. Wins, Failures & What's Next

### 5.1 Highest-Impact Wins (Ranked by RMSE Gain)

| Win | RMSE Gain | Version | Lesson |
|---|---|---|---|
| Linear → LightGBM | ~$34,000 | V1 | Non-linearity dominates HDB pricing; tree models are not optional |
| Log1p target transform | Structural | V1 | Skew correction is table stakes for price prediction |
| OOF target encoding | ~$700 | V1–V2 | Without OOF, target encoding leaks; always encode within CV folds |
| Switching from Ridge to nnls meta | ~$200 | V5 | Interpretability and stability; negative meta-weights are a code smell |
| Feature pruning (dropping weak cols) | ~$300 | V6 | Noise features hurt tree splits and confuse meta-learner equally |
| Age-when-sold correction | ~$150 | V2 | Raw `hdb_age` was as-of-2021, not transaction date — critical data bug |
| Lag features (3/6/12m rolling medians) | ~$300 | V10–V14 | Local price momentum is predictive; temporal context matters |
| Seeded LGBM averaging (3 seeds) | ~$100 | V14 | Tail stability on high-value properties; cheap to implement |
| Stack pruning (6 → 4 bases) | ~$0 (clean) | V15 | Confirmed zero-weight bases; cut 3,095s of runtime for free |
| Return to v7 template (v14) | ~$250 recovery | V14 | Re-grounding after walk-forward detour recovered the best known state |

### 5.2 What Did Not Work

| Failure | Version | Root Cause |
|---|---|---|
| **LOF outlier detection** | V2 → V5 | Flagged legitimate premium properties as outliers; removing them worsened predictions at high-value end |
| **LGBM-DART** | V5 | Catastrophic failure: log RMSE 1.4163 (vs 0.049 target); DART hyperparameters too sensitive for this dataset without deep tuning |
| **block_enc (target-encoded block ID)** | V8 → V9 | 59% of tree importance concentration — over-parameterised; the signal was already captured by postal/planning_area |
| **Crossed TE (flat_type × town)** | V8 | Redundant with existing town_te + flat_type features already in tree splits |
| **ExtraTrees as base** | V4–V7 | Meta-weight 0.000 consistently; ExtraTrees adds no diversity over RF for this problem |
| **HGBT-Quantile as base** | V6–V14 | Meta-weight 0.000 in v15 despite v6 gains; 442s wall time per run with no RMSE contribution by final stack |
| **RandomForest as base** | V1–V14 | Meta-weight 0.060 in final config; solo $24k RMSE; not competitive with GBDTs — costs 2,653s per run |
| **Walk-forward as primary metric** | V12–V13 | Walk-forward RMSE ($29.8k) didn't predict leaderboard performance (i.i.d. test); misaligned optimisation target |
| **address-level TE** | V13 | 9,157 unique addresses with few samples per address; sparse encoding, prone to overfitting |
| **Macro price index (town YoY)** | V4 | Low importance after other macro signals included; redundant with `months_since_2012` + `Tranc_Year` |

### 5.3 Patterns in Failures

```
                "This feature is important in isolation"
                         │
                         ▼
        Does it add signal not already captured?
               │                    │
              YES                   NO
               │                    │
              Use                  Drop ─── block_enc, address_te,
                                            price_tier_enc, floor_ratio
```

The most common failure mode: a feature with high solo importance that turns out to be **redundant** once the full feature set is assembled. The fix is always to check whether removing the feature changes CV RMSE — if it doesn't, drop it.

### 5.4 What's Next (No Constraints)

#### Immediate improvements (high confidence, low effort)

| Idea | Expected Gain | Rationale |
|---|---|---|
| **Optuna / Bayesian hyperparameter tuning** | ~$300–$500 | GridSearchCV was used in v1; Optuna with TPE sampler would explore a much larger hyperparameter space efficiently |
| **CatBoost as 5th base** | ~$100–$200 | Handles categoricals natively with ordered boosting; structurally different from LGBM/XGBoost |
| **Postal-level lag features** | ~$200 | Current lags are at town×flat_type; postal-level would be more granular (already have 9,125 postals) |
| **Expand seeded averaging (5 seeds)** | ~$50–$100 | More seeds reduces variance further; diminishing returns but cheap |
| **Time-series CV as meta-learner input** | ~$100 | Stack meta-learner trained separately on temporal splits to capture drift |

#### Medium-term (meaningful investment)

| Idea | Expected Gain | Rationale |
|---|---|---|
| **Neural network base (tabular transformer)** | ~$500–$1,000 | TabNet / FT-Transformer on tabular data can capture feature interactions GBDT misses; genuinely uncorrelated residuals |
| **Graph features (HDB block neighbourhood graph)** | ~$300–$600 | Model block-to-block adjacency; prices propagate through spatial networks not captured by Euclidean distance |
| **Property transaction history per block** | ~$400–$700 | Full block-level price history (not just 6-month rolling) as feature; captures renovation premiums and estate maturity |
| **School proximity scoring with ranking** | ~$200 | Binary school proximity is weak; weighted by school ranking/type (Primary 1 phase, MOE tier) is a known HDB premium |
| **MRT phase completion features** | ~$300 | Upcoming MRT stations drive pre-completion premiums; phase data is public |

#### Architectural shift (if given full resources)

| Idea | Notes |
|---|---|
| **Online learning / model refresh** | HDB prices shift quarterly; retrain on rolling 24-month window rather than full history |
| **Separate models per flat type** | Train specialist models for 3-ROOM vs 4-ROOM vs Executive; each has a different price structure and feature importance profile |
| **Causal price model** | Current model is purely predictive; a structural equation model capturing the causal path (location → school → price) would generalise better under policy shifts |
| **Retrieval-augmented pricing** | Embed each flat as a vector (location, features, age) and retrieve nearest comparable transactions at inference time; blend retrieval-based estimate with model output |

---

## 6. ML Student Learning Points

### 6.1 Always Transform a Skewed Target

```python
# Before: skewness 1.084 — RMSE penalises high values too heavily
y_train = df['resale_price']

# After: skewness 0.252 — symmetric, model optimises evenly
y_train = np.log1p(df['resale_price'])
# At prediction time:
predictions = np.expm1(model.predict(X_test))
```

**Why it matters:** When you minimise MSE on a skewed target, the model over-corrects for high-value outliers at the expense of the bulk of predictions. Log transform brings the distribution closer to normal so every prediction is treated equally.

### 6.2 Target Encoding Must Be Done Inside Cross-Validation

```python
# WRONG — leaks target into features
df['town_te'] = df.groupby('town')['resale_price'].transform('mean')
model.fit(df[features], df['resale_price'])

# CORRECT — OOF encoding prevents leakage
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    # Fit encoder on training fold only
    town_enc = X_tr.groupby('town')['resale_price'].mean()
    X_val['town_te'] = X_val['town'].map(town_enc)
```

If you encode using the full training set, the encoded value for any row contains information about that row's own target — the model will appear to learn well in CV but fail on unseen data.

### 6.3 Out-of-Fold Stacking Prevents Double Leakage

Stacking without OOF is a silent, hard-to-detect source of overfitting:

```
Without OOF:  Base model trains on ALL data → meta features "already know" targets
With OOF:     Each row's meta feature = base model trained on OTHER rows
              → Meta-learner sees only honest out-of-sample base predictions
```

This matters most when the meta-learner is powerful (e.g., LightGBM meta). With a simple Ridge meta, the damage is limited but still present.

### 6.4 High Feature Importance ≠ Useful Feature

`block_enc` had 59% of LightGBM's gain importance in isolation. It was still dropped because:
1. In the presence of `postal` and `planning_area`, it added no new information.
2. It was a target-encoded feature of a high-cardinality categorical — encoding noise was reinforcing itself.

**Rule of thumb:** Always check whether adding or removing a feature changes your CV RMSE. Feature importance rankings describe what the model used, not what it should use.

### 6.5 Ensemble Diversity Requires Structural Difference

Two models are not diverse just because they have different hyperparameters. LightGBM-Huber and LightGBM-MSE are highly correlated (~1.000 OOF correlation) because they share the same leaf-wise growth algorithm. XGBoost uses level-wise growth, which genuinely produces different residuals.

```
OOF correlation matrix (v15):

              LGBM-H  LGBM-M  XGB   SegBlend
LGBM-Huber    1.000   0.998   0.982   0.985
LGBM-MSE      0.998   1.000   0.981   0.984
XGBoost       0.982   0.981   1.000   0.977
Seg-blend     0.985   0.984   0.977   1.000
```

Despite near-identical correlations, the ensemble still outperforms any single base because the small disagreements between bases are precisely where each model is wrong — the stack corrects them.

### 6.6 Non-Negative Meta-Learner Weights Are a Sanity Check

If your meta-learner assigns negative weight to a base model, that base is being used as a "correction term" — the ensemble improves by subtracting it. This is a sign that either (a) the base is actively harmful, or (b) you have too many correlated bases and the meta-learner is overfitting to the OOF predictions.

**Use nnls (non-negative least squares):** forces all base contributions to be additive and positive. If a base gets zero weight, it means the other bases already cover its signal — drop it and save compute time.

### 6.7 Walk-Forward CV Is Not Always the Right Metric

Walk-forward CV simulates "train on past, predict the future." It is essential for deployment but may not match the evaluation metric of your competition or system.

| Situation | Use |
|---|---|
| Kaggle competition with i.i.d. test set | Random KFold (match the test distribution) |
| Production model that gets retrained monthly | Walk-forward (match deployment reality) |
| Detecting if features leak future data | Walk-forward (diagnostic) |
| Comparing models fairly when test is temporal | Walk-forward as primary |

The mistake in v12–v13 was optimising exclusively for walk-forward ($29.8k) when the leaderboard was i.i.d. (~$22k). Both metrics belong in the dashboard; only one should drive your hyperparameter decisions.

### 6.8 The Pruning Pass Is Not Optional

After v14, the 6-base stack had two bases (HGBT-Quantile, RandomForest) consuming **3,095 seconds** of wall time per full run for a combined meta-weight of 0.060. The v15 pruning pass removed them both. Result: pipeline time dropped from ~174 minutes to ~104 minutes, RMSE was unchanged ($21,831 → $21,832).

**Always audit zero-weight or near-zero-weight components** before freezing an architecture. A base model that does not move your OOF RMSE is dead weight, and its compute cost compounds across every experiment iteration.

### 6.9 Data Bugs Are Catastrophically Cheap to Fix

The `hdb_age` column was labelled as age-at-transaction but was actually age-as-of-2021. Correcting this to `age_when_sold = Tranc_Year - year_completed` took five lines of code and improved the model's understanding of lease decay — the mechanism most directly linked to HDB resale pricing rules. Small data corrections often outperform architectural changes.

### 6.10 Checkpoint Your RMSE at Every Feature Engineering Stage

```
Stage                    RMSE
Raw features             $57,198
+ OOF target encoding    $50,728   ← +$6,470 gain
+ remaining_lease        $49,658   ← +$1,070 gain
+ ordinal encoding       $50,403   ← regression: ordinal added noise
+ LightGBM (tuned)       $22,963   ← model > features at this stage
```

Without checkpoints you cannot identify which feature engineering step caused a regression. Run your baseline model after every meaningful feature change — the 5-minute overhead per checkpoint saves hours of debugging.

---

## 7. Appendix — Final Stack Summary

### Architecture

```
Training Data (149,127 rows)
         │
         ▼
Feature Engineering (~55 features)
  ├── Location:  postal_te, town_te, planning_area, dist_to_cbd, knn_logprice
  ├── Physical:  floor_area_sqm, storey_pct, flat_type_ord
  ├── Lease:     remaining_lease_at_sale, age_when_sold, months_since_2012
  ├── Temporal:  Tranc_Year, lag_3m, lag_6m, lag_12m
  └── Derived:   lease_value, geo_cluster_te, storey_band_town_enc
         │
         ▼
5-Fold KFold CV  →  Out-of-Fold Predictions (OOF)
  ├── Base 1: LGBM-Huber (3-seed avg)   OOF RMSE: $22,042
  ├── Base 2: LGBM-MSE   (3-seed avg)   OOF RMSE: $22,012
  ├── Base 3: XGBoost                   OOF RMSE: $22,472
  └── Base 4: Segment-blend LGBM        OOF RMSE: $22,379
         │
         ▼
Meta-Learner: nnls LinearRegression
  Weights: [0.3251, 0.2322, 0.1349, 0.3094] + intercept −0.0196
         │
         ▼
Stack OOF RMSE: $21,832 (0.0473 log)
         │
         ▼
100% Refit → Submission CSV (16,735 rows)
```

### Version History Quick Reference

| Version | Highlight | RMSE (SGD) | Status |
|---|---|---|---|
| V1 | First GBM pipeline | $22,963 | Foundation |
| V2 | OOF stacking introduced | $22,903 | — |
| V3 | Walk-forward CV | $22,740 | — |
| V4 | Bayesian TE + macro index | $23,040 | Regression |
| V5 | Drop LOF, nnls meta | $22,434 | Recovery |
| V6 | Feature pruning, interactions | $22,128 | — |
| **V7** | **lease_per_sqm, PCA** | **$22,094** | **Previous best** |
| V8 | block_enc over-encoding | $22,733 | Regression |
| V9 | Fix v8, drop noise | $22,277 | Recovery |
| V10 | KNN + lag features | ~$24k (WF) | Curriculum build |
| V11 | Walk-forward stacking | ~$22k | — |
| V12 | Native cats, ppsm diversity | $29,814 (WF) | Honest WF eval |
| V13 | XGBoost + address TE | ~$29k (WF) | — |
| V14 | Return to v7 + seeded LGBM | $21,831 | Near-best |
| **V15** | **Pruned 4-base stack** | **$21,832** | **Final submission ✓** |

---

*Post-mortem compiled from v1–v15 notebooks. Final submission: `20260430_21832_likhong.csv`.*
