# PCA Exploration

> Unsupervised PCA experiment on 55 numerical HDB features reveals genuinely high-dimensional data — 22 components needed for 80% variance — with floor area and location as the dominant axes of variation.

## Overview

This notebook applies Principal Component Analysis to the HDB resale dataset to answer two questions: *How many independent dimensions does the data truly have?* and *Which features drive the most variation across flats?*

Working with all 55 numerical columns from the 150,634-row training set, the notebook imputes missing values with medians, standardises each feature to unit variance, then fits a full PCA. The resulting scree plot, 2D scatter views, and loading charts give an interpretable picture of the data's structure before any modelling.

The headline finding is that the dataset is **genuinely high-dimensional**: you need 22 principal components to explain 80% of the variance, and 33 for 95%. This rules out aggressive dimensionality reduction and confirms that the full feature set is carrying non-redundant information — a useful sanity check before building ensemble models.

## Section 1 — Select Numerical Features

PCA operates only on numbers, so the 55 numerical columns are selected after dropping `id`, `resale_price` (the target), and `Tranc_YearMonth` (a date string). These 55 features span flat physical attributes (floor area, storey, age), block-level statistics (unit mix sold/rental), and proximity features (distances to MRT, hawker centres, malls, primary and secondary schools).

```python
DROP = ['id', 'resale_price', 'Tranc_YearMonth']
num_cols = [c for c in train.select_dtypes(include='number').columns
            if c not in DROP]
# → 55 features
```

## Section 2 — Impute & Scale

PCA cannot handle missing values, and large-scale features (e.g., latitude ~1.3 vs. mall count ~5) would dominate without standardisation. Median imputation followed by `StandardScaler` puts every feature on equal footing.

```python
imputer  = SimpleImputer(strategy='median')
X_imp    = imputer.fit_transform(X_raw)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)
# → 150,634 rows × 55 features, zero-mean unit-variance
```

## Section 3 — Scree Plot

| Variance threshold | PCs required |
|---|---|
| 80% | **22** |
| 90% | **28** |
| 95% | **33** |

The first PC explains roughly 19–20% of variance; by PC 3 the cumulative total reaches ~35.6%. The "elbow" in the scree plot is shallow and drawn out, a hallmark of datasets where many features contribute independently. Contrast this with a dataset like MNIST where the top 2–3 PCs often capture 50%+.

**Implication for modelling:** there is no shortcut dimension reduction here. Dropping to 10–15 PCs would discard over 40% of the information. Tree-based models that use all features directly are better suited than PCA-preprocessed linear models.

## Section 4 — 2D Scatter: PC1 vs PC2

Projecting 8,000 randomly sampled flats onto PC1 and PC2 and colouring by resale price produces a visible gradient: higher-priced flats cluster toward higher PC1 values, confirming that PC1 is loosely correlated with price. The town-coloured version shows partial clustering by location, especially for towns like Bishan and Bukit Timah, which separate from mass-market towns along PC2. The overlap is substantial, however, meaning neither PC alone is a clean price or location proxy.

## Section 5 — Feature Loadings

The loading chart reveals the semantic meaning of each principal component:

- **PC1 ("size & amenity density")** — dominated by `floor_area_sqm`, `floor_area_sqft`, and unit-mix sold counts (positively), with MRT and hawker distance loading negatively (shorter distance = higher PC1).
- **PC2 ("geographic latitude")** — latitude, longitude, MRT latitude/longitude, and school coordinates contribute most, capturing the north–south and east–west spread across Singapore.

```python
loadings = pd.DataFrame(
    pca2.components_.T, index=num_cols, columns=['PC1', 'PC2']
).sort_values('PC1', ascending=False)
```

## Section 6 — Interpretation Summary

| Observation | Implication |
|---|---|
| 22 PCs for 80% variance | Data is genuinely high-dimensional; avoid aggressive PCA compression |
| PC1 ≈ size + amenity density | Confirms floor_area_sqm as the #1 raw predictor from EDA |
| PC2 ≈ geography | Location encoding (town, MRT distance) is the next major axis |
| Town clusters in 2D scatter | Location features carry real signal, justifying geographic feature engineering |
| PC1+PC2+PC3 = 35.6% | Even the top 3 PCs leave most variance unexplained |

### Limitations
- PCA only sees 55 numerical columns; categorical features (town, flat_type, model) are excluded
- PCA maximises **variance**, not **price predictability** — a feature can be important for regression yet have low PC loading
- This is exploratory; PCA features are not used as model inputs in subsequent notebooks

## Key Takeaways

- The HDB dataset is **high-dimensional with no dominant shortcut**: 22 components are needed to reach 80% explained variance, ruling out simple PCA compression.
- **Floor area is the single strongest signal** in PC1, consistent with every prior EDA step.
- **Geography forms a distinct second axis** (PC2), validating the investment in location-based feature engineering (MRT distances, town encoding, lat/lon).
- The 2D scatter shows **price gradient is visible but noisy** — linear methods on raw PCs would lose significant predictive power compared to full-feature trees.
- PCA here is diagnostic, not prescriptive: it confirms the feature set is information-rich and that ensemble models should use all features rather than a compressed representation.
