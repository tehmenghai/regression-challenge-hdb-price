# Baseline Model — Full Explanation in Layman Terms

---

## The Big Picture

Before we explain the code, understand what we are trying to do:

> **We want to teach a computer to predict HDB resale prices.**
> 
> We show it 150,000 past transactions (with actual prices).
> It learns the patterns.
> Then we ask it to predict prices for 16,737 new transactions it has never seen.

Think of it like teaching a new property agent:
- You show them 150,000 past deals → they learn what affects price
- You then ask them to estimate the price of a new flat → they give you a number based on what they learned

---

## Step 1 — Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
```

### What is a library?
A library is a **toolbox** someone else built for you. Instead of building a hammer from scratch, you just open the toolbox and use it.

| Library | What it does | Real world analogy |
|---|---|---|
| `pandas` | Handles tables of data (like Excel) | Excel for Python |
| `numpy` | Does fast maths on large numbers | A scientific calculator |
| `matplotlib` | Draws charts and graphs | Microsoft Paint for data |
| `warnings` | Hides annoying warning messages | Turning off notifications |

---

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

`sklearn` (scikit-learn) is the main machine learning toolbox in Python. We are importing specific tools from it:

| Tool | What it does |
|---|---|
| `Pipeline` | Chains multiple steps together so they run in order automatically |
| `ColumnTransformer` | Applies different processing to different columns |
| `StandardScaler` | Rescales numbers so they are all on the same scale |
| `OrdinalEncoder` | Converts text categories into numbers |
| `SimpleImputer` | Fills in missing values |
| `Ridge` | A linear regression model |
| `RandomForestRegressor` | A tree-based model (our main model) |
| `train_test_split` | Splits data into training and validation sets |
| `cross_val_score` | Tests model reliability across multiple splits |
| `KFold` | Defines how to split data for cross-validation |
| `mean_squared_error` | Measures how wrong predictions are (squared) |
| `mean_absolute_error` | Measures average dollar error |
| `r2_score` | Measures how much of price variation the model explains |

---

## Step 2 — Load Data

```python
train = pd.read_csv('../../data/raw/train.csv', low_memory=False)
test  = pd.read_csv('../../data/raw/test.csv',  low_memory=False)
```

### Breaking it down

| Code | Meaning |
|---|---|
| `pd.read_csv(...)` | Open a CSV file and load it into a table (DataFrame) |
| `'../../data/raw/train.csv'` | File path — go up 2 folders, then into data/raw/ |
| `low_memory=False` | Read the entire file at once — prevents dtype warning errors |

### Result
```
train = a table with 150,634 rows and 75 columns
test  = a table with  16,737 rows and 74 columns (no resale_price)
```

---

## Step 3 — Drop Redundant Columns

```python
DROP_COLS = [
    'floor_area_sqft',   # duplicate of floor_area_sqm
    'lower', 'upper', 'mid',  # duplicate of mid_storey
    'full_flat_type',    # duplicate of flat_type + flat_model
    'address',           # duplicate of block + street_name
    'Tranc_YearMonth',   # already split into Tranc_Year + Tranc_Month
]

train = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
test  = test.drop(columns=[c for c in DROP_COLS if c in test.columns])
```

### Why drop them?
These columns contain information that already exists in other columns.
Keeping duplicates confuses the model — it sees the same information twice and thinks it is twice as important.

Example:
```
floor_area_sqm  = 90        ← keep this
floor_area_sqft = 968.76    ← drop this (it's just 90 × 10.764)
```

### The code `[c for c in DROP_COLS if c in train.columns]`
This is called a **list comprehension** — a short way to write a loop.

Long version (same result):
```python
cols_to_drop = []
for c in DROP_COLS:
    if c in train.columns:   # only drop if the column actually exists
        cols_to_drop.append(c)
train = train.drop(columns=cols_to_drop)
```

The `if c in train.columns` check is a safety net — it prevents errors if a column does not exist.

---

## Step 4 — Fill Missing Values

```python
fill_zero_cols = ['Hawker_Within_500m', 'Mall_Within_500m', 'Hawker_Within_1km',
                  'Mall_Within_1km', 'Hawker_Within_2km', 'Mall_Within_2km']

for col in fill_zero_cols:
    train[col] = train[col].fillna(0)
    test[col]  = test[col].fillna(0)
```

### What is `.fillna(0)`?
It finds every empty (NaN) cell in that column and replaces it with 0.

```
Before fillna:          After fillna(0):
Hawker_Within_500m      Hawker_Within_500m
NaN          ───→       0
NaN          ───→       0
1.0                     1.0
NaN          ───→       0
```

### Why 0?
From our EDA validation (ID 88392), we confirmed:
- NaN in `Hawker_Within_500m` always means the nearest hawker is >500m away
- So the count of hawkers within 500m is genuinely **zero**, not unknown

---

## Step 5 — Separate Features and Target

```python
X = train.drop(columns=['id', 'resale_price'])   # features (inputs)
y = train['resale_price']                          # target (answer)
X_test = test.drop(columns=['id'])                 # test features
```

### The analogy
Think of a student doing an exam:

```
X = the exam questions    (floor area, town, storey, flat type...)
y = the answer key        (actual resale price)

X_test = a new exam with no answer key (this is what we predict)
```

The model studies `X` paired with `y` to learn the relationship, then answers `X_test` on its own.

---

## Step 6 — Identify Column Types

```python
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=np.number).columns.tolist()
```

### Why do we separate them?
Numbers and text need **different types of processing**:

| Type | Example columns | Problem | Solution |
|---|---|---|---|
| Numerical | `floor_area_sqm`, `mid_storey` | Different scales (90 sqm vs 632m distance) | Rescale to same range |
| Categorical (text) | `town`, `flat_type`, `flat_model` | Models only understand numbers, not text | Convert text → numbers |

### `select_dtypes(include='object')`
`object` is pandas' way of saying "text columns". This line finds all text columns automatically.

---

## Step 7 — Train / Validation Split

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### What is this doing?
Splitting 150,634 rows into two groups:

```
150,634 rows total
│
├── 120,507 rows → X_train + y_train  (model learns from these)
└──  30,127 rows → X_val   + y_val   (we test the model on these)
```

### Parameters explained

| Parameter | Value | Meaning |
|---|---|---|
| `X, y` | Our features and target | What to split |
| `test_size=0.2` | 0.2 = 20% | Keep 20% aside for validation |
| `random_state=42` | Any fixed number | Makes the random split reproducible — same split every time you run |

### Why 42?
No special reason — it is a convention in data science (from The Hitchhiker's Guide to the Galaxy). Any fixed number works. The important thing is it is **fixed** so results are reproducible.

### Why do we need a validation set?
We cannot use `test.csv` to check our model (no real prices). So we hide 20% of training data from the model, train on the other 80%, then check accuracy on the hidden 20%.

```
train.csv (150K rows)
│
├── 80% → model trains here    (model SEES these)
│
└── 20% → validation set       (model NEVER sees these during training)
                                 ↓
                         We compare predictions here
                         to check model accuracy
```

---

## Step 8 — Preprocessing Pipelines

### What is a Pipeline?
A Pipeline is a **production line** — it processes data step by step in a fixed order.

Like a car assembly line:
```
Raw steel → Cut → Weld → Paint → Finish
Raw data  → Impute → Scale → (model ready)
```

```python
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
```

#### Step 1: SimpleImputer
```python
SimpleImputer(strategy='median')
```

Fills remaining missing values in numerical columns.

| Parameter | Value | Meaning |
|---|---|---|
| `strategy='median'` | Use the median | Fill blanks with the middle value of that column |

Example:
```
mrt_nearest_distance: [200, 500, NaN, 800, 300]
median = 400
After impute:         [200, 500, 400, 800, 300]
```

Why median and not mean?
- Mean is pulled by outliers (one flat 5km from MRT skews the average)
- Median is the middle value — more robust

#### Step 2: StandardScaler
```python
StandardScaler()
```

Rescales all numbers to the same range (mean=0, std=1).

### Why do we need to scale?

Imagine the model is comparing:
```
floor_area_sqm    = 90       (range: 30–300)
mrt_distance      = 632      (range: 50–5000)
Tranc_Year        = 2019     (range: 2012–2022)
```

Without scaling, `Tranc_Year` (2019) looks huge compared to `floor_area_sqm` (90).
The model might think year is more important just because the number is bigger — which is wrong.

After StandardScaler:
```
floor_area_sqm    = 0.3      (all on same scale)
mrt_distance      = -0.1     (all centred around 0)
Tranc_Year        = 0.8      (model can compare fairly)
```

> **Note:** Tree models (Random Forest, XGBoost) do NOT actually need scaling. But it is harmless, and it is required for Ridge regression. We include it so both models use the same pipeline.

---

```python
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
```

#### Step 1: SimpleImputer for text
```python
SimpleImputer(strategy='most_frequent')
```

Fills missing text values with the **most common value** in that column.

Example:
```
flat_model: ['Model A', NaN, 'Improved', 'Model A', NaN]
most_frequent = 'Model A'
After impute: ['Model A', 'Model A', 'Improved', 'Model A', 'Model A']
```

#### Step 2: OrdinalEncoder
```python
OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
```

Converts text categories into numbers — models only understand numbers.

Example:
```
flat_type (before):          flat_type (after):
'1 ROOM'          ───→       0
'2 ROOM'          ───→       1
'3 ROOM'          ───→       2
'4 ROOM'          ───→       3
'5 ROOM'          ───→       4
'EXECUTIVE'       ───→       5
```

| Parameter | Meaning |
|---|---|
| `handle_unknown='use_encoded_value'` | If test data has a new category not seen in training, do not crash |
| `unknown_value=-1` | Assign -1 to any unknown category |

---

```python
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])
```

### What is ColumnTransformer?
It applies **different pipelines to different columns** at the same time.

```
All columns
│
├── Numerical columns → num_pipeline (impute with median → scale)
└── Categorical columns → cat_pipeline (impute with mode → encode)
```

Think of it like a passport control with two lanes:
```
Passengers (data)
│
├── Citizens (numerical) ──→ Fast lane (StandardScaler)
└── Foreigners (categorical) → Processing lane (OrdinalEncoder)
```

---

## Step 9 — Model 1: Ridge Regression

```python
ridge_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=10))
])

ridge_pipeline.fit(X_train, y_train)
ridge_pred = ridge_pipeline.predict(X_val)
```

### What is Ridge Regression?
Ridge is a **linear model** — it draws a straight line through the data.

Simple example (one feature):
```
resale_price = (floor_area × 3,500) + (storey × 2,000) + 200,000
```

It finds the best multipliers (3,500, 2,000, 200,000) to fit the training data.

### `alpha=10`
Alpha controls how much the model is penalised for using large multipliers.
- High alpha → simpler model, less risk of overfitting
- Low alpha → more complex, risks overfitting
- 10 is a reasonable starting default

### `.fit(X_train, y_train)`
This is where the **learning happens**.
The model looks at all 120,507 training rows and figures out the best multipliers.

### `.predict(X_val)`
The model uses what it learned to estimate prices for the 30,127 validation rows — rows it has never seen.

---

## Step 10 — Model 2: Random Forest

```python
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    ))
])
```

### What is a Random Forest?
A Random Forest is a collection of **decision trees** — each tree votes, and the average vote is the final prediction.

### What is a Decision Tree?
A tree that asks yes/no questions to narrow down the price:

```
Is floor_area_sqm > 100?
├── YES → Is town == 'BISHAN'?
│         ├── YES → Predict $650,000
│         └── NO  → Is mid_storey > 15?
│                   ├── YES → Predict $580,000
│                   └── NO  → Predict $520,000
└── NO  → Is flat_type == '3 ROOM'?
          ├── YES → Predict $380,000
          └── NO  → Predict $290,000
```

### Why a "Forest" (many trees)?
One tree can be wrong. 100 trees voting together are much more accurate.

Like asking 100 property agents for a price estimate and averaging their answers — more reliable than asking just one.

### Parameters explained

| Parameter | Value | Meaning |
|---|---|---|
| `n_estimators=100` | 100 trees | Build 100 different decision trees and average their predictions |
| `max_depth=15` | Max 15 levels deep | Limits tree complexity — prevents overfitting |
| `min_samples_leaf=5` | Min 5 rows per leaf | A tree cannot split if fewer than 5 rows remain — prevents overfitting |
| `n_jobs=-1` | Use all CPU cores | Train faster by using all available processors |
| `random_state=42` | Fixed seed | Same forest every time you run the code |

---

## Step 11 — Evaluation Metrics

```python
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
```

### RMSE — Root Mean Squared Error

```
Error for each flat = Actual Price − Predicted Price
RMSE = √(average of all errors²)
```

Example:
```
Flat 1: Actual $500K, Predicted $480K → Error = $20K  → Squared = 400,000,000
Flat 2: Actual $400K, Predicted $450K → Error = $50K  → Squared = 2,500,000,000
Flat 3: Actual $600K, Predicted $590K → Error = $10K  → Squared = 100,000,000

Average squared error = 1,000,000,000
RMSE = √1,000,000,000 = $31,623
```

**RMSE punishes big errors more** (because of squaring). A $50K error hurts 6.25× more than a $20K error.

Our RF RMSE = **$25,871** → on average, predictions are off by ~$26K

### MAE — Mean Absolute Error

```
MAE = average of |Actual − Predicted| across all flats
```

Simpler than RMSE — just the average dollar error, no squaring.

Our RF MAE = **$18,950** → the typical prediction is off by ~$19K

### R² — R-squared

Measures how much of the price variation the model explains.

```
R² = 0.00 → model explains nothing (just predicts the average price for every flat)
R² = 0.50 → model explains half the variation
R² = 1.00 → model predicts perfectly
```

Our RF R² = **0.9672** → model explains 96.7% of why prices differ between flats

---

## Step 12 — Cross-Validation

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    rf_pipeline, X, y,
    cv=kf,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
```

### Why not just use the single validation score?

The single val split could get **lucky or unlucky** depending on which rows end up in the 20%.

Cross-validation runs the test 5 times with different splits:

```
Run 1: Train on folds 2,3,4,5 → Test on fold 1 → RMSE = $25,100
Run 2: Train on folds 1,3,4,5 → Test on fold 2 → RMSE = $26,800
Run 3: Train on folds 1,2,4,5 → Test on fold 3 → RMSE = $25,900
Run 4: Train on folds 1,2,3,5 → Test on fold 4 → RMSE = $27,200
Run 5: Train on folds 1,2,3,4 → Test on fold 5 → RMSE = $26,900

Average = $26,389 ← much more reliable than a single run
```

### Parameters explained

| Parameter | Value | Meaning |
|---|---|---|
| `n_splits=5` | 5 folds | Split data into 5 equal parts |
| `shuffle=True` | Shuffle rows first | Prevents time-ordered bias |
| `scoring='neg_root_mean_squared_error'` | Negative RMSE | sklearn returns negative so higher = better (convention) |
| `n_jobs=-1` | All CPU cores | Run folds in parallel — faster |

### Why `neg_root_mean_squared_error`?
sklearn always maximises the score. Since lower RMSE = better, it uses negative RMSE so the best score is the "highest" (least negative) number.

```python
cv_rmse = -cv_scores   # flip the sign back to positive
```

Our CV RMSE = **$26,389 ± $320** → consistent across all 5 folds, model is stable

---

## Step 13 — Generate Submission

```python
# Retrain on ALL training data (not just 80%)
rf_pipeline.fit(X, y)

test_pred = rf_pipeline.predict(X_test)

submission = pd.DataFrame({
    'id': test['id'],
    'resale_price': test_pred.round(0)
})

submission.to_csv('../../outputs/submissions/sub_v1_rf_baseline.csv', index=False)
```

### Why retrain on ALL data before submitting?
During evaluation we only used 80% for training (to keep 20% for validation).
But for the final submission, we want the model to learn from **as much data as possible**.
So we retrain on the full 150,634 rows before predicting the 16,737 test rows.

### `.round(0)`
Rounds predictions to the nearest dollar (no decimals).
```
477,234.67  ───→  477,235.0
```

### `index=False`
Do not write the row numbers (0, 1, 2...) into the CSV — Kaggle only wants `id` and `resale_price`.

---

## Summary — The Full Pipeline at a Glance

```
Raw train.csv (150K rows)
│
├─ Drop redundant columns
├─ Fill missing values with 0
├─ Split into X (features) and y (target)
├─ Split into 80% train / 20% validation
│
├─ Preprocessing pipeline:
│   ├─ Numerical columns → fill blanks with median → rescale
│   └─ Categorical columns → fill blanks with mode → encode as numbers
│
├─ Ridge Regression → RMSE $58,387  (linear, our floor)
├─ Random Forest   → RMSE $25,871  (tree, our baseline)
├─ Cross-validation → RMSE $26,389 ± $320  (reliable estimate)
│
└─ Retrain on full data → predict test.csv → save submission CSV

Result: sub_v1_rf_baseline.csv (16,737 predictions)
```

---

## What's Next

Our baseline RMSE is **$25,871**. The goal in the next notebooks is to beat this number through:

1. **Feature engineering** — add `dist_to_cbd`, `remaining_lease`, better encode `town`
2. **Log-transform target** — reduce impact of expensive outlier flats
3. **Better model** — XGBoost / LightGBM (faster and more powerful than Random Forest)
4. **Hyperparameter tuning** — find the best settings for the model

→ `03_feature_engineering.ipynb`
