# Feature Engineering — Full Explanation in Layman Terms

---

## The Big Picture

After our baseline Random Forest scored **$25,871 RMSE**, the question is: can we give the model **better information** to learn from?

> **Feature engineering is the art of creating new columns from existing ones.**
>
> The model can only learn what we show it. If we create smarter, more meaningful columns, it can find better patterns — and make better predictions.

Think of it like this:
- Raw data gives the model GPS coordinates → model has to figure out "this is close to the CBD"
- Engineered data directly gives it `dist_to_cbd` → model immediately knows "closer to CBD = pricier"

We engineer **11 new features** in this notebook.

---

## Why Feature Engineering?

The baseline Random Forest scored $25,871. But several raw columns are:
- **Redundant** — `floor_area_sqft` is just `floor_area_sqm × 10.764`
- **Implicit** — the model has to guess that lat/lon near Raffles Place = expensive
- **Leaking** — `resale_price` itself could be used to create target-encoded features
- **Poorly encoded** — Month 12 and Month 1 look far apart as numbers but are adjacent in a calendar

Feature engineering addresses all four problems.

---

## Step 1 — The `engineer_features()` Function

```python
def engineer_features(df, is_train=True, town_price_map=None):
    df = df.copy()
    ...
    return df, town_price_map
```

### Why wrap everything in a function?

We need to apply the **exact same transformations** to both training data and test data.
If we write the code twice, we risk making it slightly different — and the model will break.

A function ensures consistency:
```
train → engineer_features(train, is_train=True)  → X with new columns
test  → engineer_features(test,  is_train=False) → same new columns
```

The `is_train` parameter matters for one feature — target encoding (see Feature 9 below).

---

## Feature 1 — `remaining_lease`

```python
df['remaining_lease'] = 99 - (df['Tranc_Year'] - df['lease_commence_date'])
```

### What is it?
HDB flats have a **99-year lease**. Once the lease expires, the flat reverts to HDB with no compensation.

```
lease_commence_date = 1985
Tranc_Year = 2023
hdb_age = 2023 - 1985 = 38 years
remaining_lease = 99 - 38 = 61 years remaining
```

### Why is this better than `hdb_age`?

Both contain the same information, but remaining_lease is **more interpretable**:
- A buyer thinks "does this flat still have 60 years left?" — not "is this flat 39 years old?"
- Banks also use remaining lease for loan eligibility (cannot loan if lease < 30 years at end of loan)
- The model sees the same signal but framed in the more meaningful direction

---

## Feature 2 — `dist_to_cbd`

```python
CBD_LAT = 1.2830   # Raffles Place MRT
CBD_LON = 103.8513

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

df['dist_to_cbd'] = haversine(df['latitude'], df['longitude'], CBD_LAT, CBD_LON)
```

### What is the Haversine formula?

GPS coordinates (latitude/longitude) are angles on a sphere (Earth). You cannot calculate straight-line distance with simple subtraction because the Earth is curved.

The Haversine formula accounts for Earth's curvature:

```
Simple subtraction: √((lat2-lat1)² + (lon2-lon1)²) ← WRONG (flat Earth)
Haversine:          accounts for Earth's radius ← CORRECT (real distance in km)
```

### Why `dist_to_cbd` became the #2 feature?

Location is one of the biggest drivers of property price worldwide.
- Before: the model had `latitude` and `longitude` as two separate columns
- After: the model has one direct signal — **kilometres from the CBD**

This jumped to the **2nd most important feature** in XGBoost (21.65%), behind only floor area.

CBD reference point: **Raffles Place MRT** (lat=1.2830, lon=103.8513) — the heart of Singapore's financial district.

---

## Feature 3 — `is_mature_estate`

```python
MATURE_TOWNS = {
    'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT MERAH', 'BUKIT TIMAH',
    'CENTRAL AREA', 'CLEMENTI', 'GEYLANG', 'KALLANG/WHAMPOA',
    'MARINE PARADE', 'PASIR RIS', 'QUEENSTOWN', 'SERANGOON',
    'TAMPINES', 'TOA PAYOH'
}

df['is_mature_estate'] = df['town'].isin(MATURE_TOWNS).astype(int)
```

### What is a mature estate?

HDB officially classifies towns into mature and non-mature:
- **Mature**: older, well-developed towns with established amenities, schools, and MRT access
- **Non-mature**: newer towns still being developed (e.g. Punggol, Tengah)

### Why does this matter?

Mature estates command a **10–20% price premium** on average because:
- More MRT lines, bus routes, amenities already in place
- Established schools and community facilities
- Proven rental demand

Rather than expecting the model to learn this implicitly from `town` alone, we give it a direct binary signal: mature = 1, non-mature = 0.

### `.astype(int)`
Converts `True`/`False` → `1`/`0` so the model can process it numerically.

---

## Feature 4 & 5 — `tranc_month_sin` and `tranc_month_cos`

```python
df['tranc_month_sin'] = np.sin(2 * np.pi * df['Tranc_Month'] / 12)
df['tranc_month_cos'] = np.cos(2 * np.pi * df['Tranc_Month'] / 12)
```

### The problem with month as a number

If we use `Tranc_Month` = 1 to 12 directly, the model treats month like a straight line:
```
Jan=1, Feb=2, ... Dec=12
```

But December (12) and January (1) are only 1 month apart — not 11 apart.
The model sees `12 - 1 = 11` and thinks they are far apart. This is wrong.

### Cyclical encoding: the clock analogy

Think of months as a **clock face**. On a clock, 12 and 1 are adjacent.

We encode this using sine and cosine (two waves that are 90° offset):

```
Month 1  (Jan): sin=0.50,  cos=0.87
Month 3  (Mar): sin=1.00,  cos=0.00
Month 6  (Jun): sin=0.00,  cos=-1.00
Month 9  (Sep): sin=-1.00, cos=0.00
Month 12 (Dec): sin=-0.50, cos=0.87   ← close to Jan (0.50, 0.87) ✓
```

December and January are now numerically similar. The model correctly understands seasonal proximity.

### Why two columns (sin AND cos)?

Sin alone is ambiguous — sin(30°) = sin(150°), meaning March and September would look the same.
Using both sin and cos together uniquely identifies every month on the circle.

---

## Feature 6 — `total_sold`

```python
sold_cols = ['1room_sold','2room_sold','3room_sold','4room_sold',
             '5room_sold','exec_sold','multigen_sold','studio_apartment_sold']

df['total_sold'] = df[sold_cols].sum(axis=1)
```

### What is it?

The dataset has 8 separate columns counting how many units of each flat type were sold in the block.
We collapse all 8 into a single `total_sold` — the total number of units sold in that block.

### Why?

- Reduces 8 correlated columns to 1 (less noise)
- Gives a direct signal for block popularity / turnover rate
- `axis=1` means sum across columns (row by row), not down each column

---

## Feature 7 — `rental_ratio`

```python
rental_cols = ['1room_rental','2room_rental','3room_rental','4room_rental']
df['total_rental'] = df[rental_cols].sum(axis=1)

df['rental_ratio'] = df['total_rental'] / (df['total_units'] + 1)
```

### What is it?

The proportion of units in the block that are being rented out (rather than owner-occupied).

```
total_rental = 20 units rented out
total_units  = 100 units in block
rental_ratio = 20 / 101 = 0.198  (about 20% of units are rental)
```

### Why `+ 1`?

This is called **Laplace smoothing** — it prevents division by zero when `total_units = 0`.

### Why does rental ratio matter?

A block with high rental ratio suggests:
- Investors buying to rent out → often a sign of good location or school catchment
- High demand from tenants → indirectly signals desirability
- Proxy for block-level "investment grade" quality

---

## Feature 8 — `floor_area_per_room`

```python
df['floor_area_per_room'] = df['floor_area_sqm'] / (df['room_count'] + 1)
```

Where `room_count` is derived from `flat_type`:
```python
room_map = {'1 ROOM': 1, '2 ROOM': 2, '3 ROOM': 3, '4 ROOM': 4,
            '5 ROOM': 5, 'EXECUTIVE': 5, 'MULTI-GENERATION': 6}
df['room_count'] = df['flat_type'].map(room_map).fillna(3)
```

### What is it?

Spaciousness per room. Two flats can both be "4-room" but one might be 85 sqm and another 110 sqm.

```
Flat A: 4 ROOM, 85 sqm  → 85 / (4+1) = 17.0 sqm per room  (compact)
Flat B: 4 ROOM, 110 sqm → 110 / (4+1) = 22.0 sqm per room (spacious)
```

A more spacious flat generally commands a premium. This feature captures that directly.

---

## Feature 9 — `town_median_price` (Target Encoding)

```python
if is_train:
    town_price_map = df.groupby('town')['resale_price'].median().to_dict()
    df['town_median_price'] = df['town'].map(town_price_map)
else:
    df['town_median_price'] = df['town'].map(town_price_map)
```

### What is target encoding?

Instead of encoding `town` as a number (Bishan=1, Bedok=2...), we replace each town name with the **median resale price of that town**.

```
Before encoding:
  town = 'BISHAN'     → town_median_price = $550,000

After encoding:
  town = 'QUEENSTOWN' → town_median_price = $620,000
  town = 'WOODLANDS'  → town_median_price = $380,000
```

### Why is this powerful?

The model now directly sees "this flat is in a $620K median town" rather than just seeing the number 3 for Queenstown.

### The leakage warning

> **Data leakage** means using information that would not be available at prediction time.

If we compute the median from the **full training set** (including the row being predicted), we are technically "cheating" — the actual price affects its own prediction.

The correct way is **per-fold target encoding** (compute median from the other 4 folds, never the fold being validated). This is marked as a future improvement in README.md.

### Why `is_train=True/False`?

- Training: compute the map from training prices (`is_train=True`)
- Test: apply the **same map** computed from training — never compute a new map from test data (no prices in test!)

```python
X_fe, town_map = engineer_features(train, is_train=True)
X_test_fe, _   = engineer_features(test, is_train=False, town_price_map=town_map)
```

---

## Feature 10 — `amenity_score`

```python
df['amenity_score'] = (
    (1 / (df['mrt_nearest_distance'] + 1)) +
    (1 / (df['mall_nearest_distance'] + 1)) +
    (1 / (df['hawker_nearest_distance'] + 1))
)
```

### What is it?

A composite convenience score combining three distance columns into one.

### Why inverse distance?

Closer = better, so we take `1 / distance`:
```
MRT 200m away: 1/(200+1) = 0.00499   (very close, high score)
MRT 2km away:  1/(2001)  = 0.00050   (far, low score)
```

Higher `amenity_score` = better access to MRT + mall + hawker centre combined.

### Why `+ 1`?

Prevents division by zero if any distance = 0.

### Why a composite instead of 3 separate features?

Reduces dimensionality — instead of three separate distances that are all correlated with each other, we get one score that captures overall "urban convenience".

---

## The Full Pipeline: Before vs After

```
Before engineer_features():
  - 68 raw features
  - town as text label
  - month as number 1-12
  - lat/lon as raw GPS
  - floor area and room count separate

After engineer_features():
  + remaining_lease     (lease years left — more interpretable than age)
  + dist_to_cbd         (km from Raffles Place — #2 feature in XGBoost)
  + is_mature_estate    (1=mature, 0=non-mature — 10-20% premium signal)
  + tranc_month_sin     (cyclical month — fixes Dec/Jan gap)
  + tranc_month_cos     (cyclical month complement)
  + total_sold          (block turnover — 8 cols collapsed to 1)
  + total_rental        (rental units count)
  + rental_ratio        (rental proportion — desirability proxy)
  + floor_area_per_room (spaciousness per room)
  + town_median_price   (target encoding — direct price signal per town)
  + amenity_score       (composite MRT + mall + hawker convenience)
```

---

## Results: Why Did v2 Score Worse Than Baseline?

| Version | Model | Kaggle RMSE |
|---|---|---|
| v1 | RF Baseline (raw features) | 25,943.38 |
| **v2** | **RF + Feature Engineering** | **27,582.06** ← worse! |

### Why?

Random Forest already captures implicit relationships from raw features:
- It uses `latitude` + `longitude` together and discovers CBD proximity implicitly
- It uses `Tranc_Month` and handles the 12→1 gap reasonably through tree splits
- Target encoding via `town_median_price` introduces slight leakage that can hurt generalisation

**Feature engineering benefits gradient boosting (XGBoost/LightGBM) far more than Random Forest.**

RF's nature — building 100 diverse trees on random subsets — already provides implicit feature interactions. XGBoost learns sequentially and benefits directly from clearer input signals.

This is why we carry the features forward into v3 XGBoost (notebook 04), which scored **22,800.92** — a major improvement over the $25,943 baseline.

---

## Summary

```
Feature engineering is not always about adding more columns.
It's about making information more explicit for the model to use.

Raw GPS coords   → explicit dist_to_cbd
Raw month number → cyclical sin/cos encoding
Raw flat_type    → derived floor_area_per_room
Raw town text    → town_median_price (target encoding)
8 sold columns   → total_sold
4 rental columns → rental_ratio + rental_count
```

The payoff comes in notebook 04 where gradient boosting models fully leverage these features.

→ `04_model_tuning.ipynb`
