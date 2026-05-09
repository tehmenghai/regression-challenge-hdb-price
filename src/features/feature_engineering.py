import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt

DROP_COLS   = ['floor_area_sqft', 'lower', 'upper', 'mid', 'full_flat_type',
               'address', 'Tranc_YearMonth', 'residential', 'year_completed']
SOLD_COLS   = ['1room_sold', '2room_sold', '3room_sold', '4room_sold',
               '5room_sold', 'exec_sold', 'multigen_sold', 'studio_apartment_sold']
RENTAL_COLS = ['1room_rental', '2room_rental', '3room_rental', 'other_room_rental']
FILL_ZERO   = ['Hawker_Within_500m', 'Mall_Within_500m', 'Hawker_Within_1km',
               'Mall_Within_1km', 'Hawker_Within_2km', 'Mall_Within_2km']
MATURE_ESTATES = {
    'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT MERAH', 'BUKIT TIMAH',
    'CENTRAL AREA', 'CLEMENTI', 'GEYLANG', 'KALLANG/WHAMPOA',
    'MARINE PARADE', 'PASIR RIS', 'QUEENSTOWN', 'SERANGOON', 'TAMPINES', 'TOA PAYOH'
}
ROOM_COUNT = {
    '1 ROOM': 1, '2 ROOM': 2, '3 ROOM': 3, '4 ROOM': 4,
    '5 ROOM': 5, 'EXECUTIVE': 5, 'MULTI-GENERATION': 6
}
FLAT_TYPE_RANK = {
    '1 ROOM': 1, '2 ROOM': 2, '3 ROOM': 3, '4 ROOM': 4,
    '5 ROOM': 5, 'EXECUTIVE': 6, 'MULTI-GENERATION': 7
}
CBD_LAT, CBD_LON = 1.2847, 103.8510


def haversine_km(lat, lon, lat2=CBD_LAT, lon2=CBD_LON):
    R = 6371
    lat, lon, lat2, lon2 = map(radians, [lat, lon, lat2, lon2])
    a = sin((lat2 - lat) / 2) ** 2 + cos(lat) * cos(lat2) * sin((lon2 - lon) / 2) ** 2
    return 2 * R * asin(sqrt(a))


def engineer_features(df, amenity_caps=None, street_freq_map=None):
    """
    Build all engineered features on top of raw HDB data.

    amenity_caps:    dict of 99th-pct caps for amenity distances.
                     Pass None for train (caps computed from df); pass train caps for test.
    street_freq_map: dict of street_name -> count from training data.
                     Pass None to compute from df. Always pass train map to test.

    Returns (enriched_df, amenity_caps).
    """
    df = df.copy()

    for c in FILL_ZERO:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    df['remaining_lease']     = 99 - (df['Tranc_Year'] - df['lease_commence_date'])
    df['dist_to_cbd']         = df.apply(
        lambda r: haversine_km(r['Latitude'], r['Longitude']), axis=1
    )
    df['is_mature_estate']    = df['town'].str.upper().isin(MATURE_ESTATES).astype(int)
    df['tranc_month_sin']     = np.sin(2 * np.pi * df['Tranc_Month'] / 12)
    df['tranc_month_cos']     = np.cos(2 * np.pi * df['Tranc_Month'] / 12)
    df['total_sold']          = df[SOLD_COLS].sum(axis=1)
    df['total_rental']        = df[RENTAL_COLS].sum(axis=1)
    df['rental_ratio']        = (
        df['total_rental'] / df['total_dwelling_units'].replace(0, np.nan)
    ).fillna(0)
    df['num_rooms']           = df['flat_type'].str.upper().map(ROOM_COUNT).fillna(4)
    df['floor_area_per_room'] = df['floor_area_sqm'] / df['num_rooms']

    caps = amenity_caps or {}
    new_caps = {}
    for col in ['mrt_nearest_distance', 'Mall_Nearest_Distance', 'Hawker_Nearest_Distance']:
        cap = caps.get(col, df[col].quantile(0.99))
        new_caps[col] = cap
        inv = 1 / df[col].clip(1, cap)
        inv_min, inv_max = inv.min(), inv.max()
        df[f'_inv_{col}'] = (inv - inv_min) / (inv_max - inv_min + 1e-9)
    df['amenity_score'] = (
        df['_inv_mrt_nearest_distance']
        + df['_inv_Mall_Nearest_Distance']
        + df['_inv_Hawker_Nearest_Distance']
    ) / 3
    df.drop(columns=[c for c in df.columns if c.startswith('_inv_')], inplace=True)

    df['dist_x_storey']   = df['dist_to_cbd'] * df['mid_storey']
    df['lease_x_area']    = df['remaining_lease'] * df['floor_area_sqm']
    df['log_dist_to_cbd'] = np.log1p(df['dist_to_cbd'])

    if street_freq_map is None:
        street_freq_map = df['street_name'].value_counts().to_dict()
    df['street_freq'] = df['street_name'].map(street_freq_map).fillna(0)

    df['postal_sector'] = df['postal'].astype(str).str[:4]
    df['block_num']     = pd.to_numeric(
        df['block'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'
    ).fillna(0)

    # Remaining lease non-linearity (v14b)
    df['log_remaining_lease'] = np.log1p(df['remaining_lease'])
    df['lease_below_60']      = (df['remaining_lease'] < 60).astype(int)
    df['lease_x_below60']     = df['remaining_lease'] * df['lease_below_60']

    # Storey × flat_type interaction (v14b)
    df['flat_type_rank']    = df['flat_type'].str.upper().map(FLAT_TYPE_RANK).fillna(3)
    df['storey_x_flattype'] = df['mid_storey'] * df['flat_type_rank']
    df['storey_pct']        = df['mid_storey'] / df['max_floor_lvl'].replace(0, np.nan).fillna(1)

    # Composite group key for (town, flat_type) OOF encoding — dropped from X before training
    df['town_flat_type'] = df['town'].astype(str) + '_' + df['flat_type'].astype(str)

    return df, new_caps
