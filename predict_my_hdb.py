"""
HDB Resale Price Estimator
Run: python predict_my_hdb.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

# ── Load model & metadata ──────────────────────────────────────────────────
MODEL_PATH = 'outputs/models/rf_baseline.pkl'
META_PATH  = 'outputs/models/model_meta.json'

pipeline = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    meta = json.load(f)

num_cols       = meta['num_cols']
cat_cols       = meta['cat_cols']
global_medians = meta['global_medians']
town_medians   = meta['town_medians']

# ── Helpers ────────────────────────────────────────────────────────────────

def print_header():
    print('\n' + '='*55)
    print('       HDB Resale Price Estimator')
    print('       Powered by Random Forest Baseline')
    print('='*55 + '\n')

def ask_choice(prompt, options, show_options=True):
    """Ask user to pick from a numbered list."""
    if show_options:
        for i, opt in enumerate(options, 1):
            print(f'  {i:>2}. {opt}')
    while True:
        raw = input(f'\n{prompt}: ').strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        # Allow typing directly
        matches = [o for o in options if o.upper() == raw.upper()]
        if matches:
            return matches[0]
        print(f'     Please enter a number between 1 and {len(options)}, or type the exact value.')

def ask_float(prompt, min_val=None, max_val=None, default=None):
    """Ask user for a number."""
    hint = f' (default: {default})' if default is not None else ''
    while True:
        raw = input(f'{prompt}{hint}: ').strip()
        if raw == '' and default is not None:
            return float(default)
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f'     Must be at least {min_val}.')
                continue
            if max_val is not None and val > max_val:
                print(f'     Must be at most {max_val}.')
                continue
            return val
        except ValueError:
            print('     Please enter a valid number.')

def ask_int(prompt, min_val=None, max_val=None, default=None):
    return int(ask_float(prompt, min_val, max_val, default))

def get_town_median(col, town):
    """Get median value for a column, using town-specific median if available."""
    town_dict = town_medians.get(col, {})
    return town_dict.get(town, global_medians.get(col, 0))

def storey_range_to_mid(storey_range):
    """Convert '07 TO 09' → mid storey 8."""
    try:
        parts = storey_range.split(' TO ')
        return (int(parts[0]) + int(parts[1])) / 2
    except:
        return 8.0

# ── Main input collection ──────────────────────────────────────────────────

def collect_inputs():
    print_header()
    print('Answer the questions below. Press Enter to accept defaults.\n')

    # ── Town
    print('── LOCATION ──────────────────────────────────────')
    print('Select your town:')
    town = ask_choice('Enter number', meta['towns'])

    # ── Flat type
    print('\nSelect your flat type:')
    flat_type = ask_choice('Enter number', meta['flat_types'])

    # ── Flat model
    print('\nSelect your flat model:')
    flat_model = ask_choice('Enter number', meta['flat_models'])

    # ── Floor area
    print('\n── FLAT DETAILS ──────────────────────────────────')
    floor_area_sqm = ask_float('Floor area (sqm)', min_val=20, max_val=400,
                                default=round(get_town_median('floor_area_sqm', town), 0))

    # ── Storey range
    print('\nSelect your storey range:')
    storey_options = sorted(set(
        s for s in meta['storey_ranges']
        if 'TO' in str(s)
    ))
    storey_range = ask_choice('Enter number', storey_options)
    mid_storey   = storey_range_to_mid(storey_range)
    lower        = int(storey_range.split(' TO ')[0])
    upper        = int(storey_range.split(' TO ')[1])

    # ── Lease
    print('\n── LEASE & AGE ───────────────────────────────────')
    lease_commence_date = ask_int('Lease commence year (year HDB was built)',
                                   min_val=1960, max_val=2024,
                                   default=int(get_town_median('lease_commence_date', town)))

    # ── Transaction date
    print('\n── TRANSACTION DATE ──────────────────────────────')
    current_year  = datetime.now().year
    current_month = datetime.now().month
    tranc_year  = ask_int('Transaction year',  min_val=2012, max_val=2030, default=current_year)
    tranc_month = ask_int('Transaction month (1–12)', min_val=1, max_val=12, default=current_month)

    return {
        'town': town,
        'flat_type': flat_type,
        'flat_model': flat_model,
        'floor_area_sqm': floor_area_sqm,
        'storey_range': storey_range,
        'mid_storey': mid_storey,
        'lower': lower,
        'upper': upper,
        'lease_commence_date': lease_commence_date,
        'Tranc_Year': tranc_year,
        'Tranc_Month': tranc_month,
    }

# ── Build full feature row ─────────────────────────────────────────────────

def build_feature_row(inputs):
    town = inputs['town']
    row  = {}

    # Fill all columns with town-specific medians / global medians as defaults
    for col in num_cols:
        row[col] = get_town_median(col, town)
    for col in cat_cols:
        row[col] = town   # reasonable default for most text cols

    # Override with user inputs
    row['town']                = inputs['town']
    row['flat_type']           = inputs['flat_type']
    row['flat_model']          = inputs['flat_model']
    row['full_flat_type']      = f"{inputs['flat_type']} {inputs['flat_model']}"
    row['floor_area_sqm']      = inputs['floor_area_sqm']
    row['storey_range']        = inputs['storey_range']
    row['mid_storey']          = inputs['mid_storey']
    row['lower']               = inputs['lower']
    row['upper']               = inputs['upper']
    row['mid']                 = inputs['mid_storey']
    row['lease_commence_date'] = inputs['lease_commence_date']
    row['Tranc_Year']          = inputs['Tranc_Year']
    row['Tranc_Month']         = inputs['Tranc_Month']

    # Derive columns
    row['hdb_age']        = inputs['Tranc_Year'] - inputs['lease_commence_date']
    row['year_completed'] = inputs['lease_commence_date'] + 3   # approx

    # Fill zero cols
    for col in ['Hawker_Within_500m','Mall_Within_500m','Hawker_Within_1km',
                'Mall_Within_1km','Hawker_Within_2km','Mall_Within_2km']:
        if col not in row or pd.isna(row.get(col)):
            row[col] = 0

    return row

# ── Predict ────────────────────────────────────────────────────────────────

def predict(row):
    df = pd.DataFrame([row])

    # Ensure all required columns exist
    for col in num_cols:
        if col not in df.columns:
            df[col] = global_medians.get(col, 0)
    for col in cat_cols:
        if col not in df.columns:
            df[col] = 'unknown'

    X = df[num_cols + cat_cols]
    pred = pipeline.predict(X)[0]
    return round(pred, -3)   # round to nearest $1,000

# ── Display result ─────────────────────────────────────────────────────────

def show_result(inputs, predicted_price):
    print('\n' + '='*55)
    print('       ESTIMATED RESALE PRICE')
    print('='*55)
    print(f"""
  Flat          : {inputs['flat_type']} ({inputs['flat_model']})
  Town          : {inputs['town']}
  Floor         : {inputs['storey_range']}
  Floor Area    : {inputs['floor_area_sqm']} sqm
  Lease Start   : {inputs['lease_commence_date']}
  HDB Age       : {inputs['Tranc_Year'] - inputs['lease_commence_date']} years
  Transaction   : {inputs['Tranc_Month']}/{inputs['Tranc_Year']}
""")
    print(f'  ╔═══════════════════════════════════╗')
    print(f'  ║  Estimated Price: SGD {predicted_price:>10,.0f}  ║')
    print(f'  ╚═══════════════════════════════════╝')
    print(f"""
  Confidence range (± model RMSE ~$26K):
    Low  estimate : SGD {predicted_price - 26000:>10,.0f}
    High estimate : SGD {predicted_price + 26000:>10,.0f}

  Note: This is an estimate based on ~150K past transactions.
  Actual price depends on negotiation, condition, and market.
""")
    print('='*55)

# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        inputs = collect_inputs()
        row    = build_feature_row(inputs)
        price  = predict(row)
        show_result(inputs, price)

        again = input('\nEstimate another flat? (y/n): ').strip().lower()
        if again == 'y':
            print()
            inputs = collect_inputs()
            row    = build_feature_row(inputs)
            price  = predict(row)
            show_result(inputs, price)

    except (KeyboardInterrupt, EOFError):
        print('\n\nExited.')
