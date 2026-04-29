import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def oof_group_median(group_series, y_series, n_splits=5, random_state=42):
    """Compute OOF median target encoding for a single group column (no leakage)."""
    groups     = group_series.values
    y          = y_series.values
    encoded    = np.zeros(len(groups))
    global_med = np.median(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, va_idx in kf.split(groups):
        fold_map = {}
        for g, p in zip(groups[tr_idx], y[tr_idx]):
            fold_map.setdefault(g, []).append(p)
        fold_map = {g: np.median(ps) for g, ps in fold_map.items()}
        for i in va_idx:
            encoded[i] = fold_map.get(groups[i], global_med)
    return encoded


def apply_global_oof_encodings(train_fe, y_train, test_fe, encode_pairs,
                                global_price_med, n_splits=5, random_state=42):
    """
    Add OOF-encoded price columns to train_fe and test_fe (both modified in-place).

    train_fe gets OOF encodings (cross-validated, no leakage).
    test_fe gets full-training-set map encodings (test labels unknown — no leakage risk).

    encode_pairs: list of (group_col, price_col, min_count)

    Returns dict of full-training-set group → median maps (useful for 100% retrain).
    """
    full_maps = {}
    y_series  = pd.Series(y_train)

    for group_col, price_col, _ in encode_pairs:
        train_fe[price_col] = oof_group_median(
            train_fe[group_col], y_series, n_splits=n_splits, random_state=random_state
        )
        full_map = (
            pd.DataFrame({'g': train_fe[group_col].values, 'p': y_train})
            .groupby('g')['p']
            .median()
        )
        full_maps[group_col] = full_map
        test_fe[price_col] = test_fe[group_col].map(full_map).fillna(global_price_med)

    return full_maps
