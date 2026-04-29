import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor


def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _build_preprocessor(X):
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    return ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])


def _build_model(name, params):
    if name == 'xgb':
        return XGBRegressor(**params)
    if name == 'lgbm':
        return LGBMRegressor(**params)
    if name == 'et':
        return ExtraTreesRegressor(**params)
    if name == 'catboost':
        return CatBoostRegressor(**params)
    raise ValueError(f'Unknown model name: {name}')


def run_stacking(X, y, X_test, train_fe, test_fe, config, n_splits=5, random_state=42):
    """
    5-fold OOF stacking loop for all active base models.

    Per-fold target encodings (ENCODE_PAIRS) are recomputed inside each fold from
    train_fe.iloc[tr_idx] — never from X_tr — to prevent target leakage when the
    group column was excluded from X via DROP_ALL.

    Returns dict:
        oof_preds   — {model_name: array of shape (n_train,)}
        test_preds  — {model_name: array of shape (n_test,)} averaged across folds
        fold_scores — {model_name: list of per-fold RMSE in dollars}
    """
    encode_pairs = config['features']['encode_pairs']
    model_cfg    = config['models']
    active       = model_cfg.get('active', ['xgb', 'lgbm', 'et', 'catboost'])

    n_train, n_test = len(X), len(X_test)
    oof_preds   = {m: np.zeros(n_train) for m in active}
    test_folds  = {m: np.zeros((n_splits, n_test)) for m in active}
    fold_scores = {m: [] for m in active}

    kf    = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_log = np.log1p(y)

    preprocessor = _build_preprocessor(X)

    col_w = 14
    header = f'{"Fold":>5}' + ''.join(f'  {(m.upper()+"-RMSE"):<{col_w}}' for m in active)
    print(header)
    print('-' * (7 + (col_w + 2) * len(active)))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        y_tr, y_va   = y[tr_idx], y[va_idx]
        y_tr_log     = np.log1p(y_tr)
        global_med_tr = np.median(y_tr)

        # Use train_fe for group col lookups — retains all cols even if dropped from X
        fe_tr = train_fe.iloc[tr_idx]
        fe_va = train_fe.iloc[va_idx]

        for group_col, price_col, min_ct in encode_pairs:
            fold_map = {}
            for g, p in zip(fe_tr[group_col].values, y_tr):
                fold_map.setdefault(g, []).append(p)
            fold_map = {
                g: np.median(ps) for g, ps in fold_map.items() if len(ps) >= min_ct
            }
            # .values strips the original DataFrame index to match X_tr's reset index
            X_tr[price_col] = fe_tr[group_col].map(fold_map).fillna(global_med_tr).values
            X_va[price_col] = fe_va[group_col].map(fold_map).fillna(global_med_tr).values

        row = f'{fold + 1:>5}'
        for m in active:
            params = dict(model_cfg[m])
            pipe   = Pipeline([('pre', preprocessor), ('model', _build_model(m, params))])
            pipe.fit(X_tr, y_tr_log)
            oof_preds[m][va_idx]  = pipe.predict(X_va)
            test_folds[m][fold]   = pipe.predict(X_test)
            fold_rmse = _rmse(y_va, np.expm1(oof_preds[m][va_idx]))
            fold_scores[m].append(fold_rmse)
            row += f'  ${fold_rmse:>{col_w - 1},.0f}'
        print(row)

    print('-' * (7 + (col_w + 2) * len(active)))
    means = ''.join(f'  ${np.mean(fold_scores[m]):>{col_w - 1},.0f}' for m in active)
    print(f'{"Mean":>5}{means}')

    test_preds = {m: test_folds[m].mean(axis=0) for m in active}
    return {
        'oof_preds':   oof_preds,
        'test_preds':  test_preds,
        'fold_scores': fold_scores,
    }


def retrain_full_data(X, y, X_test, train_fe, encode_pairs, full_maps,
                      global_price_med, config):
    """
    Retrain all active base models on 100% of training data.

    Replaces OOF-encoded price columns in X with full-training-set maps so both
    train and test use the same encoding — valid here because Ridge weights are
    already fixed from the OOF step and these predictions are only used for the
    final test submission.

    Returns dict: {model_name: test_predictions (log-space)}
    """
    active    = config['models'].get('active', ['xgb', 'lgbm', 'et', 'catboost'])
    model_cfg = config['models']
    y_log     = np.log1p(y)

    # Replace per-fold OOF price cols with full-training-set maps
    X_full = X.copy()
    for group_col, price_col, _ in encode_pairs:
        X_full[price_col] = (
            train_fe[group_col].map(full_maps[group_col]).fillna(global_price_med).values
        )

    # Fit a fresh preprocessor on all training data, then transform both splits
    preprocessor = _build_preprocessor(X_full)
    preprocessor.fit(X_full)
    Xf_arr = preprocessor.transform(X_full)
    Xt_arr = preprocessor.transform(X_test)

    test_preds = {}
    print('\nRetraining on 100% of data...')
    for m in active:
        model = _build_model(m, dict(model_cfg[m]))
        model.fit(Xf_arr, y_log)
        test_preds[m] = model.predict(Xt_arr)
        print(f'  {m} done')

    return test_preds  # log-space — pass through existing Ridge meta-model
