import os
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow

# Always store mlflow.db at the project root regardless of where the notebook runs from
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_TRACKING_URI  = f"sqlite:///{_PROJECT_ROOT}/mlflow.db"
mlflow.set_tracking_uri(_TRACKING_URI)

from src.features.feature_engineering import (
    engineer_features, DROP_COLS, SOLD_COLS, RENTAL_COLS
)
from src.features.oof_encoding import apply_global_oof_encodings
from src.models.stacker import run_stacking, retrain_full_data
from src.models.meta import ridge_blend


def _flatten_params(config):
    """Flatten nested CONFIG for MLflow param logging (string values only)."""
    flat = {'experiment_name': config['experiment_name']}
    for m in config['models'].get('active', []):
        for k, v in config['models'][m].items():
            flat[f'{m}.{k}'] = str(v)
    flat['cv.n_splits'] = str(config['cv']['n_splits'])
    flat['features.encode_pairs'] = str([p[0] for p in config['features']['encode_pairs']])
    extra = config['features'].get('drop_cols', [])
    if extra:
        flat['features.extra_drop_cols'] = str(extra)
    return flat


def run_pipeline(CONFIG):
    """
    End-to-end training pipeline driven entirely by CONFIG.

    CONFIG structure:
        experiment_name  — string label (also used as MLflow experiment name)
        data:
            train_path   — path to train.csv
            test_path    — path to test.csv
        features:
            encode_pairs — list of (group_col, price_col, min_count) for OOF encodings
            drop_cols    — (optional) additional columns to drop from X before training
        models:
            active       — list of model names to include: 'xgb', 'lgbm', 'et', 'catboost'
            xgb          — dict of XGBRegressor params
            lgbm         — dict of LGBMRegressor params
            et           — dict of ExtraTreesRegressor params
            catboost     — dict of CatBoostRegressor params
            ridge_alphas — list of alpha values for Ridge meta-learner sweep
        cv:
            n_splits     — number of folds (default 5)
            random_state — random seed (default 42)
        output:
            submission_path          — OOF-averaged submission CSV (always saved)
            fulldata_submission_path — (optional) 100% retrain submission CSV
            sample_path              — (optional) path to sample_sub_reg.csv for row ordering

    Returns dict: oof_rmse, weights, best_alpha, fold_scores
    """
    mlflow.set_experiment(CONFIG['experiment_name'])

    with mlflow.start_run(run_name=CONFIG['experiment_name']):
        print(f'\n{"=" * 60}')
        print(f'Experiment: {CONFIG["experiment_name"]}')
        print(f'{"=" * 60}')

        # 1. Load raw data
        train_raw = pd.read_csv(CONFIG['data']['train_path'], low_memory=False)
        test_raw  = pd.read_csv(CONFIG['data']['test_path'],  low_memory=False)
        print(f'Loaded — Train: {train_raw.shape}  |  Test: {test_raw.shape}')

        # 2. Feature engineering
        street_freq_map              = train_raw['street_name'].value_counts().to_dict()
        train_fe, train_caps         = engineer_features(train_raw, street_freq_map=street_freq_map)
        test_fe, _                   = engineer_features(
            test_raw, amenity_caps=train_caps, street_freq_map=street_freq_map
        )
        print(f'After engineering — Train: {train_fe.shape}  |  Test: {test_fe.shape}')

        # 3. Global OOF encodings (train: OOF cross-validated; test: full-training-set maps)
        y                = train_raw['resale_price'].values
        global_price_med = float(np.median(y))
        encode_pairs     = CONFIG['features']['encode_pairs']
        n_splits         = CONFIG['cv'].get('n_splits', 5)
        cv_seed          = CONFIG['cv'].get('random_state', 42)

        full_maps = apply_global_oof_encodings(
            train_fe, y, test_fe, encode_pairs,
            global_price_med, n_splits=n_splits, random_state=cv_seed
        )
        print(f'OOF encodings applied: {[p[1] for p in encode_pairs]}')

        # 4. Build feature matrices X and X_test
        always_drop = (
            ['id', 'resale_price', 'postal', 'block', 'town_flat_type', 'num_rooms']
            + DROP_COLS + SOLD_COLS + RENTAL_COLS
        )
        extra_drop = CONFIG['features'].get('drop_cols', [])
        DROP_ALL   = always_drop + extra_drop

        X      = train_fe.drop(columns=DROP_ALL, errors='ignore')
        X_test = test_fe.drop(
            columns=[c for c in DROP_ALL if c != 'resale_price'], errors='ignore'
        )
        X_test = X_test.reindex(columns=X.columns, fill_value=0)

        for col in X.select_dtypes(include='object').columns:
            X[col]      = X[col].astype(str)
            X_test[col] = X_test[col].astype(str)

        print(f'Feature matrix: {X.shape[1]} columns')
        assert X_test.shape[1] == X.shape[1], 'Column mismatch between X and X_test'

        # 5. 5-fold stacking (per-fold OOF re-encoding inside each fold)
        print('\nRunning 5-fold stacking...')
        stack = run_stacking(
            X, y, X_test, train_fe, test_fe, CONFIG,
            n_splits=n_splits, random_state=cv_seed
        )

        # 6. Ridge meta-blend
        ridge_alphas = CONFIG['models'].get('ridge_alphas', [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        blend = ridge_blend(stack['oof_preds'], y, stack['test_preds'], ridge_alphas)

        # 7. Save submission CSV
        out_path = CONFIG['output']['submission_path']
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        sample_path = CONFIG['output'].get('sample_path')
        if sample_path and os.path.exists(sample_path):
            sample_sub = pd.read_csv(sample_path)
            sub = (
                pd.DataFrame({'Id': test_raw['id'], 'Predicted': blend['final_test']})
                .set_index('Id')
                .reindex(sample_sub['Id'])
                .reset_index()
            )
        else:
            sub = pd.DataFrame({'Id': test_raw['id'], 'Predicted': blend['final_test']})
        sub.to_csv(out_path, index=False)
        print(f'\nOOF submission saved: {out_path}')

        # 7b. Optional 100% data retrain
        fulldata_path = CONFIG['output'].get('fulldata_submission_path')
        if fulldata_path:
            full_test_preds = retrain_full_data(
                X, y, X_test, train_fe, encode_pairs, full_maps,
                global_price_med, CONFIG
            )
            # Column order must match how Ridge was trained — same as oof_preds key order
            model_order  = list(stack['oof_preds'].keys())
            meta_X_full  = np.column_stack([full_test_preds[m] for m in model_order])
            final_pred_full = np.expm1(blend['model'].predict(meta_X_full))

            os.makedirs(os.path.dirname(os.path.abspath(fulldata_path)), exist_ok=True)
            if sample_path and os.path.exists(sample_path):
                sub_full = (
                    pd.DataFrame({'Id': test_raw['id'], 'Predicted': final_pred_full})
                    .set_index('Id')
                    .reindex(sample_sub['Id'])
                    .reset_index()
                )
            else:
                sub_full = pd.DataFrame(
                    {'Id': test_raw['id'], 'Predicted': final_pred_full}
                )
            sub_full.to_csv(fulldata_path, index=False)
            print(f'Full-data submission saved: {fulldata_path}')

        # 8. MLflow logging
        mlflow.log_params(_flatten_params(CONFIG))
        mlflow.log_metric('oof_rmse', blend['oof_rmse'])
        mlflow.log_metric('best_ridge_alpha', float(blend['best_alpha']))
        for m, scores in stack['fold_scores'].items():
            mlflow.log_metric(f'{m}_oof_rmse', float(np.mean(scores)))
        for m, w in blend['weights'].items():
            mlflow.log_metric(f'{m}_weight', w)
        if fulldata_path:
            mlflow.log_param('fulldata_submission', fulldata_path)

        print(f'\n{"=" * 60}')
        print(f'OOF RMSE:      ${blend["oof_rmse"]:,.0f}')
        print(f'Ridge weights: {blend["weights"]}')
        print(f'Best alpha:    {blend["best_alpha"]}')
        print(f'MLflow run logged under experiment "{CONFIG["experiment_name"]}"')
        print(f'{"=" * 60}')

    return {
        'oof_rmse':    blend['oof_rmse'],
        'weights':     blend['weights'],
        'best_alpha':  blend['best_alpha'],
        'fold_scores': stack['fold_scores'],
    }
