import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def ridge_blend(oof_preds, y, test_preds, alphas=None):
    """
    Fit a Ridge meta-learner on stacked OOF predictions to find optimal blend weights.

    oof_preds:  dict of {model_name: oof_array (log-space)}
    y:          raw target (not log-transformed)
    test_preds: dict of {model_name: test_pred_array (log-space)}
    alphas:     list of alpha values to sweep

    Returns dict:
        final_oof   — blended OOF predictions in dollar space
        final_test  — blended test predictions in dollar space
        weights     — {model_name: Ridge coefficient}
        best_alpha  — alpha that minimised OOF RMSE
        oof_rmse    — dollar RMSE on OOF predictions
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    y_log        = np.log1p(y)
    model_names  = list(oof_preds.keys())
    meta_X_train = np.column_stack([oof_preds[m] for m in model_names])
    meta_X_test  = np.column_stack([test_preds[m] for m in model_names])

    col_w  = 12
    header = f'{"alpha":>10}  {"OOF RMSE":>10}' + ''.join(
        f'  {(m + "-coef"):<{col_w}}' for m in model_names
    )
    print('\nRidge alpha sweep:')
    print(header)
    print('-' * len(header))

    best_rmse, best_alpha, best_model = float('inf'), alphas[0], None
    for alpha in alphas:
        ridge    = Ridge(alpha=alpha)
        ridge.fit(meta_X_train, y_log)
        pred_log = ridge.predict(meta_X_train)
        r        = _rmse(y, np.expm1(pred_log))
        coefs    = ''.join(f'  {c:>{col_w}.4f}' for c in ridge.coef_)
        print(f'{alpha:>10.3f}  ${r:>10,.0f}{coefs}')
        if r < best_rmse:
            best_rmse, best_alpha, best_model = r, alpha, ridge

    print(f'\nBest alpha: {best_alpha}  |  OOF RMSE: ${best_rmse:,.0f}')

    final_oof  = np.expm1(best_model.predict(meta_X_train))
    final_test = np.expm1(best_model.predict(meta_X_test))
    weights    = {m: float(c) for m, c in zip(model_names, best_model.coef_)}

    return {
        'final_oof':  final_oof,
        'final_test': final_test,
        'weights':    weights,
        'best_alpha': best_alpha,
        'oof_rmse':   best_rmse,
        'model':      best_model,   # fitted Ridge — reused for 100% retrain predictions
    }
