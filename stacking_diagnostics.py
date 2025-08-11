
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

def get_oof_predictions(models: dict, X, y, n_splits=5, random_state=42, shuffle=True):
    """
    Compute out-of-fold (OOF) predictions for a dict of models.

    Parameters
    ----------
    models : dict[str, estimator]
        Mapping of {name: sklearn-like regressor}.
    X : array-like or DataFrame
    y : array-like

    Returns
    -------
    oof_df : DataFrame (n_samples x n_models)
        OOF predictions for each model.
    fold_preds : dict[str, list[np.ndarray]]
        List of per-fold validation predictions in the original sample order indices.
    rmses : dict[str, float]
        OOF RMSE per model.
    cv_indices : list[tuple[np.ndarray, np.ndarray]]
        List of (train_idx, val_idx) per fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    oof = {name: np.zeros(len(y), dtype=float) for name in models}
    fold_preds = {name: [] for name in models}
    cv_indices = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        cv_indices.append((tr_idx, va_idx))
        X_tr, X_va = (X[tr_idx], X[va_idx]) if not hasattr(X, "iloc") else (X.iloc[tr_idx], X.iloc[va_idx])
        y_tr, y_va = y[tr_idx], y[va_idx]

        for name, est in models.items():
            est_fold = clone(est)
            est_fold.fit(X_tr, y_tr)
            p = est_fold.predict(X_va)
            oof[name][va_idx] = p
            fold_preds[name].append(p)

    oof_df = pd.DataFrame(oof, index=(X.index if hasattr(X, "index") else None))
    rmses = {name: float(np.sqrt(mean_squared_error(y, oof_df[name]))) for name in oof_df.columns}
    return oof_df, fold_preds, rmses, cv_indices

def correlation_matrix(pred_df: pd.DataFrame):
    """Return the Pearson correlation matrix of columns (models)."""
    return pred_df.corr()

def fit_meta_xgb(oof_preds: pd.DataFrame, y, params=None):
    """Fit a regularized XGB meta-model on OOF predictions and return it."""
    default_params = dict(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=200,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        n_jobs=-1,
    )
    if params:
        default_params.update(params)
    meta = XGBRegressor(**default_params)
    meta.fit(oof_preds, y)
    return meta
