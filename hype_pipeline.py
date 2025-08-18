# hype_pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import sparse

# XGBoost (we'll use both the sklearn wrapper and the native Booster)
import xgboost as xgb
from xgboost import XGBRegressor

# Sklearn & friends
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# --------------------------------------------------------------------------------------
# Column configuration
# --------------------------------------------------------------------------------------
@dataclass
class ColumnsConfig:
    num_cols: List[str]
    cat_cols: List[str]
    datetime_cols: Optional[List[str]] = None


# --------------------------------------------------------------------------------------
# Stacking pipeline
# --------------------------------------------------------------------------------------
class HypeStacker(BaseEstimator, RegressorMixin):
    """
    End-to-end wrapper that:
      • fits encoder/scaler on training data,
      • trains tuned base models on encoded features
        (CatBoost trains on raw features w/o datetime cols),
      • builds OOF base predictions to train a meta-model (XGB),
      • exposes sklearn-compatible .fit() / .predict().
    """

    def __init__(self,
                 cols: ColumnsConfig,
                 random_state: int = 42,
                 test_size: float = 0.2):
        self.cols = cols
        self.random_state = random_state
        self.test_size = test_size

        # preprocessors
        self.encoder_: Optional[OneHotEncoder] = None
        self.scaler_: Optional[StandardScaler] = None

        # base models (tuned “_hype”)
        self.xgb_: Optional[XGBRegressor] = None
        self.ridge_: Optional[Ridge] = None
        self.knn_: Optional[KNeighborsRegressor] = None
        self.lgbm_: Optional[LGBMRegressor] = None
        self.catboost_: Optional[CatBoostRegressor] = None
        self.adaboost_: Optional[AdaBoostRegressor] = None
        self.gbr_: Optional[GradientBoostingRegressor] = None

        # meta model
        self.meta_: Optional[XGBRegressor] = None

        # CatBoost bookkeeping
        self.cb_cols_: Optional[List[str]] = None
        self.cb_cat_idx_: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _fit_preprocessors(self, X: pd.DataFrame) -> None:
        self.encoder_ = OneHotEncoder(handle_unknown="ignore")
        self.scaler_ = StandardScaler()
        self.encoder_.fit(X[self.cols.cat_cols])
        self.scaler_.fit(X[self.cols.num_cols])

    def _transform_encoded(self, X: pd.DataFrame):
        X_num = self.scaler_.transform(X[self.cols.num_cols]).astype(np.float32)
        X_cat = self.encoder_.transform(X[self.cols.cat_cols]).astype(np.float32)
        return sparse.hstack([X_num, X_cat]).tocsr()

    def _drop_dt_for_catboost(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.cols.datetime_cols:
            return X.drop(columns=[c for c in self.cols.datetime_cols if c in X.columns], errors="ignore")
        return X

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------
    def _fit_base_models(self, X_enc, X_raw_for_cb, y, cb_cat_idx) -> None:
        # Tuned XGB
        self.xgb_ = XGBRegressor(
            objective='reg:squarederror',
            colsample_bytree=0.5111,
            gamma=3.6609,
            learning_rate=0.0583,
            max_depth=10,
            n_estimators=266,
            reg_lambda=9.6965,
            subsample=0.5241,
            random_state=self.random_state
        )

        self.ridge_ = Ridge(alpha=1.5, random_state=self.random_state, solver="sag")
        self.knn_ = KNeighborsRegressor(weights='distance', p=1, n_neighbors=28)

        self.lgbm_ = LGBMRegressor(
            subsample=0.8, reg_lambda=1.0, reg_alpha=1.0,
            num_leaves=63, n_estimators=300, max_depth=-1,
            learning_rate=0.05, colsample_bytree=1.0,
            random_state=self.random_state
        )

        self.catboost_ = CatBoostRegressor(
            random_strength=10, learning_rate=0.1, l2_leaf_reg=9,
            iterations=500, depth=8, border_count=64,
            bagging_temperature=0.5, random_state=self.random_state,
            verbose=False
        )

        self.adaboost_ = AdaBoostRegressor(
            estimator=XGBRegressor(max_depth=5),
            learning_rate=0.1, n_estimators=50, random_state=self.random_state
        )

        self.gbr_ = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=7,
            subsample=0.8, random_state=self.random_state
        )

        # Fit
        self.xgb_.fit(X_enc, y)
        self.ridge_.fit(X_enc.toarray(), y)  # Ridge (sag) needs dense
        self.knn_.fit(X_enc, y)
        self.lgbm_.fit(X_enc, y)
        self.catboost_.fit(X_raw_for_cb, y, cat_features=cb_cat_idx)
        self.adaboost_.fit(X_enc, y)
        self.gbr_.fit(X_enc, y)

    # ------------------------------------------------------------------
    # Safe XGB predict (bypass sklearn wrapper pitfalls)
    # ------------------------------------------------------------------
    def _xgb_predict_safe(self, X_enc):
        """
        Predict with XGBoost while avoiding the sklearn wrapper's get_params()
        introspection that can break across versions.

        Tries Booster.inplace_predict (fast), falls back to Booster.predict(DMatrix),
        and only then to wrapper.predict().
        """
        booster = None
        try:
            booster = self.xgb_.get_booster()  # for XGBRegressor wrapper
        except Exception:
            if isinstance(self.xgb_, xgb.Booster):
                booster = self.xgb_
            else:
                booster = None

        if booster is not None:
            # 1) fastest path
            try:
                return booster.inplace_predict(X_enc)
            except Exception:
                pass
            # 2) robust path
            try:
                dmat = xgb.DMatrix(X_enc)
                return booster.predict(dmat)
            except Exception:
                pass

        # 3) last resort
        return self.xgb_.predict(X_enc)

    # ------------------------------------------------------------------
    # Build stacked features from base models
    # ------------------------------------------------------------------
    def _base_predict(self, X_enc, X_raw_cb) -> pd.DataFrame:
        xgb_pred   = self._xgb_predict_safe(X_enc)
        ridge_pred = self.ridge_.predict(X_enc.toarray() if hasattr(X_enc, "toarray") else X_enc)
        knn_pred   = self.knn_.predict(X_enc)
        lgbm_pred  = self.lgbm_.predict(X_enc)
        cb_pred    = self.catboost_.predict(X_raw_cb)
        ada_pred   = self.adaboost_.predict(X_enc)
        gbr_pred   = self.gbr_.predict(X_enc)

        stacked = pd.concat([
            pd.DataFrame(xgb_pred),
            pd.DataFrame(ridge_pred),
            pd.DataFrame(knn_pred),
            pd.DataFrame(lgbm_pred),
            pd.DataFrame(cb_pred),
            pd.DataFrame(ada_pred),
            pd.DataFrame(gbr_pred),
        ], axis=1)
        stacked.columns = [f"hype_model_{i}" for i in range(stacked.shape[1])]
        return stacked

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Fit preprocessors
        self._fit_preprocessors(X)

        # Cache CatBoost columns (raw, no datetime cols)
        X_raw_full = self._drop_dt_for_catboost(X)
        self.cb_cols_ = list(X_raw_full.columns)
        # cat indices for CatBoost relative to X_raw_full order
        self.cb_cat_idx_ = [self.cb_cols_.index(c) for c in self.cols.cat_cols if c in self.cb_cols_]

        # Encoded matrix
        X_enc_full = self._transform_encoded(X)

        # Train/val split for stacking
        n = len(X)
        idx_all = np.arange(n)
        idx_tr, idx_val = train_test_split(
            idx_all, test_size=self.test_size, random_state=self.random_state
        )

        # Fit base models on the train split
        self._fit_base_models(
            X_enc_full[idx_tr],
            X_raw_full.iloc[idx_tr],
            y.iloc[idx_tr],
            cb_cat_idx=self.cb_cat_idx_
        )

        # Build stacked features on the validation split and fit meta model
        stacked_val = self._base_predict(X_enc_full[idx_val], X_raw_full.iloc[idx_val])

        self.meta_ = XGBRegressor(objective='reg:squarederror', random_state=self.random_state)
        self.meta_.fit(stacked_val, y.iloc[idx_val])

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Transform incoming data
        X_enc = self._transform_encoded(X)
        X_raw_cb = self._drop_dt_for_catboost(X)[self.cb_cols_] if self.cb_cols_ else self._drop_dt_for_catboost(X)

        # Base predictions → stacked features → meta prediction
        stacked = self._base_predict(X_enc, X_raw_cb)
        return self.meta_.predict(stacked)
