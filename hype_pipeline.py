# hype_pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse

# _hype models (your tuned versions)
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor


@dataclass
class ColumnsConfig:
    num_cols: List[str]
    cat_cols: List[str]
    datetime_cols: Optional[List[str]] = None


class HypeStacker(BaseEstimator, RegressorMixin):
    """
    End-to-end wrapper that:
      - fits encoder/scaler on training data (num+cat),
      - trains base models (_hype) on encoded features (CatBoost on raw without datetimes),
      - builds out-of-fold base predictions for meta XGB,
      - exposes sklearn-compatible .fit() / .predict().
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

        # base models
        self.xgb_: Optional[XGBRegressor] = None
        self.ridge_: Optional[Ridge] = None
        self.knn_: Optional[KNeighborsRegressor] = None
        self.lgbm_: Optional[LGBMRegressor] = None
        self.catboost_: Optional[CatBoostRegressor] = None
        self.adaboost_: Optional[AdaBoostRegressor] = None
        self.gbr_: Optional[GradientBoostingRegressor] = None

        # meta
        self.meta_: Optional[XGBRegressor] = None

        # CatBoost column tracking
        self.cb_cols_: Optional[List[str]] = None
        self.cb_cat_idx_: Optional[List[int]] = None

    # --------- preprocessing helpers ----------
    def _fit_preprocessors(self, X: pd.DataFrame):
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

    # --------- modeling helpers ----------
    def _fit_base_models(self, X_enc, X_raw_for_cb, y, RSEED, cb_cat_idx):
        # tuned params (_hype)
        self.xgb_ = XGBRegressor(objective='reg:squarederror',
                                 colsample_bytree=0.5111,
                                 gamma=3.6609,
                                 learning_rate=0.0583,
                                 max_depth=10,
                                 n_estimators=266,
                                 reg_lambda=9.6965,
                                 subsample=0.5241,
                                 random_state=RSEED)

        self.ridge_ = Ridge(alpha=1.5, random_state=RSEED, solver="sag")
        self.knn_ = KNeighborsRegressor(weights='distance', p=1, n_neighbors=28)

        self.lgbm_ = LGBMRegressor(subsample=0.8, reg_lambda=1.0, reg_alpha=1.0,
                                   num_leaves=63, n_estimators=300, max_depth=-1,
                                   learning_rate=0.05, colsample_bytree=1.0,
                                   random_state=RSEED)

        self.catboost_ = CatBoostRegressor(random_strength=10, learning_rate=0.1, l2_leaf_reg=9,
                                           iterations=500, depth=8, border_count=64,
                                           bagging_temperature=0.5, random_state=RSEED,
                                           verbose=False)

        self.adaboost_ = AdaBoostRegressor(estimator=XGBRegressor(max_depth=5),
                                           learning_rate=0.1, n_estimators=50, random_state=RSEED)

        self.gbr_ = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                              max_depth=7, subsample=0.8, random_state=RSEED)

        # fit bases
        self.xgb_.fit(X_enc, y)
        self.ridge_.fit(X_enc.toarray(), y)  # dense
        self.knn_.fit(X_enc, y)
        self.lgbm_.fit(X_enc, y)

        # CatBoost: pass cat_features indices, ensure ordered columns
        self.catboost_.fit(X_raw_for_cb, y, cat_features=cb_cat_idx)

        self.adaboost_.fit(X_enc, y)
        self.gbr_.fit(X_enc, y)

    def _base_predict(self, X_enc, X_raw_for_cb) -> pd.DataFrame:
        # ensure same order for CatBoost columns
        X_raw_for_cb = X_raw_for_cb[self.cb_cols_]

        preds = [
            self.xgb_.predict(X_enc),
            self.ridge_.predict(X_enc.toarray()),
            self.knn_.predict(X_enc),
            self.lgbm_.predict(X_enc),
            self.catboost_.predict(X_raw_for_cb),
            self.adaboost_.predict(X_enc),
            self.gbr_.predict(X_enc),
        ]
        return pd.DataFrame(
            np.vstack(preds).T,
            columns=[f"hype_model_{i}" for i in range(len(preds))]
        )

    # --------- sklearn API ----------
    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series):
        # 1) preprocessors
        self._fit_preprocessors(X)

        # 2) transforms
        X_enc_full = self._transform_encoded(X)
        X_raw_cb_full = self._drop_dt_for_catboost(X).copy()

        # keep CB column order and categorical indices
        self.cb_cols_ = list(X_raw_cb_full.columns)
        self.cb_cat_idx_ = [self.cb_cols_.index(c) for c in self.cols.cat_cols if c in self.cb_cols_]

        # coerce numeric columns to numeric (safety)
        for c in self.cols.num_cols:
            if c in X_raw_cb_full.columns:
                X_raw_cb_full[c] = pd.to_numeric(X_raw_cb_full[c], errors="coerce")

        # 3) split for meta OOF
        idx = np.arange(len(X))
        idx_tr, idx_meta = train_test_split(
            idx, stratify=y, test_size=self.test_size, random_state=self.random_state
        )

        # 4) fit bases on train split
        self._fit_base_models(
            X_enc_full[idx_tr],
            X_raw_cb_full.iloc[idx_tr][self.cb_cols_],
            y.iloc[idx_tr],
            self.random_state,
            self.cb_cat_idx_,
        )

        # 5) OOF preds for meta
        stacked_meta = self._base_predict(
            X_enc_full[idx_meta],
            X_raw_cb_full.iloc[idx_meta][self.cb_cols_]
        )

        # 6) meta
        self.meta_ = XGBRegressor(objective='reg:squarederror', random_state=self.random_state)
        self.meta_.fit(stacked_meta, y.iloc[idx_meta])

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = self._transform_encoded(X)
        X_raw_cb = self._drop_dt_for_catboost(X).copy()

        # numeric coercion (safety)
        for c in self.cols.num_cols:
            if c in X_raw_cb.columns:
                X_raw_cb[c] = pd.to_numeric(X_raw_cb[c], errors="coerce")

        # enforce same CatBoost column order
        X_raw_cb = X_raw_cb[self.cb_cols_]

        stacked = self._base_predict(X_enc, X_raw_cb)
        return self.meta_.predict(stacked)
