# hype_pipeline.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import sparse

# sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

# boosters
import xgboost as xgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import joblib


# --------------------------- config ---------------------------

@dataclass
class ColumnsConfig:
    num_cols: List[str]
    cat_cols: List[str]
    datetime_cols: Optional[List[str]] = None


# ----------------------- utilities ----------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _to_dense_if_needed(X):
    return X.toarray() if hasattr(X, "toarray") else X


def _missing(p: str) -> str:
    return f"Required artifact not found: {p}"


# ---------------------- main stacker --------------------------

class HypeStacker(BaseEstimator, RegressorMixin):
    """
    End-to-end stacked model:
      - Preprocessing: StandardScaler (numeric) + OneHotEncoder (categorical)
      - Base models: XGB, LGBM, CatBoost, Ridge, KNN, AdaBoost, GBR
      - Meta model: XGBRegressor

    Artifacts (under artifacts_dir):
      encoder.joblib, scaler.joblib,
      xgb.json, lgbm.txt, catboost.json,
      ridge_.joblib, knn_.joblib, adaboost_.joblib, gbr_.joblib, meta_.joblib

    Notes
    -----
    • We avoid pickling fitted estimators inside the top-level object. Instead,
      we save them as files and (re)load them when needed. This keeps
      model.joblib tiny and resilient to version drift.
    • predict() will automatically load artifacts if they aren't already loaded.
    """

    def __init__(
        self,
        cols: ColumnsConfig,
        random_state: int = 42,
        test_size: float = 0.2,
        artifacts_dir: str = "artifacts",
    ):
        self.cols = cols
        self.random_state = random_state
        self.test_size = test_size
        self.artifacts_dir = artifacts_dir

        # preprocessors
        self.encoder_: Optional[OneHotEncoder] = None
        self.scaler_: Optional[StandardScaler] = None

        # base models (wrappers)
        self.xgb_: Optional[XGBRegressor] = None
        self.lgbm_: Optional[LGBMRegressor] = None
        self.catboost_: Optional[CatBoostRegressor] = None
        self.ridge_: Optional[Ridge] = None
        self.knn_: Optional[KNeighborsRegressor] = None
        self.adaboost_: Optional[AdaBoostRegressor] = None
        self.gbr_: Optional[GradientBoostingRegressor] = None

        # boosters for safe prediction
        self.xgb_booster_: Optional[xgb.Booster] = None
        self.lgbm_booster_: Optional[Any] = None  # lightgbm.basic.Booster

        # meta model
        self.meta_: Optional[XGBRegressor] = None

        # catboost column bookkeeping
        self.cb_cols_: Optional[List[str]] = None
        self.cb_cat_idx_: Optional[List[int]] = None

    # ---------------- preprocessing ----------------

    def _fit_preprocessors(self, X: pd.DataFrame):
        self.encoder_ = OneHotEncoder(handle_unknown="ignore")
        self.scaler_ = StandardScaler()

        self.encoder_.fit(X[self.cols.cat_cols])
        self.scaler_.fit(X[self.cols.num_cols])

    def _transform_encoded(self, X: pd.DataFrame):
        if self.scaler_ is None or self.encoder_ is None:
            raise RuntimeError(
                "Preprocessors are not loaded. Call load_artifacts() or fit() first."
            )
        X_num = self.scaler_.transform(X[self.cols.num_cols]).astype(np.float32)
        X_cat = self.encoder_.transform(X[self.cols.cat_cols]).astype(np.float32)
        return sparse.hstack([X_num, X_cat]).tocsr()

    def _drop_dt_for_catboost(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.cols.datetime_cols:
            drop_cols = [c for c in self.cols.datetime_cols if c in X.columns]
            return X.drop(columns=drop_cols, errors="ignore")
        return X

    # ---------------- base models -------------------

    def _init_base_models(self, RSEED: int):
        self.xgb_ = XGBRegressor(
            objective="reg:squarederror",
            colsample_bytree=0.5111,
            gamma=3.6609,
            learning_rate=0.0583,
            max_depth=10,
            n_estimators=266,
            reg_lambda=9.6965,
            subsample=0.5241,
            random_state=RSEED,
        )
        self.lgbm_ = LGBMRegressor(
            subsample=0.8,
            reg_lambda=1.0,
            reg_alpha=1.0,
            num_leaves=63,
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.05,
            colsample_bytree=1.0,
            random_state=RSEED,
        )
        self.catboost_ = CatBoostRegressor(
            random_strength=10,
            learning_rate=0.1,
            l2_leaf_reg=9,
            iterations=500,
            depth=8,
            border_count=64,
            bagging_temperature=0.5,
            random_state=RSEED,
            verbose=False,
        )
        self.ridge_ = Ridge(alpha=1.5, random_state=RSEED, solver="sag")
        self.knn_ = KNeighborsRegressor(weights="distance", p=1, n_neighbors=28)
        self.adaboost_ = AdaBoostRegressor(
            estimator=XGBRegressor(max_depth=5),
            learning_rate=0.1,
            n_estimators=50,
            random_state=RSEED,
        )
        self.gbr_ = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=7, subsample=0.8, random_state=RSEED
        )

    def _fit_base_models(self, X_enc, X_raw_cb, y, RSEED):
        self._init_base_models(RSEED)

        # train
        self.xgb_.fit(X_enc, y)
        self.lgbm_.fit(X_enc, y)

        # CatBoost needs raw + categorical indices
        self.catboost_.fit(X_raw_cb, y, cat_features=self.cb_cat_idx_)

        dense = _to_dense_if_needed(X_enc)
        self.ridge_.fit(dense, y)
        self.knn_.fit(X_enc, y)
        self.adaboost_.fit(X_enc, y)
        self.gbr_.fit(X_enc, y)

        # keep boosters for portable prediction
        try:
            self.xgb_booster_ = self.xgb_.get_booster()
        except Exception:
            self.xgb_booster_ = None
        try:
            self.lgbm_booster_ = self.lgbm_.booster_
        except Exception:
            self.lgbm_booster_ = None

    # --------------- safe predict helpers ---------------

    def _xgb_predict_safe(self, X_enc):
        if self.xgb_booster_ is not None:
            # try the fast path
            try:
                return self.xgb_booster_.inplace_predict(X_enc)
            except Exception:
                try:
                    dmat = xgb.DMatrix(X_enc)
                    return self.xgb_booster_.predict(dmat)
                except Exception:
                    pass
        # fallback to wrapper
        return self.xgb_.predict(X_enc)

    def _lgbm_predict_safe(self, X_enc):
        if self.lgbm_booster_ is not None:
            return self.lgbm_booster_.predict(X_enc)
        return self.lgbm_.predict(X_enc)

    # ---------------- stacking layer -------------------

    def _base_predict(self, X_enc, X_raw_cb) -> pd.DataFrame:
        dense = _to_dense_if_needed(X_enc)

        preds = [
            self._xgb_predict_safe(X_enc),
            self.ridge_.predict(dense),
            self.knn_.predict(X_enc),
            self._lgbm_predict_safe(X_enc),
            self.catboost_.predict(X_raw_cb),
            self.adaboost_.predict(X_enc),
            self.gbr_.predict(X_enc),
        ]
        stacked = pd.concat([pd.DataFrame(p) for p in preds], axis=1)
        stacked.columns = [f"hype_model_{i}" for i in range(stacked.shape[1])]
        return stacked

    # ---------------- fit / predict API ----------------

    def fit(self, X: pd.DataFrame, y: pd.Series):
        _ensure_dir(self.artifacts_dir)

        # preprocess on full X (common in stacking workflows)
        self._fit_preprocessors(X)
        X_enc = self._transform_encoded(X)

        # prepare catboost inputs
        X_raw_cb = self._drop_dt_for_catboost(X)
        self.cb_cols_ = list(X_raw_cb.columns)
        self.cb_cat_idx_ = [
            X_raw_cb.columns.get_loc(c) for c in self.cols.cat_cols if c in X_raw_cb.columns
        ]

        # split indices for meta training
        idx = np.arange(len(X))
        tr_idx, te_idx = train_test_split(
            idx, stratify=None, test_size=self.test_size, random_state=self.random_state
        )

        # base models on train split
        self._fit_base_models(X_enc[tr_idx], X_raw_cb.iloc[tr_idx], y.iloc[tr_idx], self.random_state)

        # stacked features
        stacked_tr = self._base_predict(X_enc[tr_idx], X_raw_cb.iloc[tr_idx])

        # meta model
        self.meta_ = XGBRegressor(objective="reg:squarederror", random_state=self.random_state)
        self.meta_.fit(stacked_tr, y.iloc[tr_idx])

        # save artifacts
        self.save_artifacts()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Be forgiving: auto-load artifacts if needed
        if not self.is_ready():
            self.load_artifacts(self.artifacts_dir)

        X_enc = self._transform_encoded(X)
        X_raw_cb = self._drop_dt_for_catboost(X)
        if self.cb_cols_ is not None:
            X_raw_cb = X_raw_cb.reindex(columns=self.cb_cols_, fill_value=np.nan)

        stacked = self._base_predict(X_enc, X_raw_cb)
        return self.meta_.predict(stacked)

    # --------------- artifacts I/O ---------------------

    def is_ready(self) -> bool:
        """Return True if preprocessors + meta are loaded."""
        return all(
            [
                self.encoder_ is not None,
                self.scaler_ is not None,
                self.meta_ is not None,
                # at least one of the base models must be present (sanity)
                any(
                    m is not None
                    for m in [
                        self.xgb_booster_,
                        self.lgbm_booster_,
                        self.catboost_,
                        self.ridge_,
                        self.knn_,
                        self.adaboost_,
                        self.gbr_,
                    ]
                ),
            ]
        )

    def _p(self, name: str) -> str:
        return os.path.join(self.artifacts_dir, name)

    def save_artifacts(self):
        _ensure_dir(self.artifacts_dir)

        # preprocessors
        if self.encoder_ is None or self.scaler_ is None:
            raise RuntimeError("Cannot save artifacts: encoder_/scaler_ are None.")
        joblib.dump(self.encoder_, self._p("encoder.joblib"))
        joblib.dump(self.scaler_, self._p("scaler.joblib"))

        # boosters
        if self.xgb_booster_ is not None:
            self.xgb_booster_.save_model(self._p("xgb.json"))
        if self.lgbm_booster_ is not None:
            self.lgbm_booster_.save_model(self._p("lgbm.txt"))

        # catboost
        if self.catboost_ is not None:
            self.catboost_.save_model(self._p("catboost.json"), format="json")

        # sklearn models
        for name in ["ridge_", "knn_", "adaboost_", "gbr_", "meta_"]:
            model = getattr(self, name)
            if model is not None:
                joblib.dump(model, self._p(f"{name}.joblib"))

    def load_artifacts(self, artifacts_dir: Optional[str] = None):
        if artifacts_dir is not None:
            self.artifacts_dir = artifacts_dir

        # preprocessors
        enc_p = self._p("encoder.joblib")
        sc_p = self._p("scaler.joblib")
        if not os.path.exists(enc_p): raise FileNotFoundError(_missing(enc_p))
        if not os.path.exists(sc_p):  raise FileNotFoundError(_missing(sc_p))
        self.encoder_ = joblib.load(enc_p)
        self.scaler_  = joblib.load(sc_p)

        # boosters
        xgb_p = self._p("xgb.json")
        if not os.path.exists(xgb_p): raise FileNotFoundError(_missing(xgb_p))
        self.xgb_booster_ = xgb.Booster()
        self.xgb_booster_.load_model(xgb_p)

        lgbm_p = self._p("lgbm.txt")
        if not os.path.exists(lgbm_p): raise FileNotFoundError(_missing(lgbm_p))
        import lightgbm as lgb
        self.lgbm_booster_ = lgb.Booster(model_file=lgbm_p)

        # catboost
        cb_p = self._p("catboost.json")
        if not os.path.exists(cb_p): raise FileNotFoundError(_missing(cb_p))
        self.catboost_ = CatBoostRegressor()
        self.catboost_.load_model(cb_p, format="json")

        # sklearn bases + meta
        for name in ["ridge_", "knn_", "adaboost_", "gbr_", "meta_"]:
            p = self._p(f"{name}.joblib")
            if not os.path.exists(p):
                raise FileNotFoundError(_missing(p))
            setattr(self, name, joblib.load(p))
