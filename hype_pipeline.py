from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import sparse

# Preprocessors
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Sklearn models
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

# Boosted libraries
import xgboost as xgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import joblib


# --------------------------- Config ---------------------------

@dataclass
class ColumnsConfig:
    num_cols: List[str]
    cat_cols: List[str]
    datetime_cols: Optional[List[str]] = None


# ---------------------- Helper utilities ----------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _csr_or_ndarray(X):
    # Some estimators accept CSR, some require dense
    return X


# ---------------------- Main Stacker --------------------------

class HypeStacker(BaseEstimator, RegressorMixin):
    """
    Full stacked regressor:
      - OneHotEncoder + StandardScaler
      - Base models: XGB, LGBM, CatBoost, Ridge, KNN, AdaBoost, GBR
      - Meta model: XGBRegressor

    Artifacts:
      - Saves all base/meta models to ./artifacts/*
      - Uses Booster-level prediction for XGB/LGBM to avoid sklearn-wrapper drift.
    """

    def __init__(self,
                 cols: ColumnsConfig,
                 random_state: int = 42,
                 test_size: float = 0.2,
                 artifacts_dir: str = "artifacts"):
        self.cols = cols
        self.random_state = random_state
        self.test_size = test_size
        self.artifacts_dir = artifacts_dir

        # Preprocessors
        self.encoder_: Optional[OneHotEncoder] = None
        self.scaler_: Optional[StandardScaler] = None

        # Base models (fit-time)
        self.xgb_: Optional[XGBRegressor] = None
        self.lgbm_: Optional[LGBMRegressor] = None
        self.catboost_: Optional[CatBoostRegressor] = None
        self.ridge_: Optional[Ridge] = None
        self.knn_: Optional[KNeighborsRegressor] = None
        self.adaboost_: Optional[AdaBoostRegressor] = None
        self.gbr_: Optional[GradientBoostingRegressor] = None

        # For booster-level prediction after reload
        self.xgb_booster_: Optional[xgb.Booster] = None
        self.lgbm_booster_: Optional[Any] = None  # lightgbm.basic.Booster

        # CatBoost indexing info
        self.cb_cols_: Optional[List[str]] = None
        self.cb_cat_idx_: Optional[List[int]] = None

        # Meta model
        self.meta_: Optional[XGBRegressor] = None

        # Artifact file map populated at save-time
        self._artifact_files_: Dict[str, str] = {}

    # ---------------- Preprocessing ----------------

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
            drop_cols = [c for c in self.cols.datetime_cols if c in X.columns]
            return X.drop(columns=drop_cols, errors="ignore")
        return X

    # --------------- Base models fit ----------------

    def _fit_base_models(self, X_enc, X_raw_cb, y, RSEED):
        # Tuned _hype params you used
        self.xgb_ = XGBRegressor(
            objective='reg:squarederror',
            colsample_bytree=0.5111,
            gamma=3.6609,
            learning_rate=0.0583,
            max_depth=10,
            n_estimators=266,
            reg_lambda=9.6965,
            subsample=0.5241,
            random_state=RSEED
        )
        self.lgbm_ = LGBMRegressor(
            subsample=0.8, reg_lambda=1.0, reg_alpha=1.0,
            num_leaves=63, n_estimators=300, max_depth=-1,
            learning_rate=0.05, colsample_bytree=1.0,
            random_state=RSEED
        )
        self.catboost_ = CatBoostRegressor(
            random_strength=10, learning_rate=0.1, l2_leaf_reg=9,
            iterations=500, depth=8, border_count=64, bagging_temperature=0.5,
            random_state=RSEED, verbose=False
        )
        self.ridge_ = Ridge(alpha=1.5, random_state=RSEED, solver="sag")
        self.knn_ = KNeighborsRegressor(weights='distance', p=1, n_neighbors=28)
        self.adaboost_ = AdaBoostRegressor(
            estimator=XGBRegressor(max_depth=5),
            learning_rate=0.1, n_estimators=50, random_state=RSEED
        )
        self.gbr_ = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=7,
            subsample=0.8, random_state=RSEED
        )

        # Fit
        self.xgb_.fit(X_enc, y)
        self.lgbm_.fit(X_enc, y)
        # for CatBoost: indicate categorical indices based on raw dataframe
        self.catboost_.fit(X_raw_cb, y, cat_features=self.cb_cat_idx_)

        # The rest on encoded features
        dense = X_enc.toarray() if hasattr(X_enc, "toarray") else X_enc
        self.ridge_.fit(dense, y)
        self.knn_.fit(X_enc, y)
        self.adaboost_.fit(X_enc, y)
        self.gbr_.fit(X_enc, y)

        # Keep boosters for portable prediction
        try:
            self.xgb_booster_ = self.xgb_.get_booster()
        except Exception:
            self.xgb_booster_ = None
        try:
            self.lgbm_booster_ = self.lgbm_.booster_
        except Exception:
            self.lgbm_booster_ = None

    # ----------------- Predict helpers -----------------

    def _xgb_predict_safe(self, X_enc):
        # Prefer Booster to avoid sklearn param introspection issues
        booster = self.xgb_booster_
        if booster is not None:
            try:
                return booster.inplace_predict(X_enc)
            except Exception:
                try:
                    dmat = xgb.DMatrix(X_enc)
                    return booster.predict(dmat)
                except Exception:
                    pass
        return self.xgb_.predict(X_enc)

    def _lgbm_predict_safe(self, X_enc):
        booster = self.lgbm_booster_
        if booster is not None:
            # Booster expects numpy/csr; LightGBM handles csr directly
            return booster.predict(X_enc)
        return self.lgbm_.predict(X_enc)

    # ----------------- Stacking -----------------

    def _base_predict(self, X_enc, X_raw_cb) -> pd.DataFrame:
        dense = X_enc.toarray() if hasattr(X_enc, "toarray") else X_enc

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

    # ----------------- Fit / Predict API -----------------

    def fit(self, X: pd.DataFrame, y: pd.Series):
        _ensure_dir(self.artifacts_dir)

        # Preprocess
        self._fit_preprocessors(X)
        X_enc = self._transform_encoded(X)

        # CatBoost: raw df without datetime cols; track column order & cat idx
        X_raw_cb = self._drop_dt_for_catboost(X)
        self.cb_cols_ = list(X_raw_cb.columns)
        # categorical indices for CatBoost in the *raw* dataframe
        self.cb_cat_idx_ = [
            X_raw_cb.columns.get_loc(c) for c in self.cols.cat_cols if c in X_raw_cb.columns
        ]

        # Split for meta training (holdout to avoid leakage)
        idx = np.arange(len(X))
        tr_idx, te_idx = train_test_split(
            idx, stratify=None, test_size=self.test_size, random_state=self.random_state
        )

        # Fit base models on train split
        self._fit_base_models(X_enc[tr_idx], X_raw_cb.iloc[tr_idx], y.iloc[tr_idx], self.random_state)

        # Build stacked features
        stacked_tr = self._base_predict(X_enc[tr_idx], X_raw_cb.iloc[tr_idx])
        stacked_te = self._base_predict(X_enc[te_idx], X_raw_cb.iloc[te_idx])

        # Meta model
        self.meta_ = XGBRegressor(
            objective='reg:squarederror',
            random_state=self.random_state
        )
        self.meta_.fit(stacked_tr, y.iloc[tr_idx])

        # Save artifacts (all models + boosters)
        self._save_artifacts()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Transform
        X_enc = self._transform_encoded(X)
        # Prepare CatBoost raw with same columns/ordering as during fit
        X_raw_cb = self._drop_dt_for_catboost(X)
        if self.cb_cols_ is not None:
            X_raw_cb = X_raw_cb.reindex(columns=self.cb_cols_, fill_value=np.nan)

        # Base predictions → meta
        stacked = self._base_predict(X_enc, X_raw_cb)
        return self.meta_.predict(stacked)

    # ----------------- Artifact save/load -----------------

    def _save_artifacts(self):
        """
        Persist every model to disk under artifacts_dir, replacing in-memory models
        with file references via __getstate__/__setstate__.
        """
        files = {}

        # XGBoost: save booster json
        if self.xgb_booster_ is not None:
            path = os.path.join(self.artifacts_dir, "xgb.json")
            self.xgb_booster_.save_model(path)
            files["xgb_booster"] = path

        # LightGBM: save txt
        if self.lgbm_booster_ is not None:
            path = os.path.join(self.artifacts_dir, "lgbm.txt")
            self.lgbm_booster_.save_model(path)
            files["lgbm_booster"] = path

        # CatBoost: save model json
        if self.catboost_ is not None:
            path = os.path.join(self.artifacts_dir, "catboost.json")
            self.catboost_.save_model(path, format="json")
            files["catboost"] = path

        # Sklearn models: save joblib
        for name in ["ridge_", "knn_", "adaboost_", "gbr_", "meta_"]:
            model = getattr(self, name)
            if model is not None:
                path = os.path.join(self.artifacts_dir, f"{name}.joblib")
                joblib.dump(model, path)
                files[name] = path

        # Also persist preprocessors explicitly (handy for debugging)
        if self.encoder_ is not None:
            path = os.path.join(self.artifacts_dir, "encoder.joblib")
            joblib.dump(self.encoder_, path)
            files["encoder"] = path
        if self.scaler_ is not None:
            path = os.path.join(self.artifacts_dir, "scaler.joblib")
            joblib.dump(self.scaler_, path)
            files["scaler"] = path

        self._artifact_files_ = files

    # Python pickling hooks to keep the main object lean and portable
    def __getstate__(self):
        state = self.__dict__.copy()

        # Drop heavy in-memory objects; keep only file refs
        # Boosters
        state["xgb_booster_"] = None
        state["lgbm_booster_"] = None

        # Wrappers (we reload from files)
        state["xgb_"] = None
        state["lgbm_"] = None
        state["catboost_"] = None

        # Sklearn models will be dropped; we reload their joblib files
        for name in ["ridge_", "knn_", "adaboost_", "gbr_", "meta_",
                     "encoder_", "scaler_"]:
            state[name] = None

        # Keep artifact file paths
        # (self._artifact_files_ already stored by _save_artifacts)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # On unpickle, reload from files
        files = getattr(self, "_artifact_files_", {})

        # Preprocessors
        enc_p = files.get("encoder")
        sc_p = files.get("scaler")
        if enc_p and os.path.exists(enc_p):
            self.encoder_ = joblib.load(enc_p)
        if sc_p and os.path.exists(sc_p):
            self.scaler_ = joblib.load(sc_p)

        # XGB booster
        xgb_p = files.get("xgb_booster")
        if xgb_p and os.path.exists(xgb_p):
            self.xgb_booster_ = xgb.Booster()
            self.xgb_booster_.load_model(xgb_p)
        else:
            self.xgb_booster_ = None

        # LGBM booster
        lgbm_p = files.get("lgbm_booster")
        if lgbm_p and os.path.exists(lgbm_p):
            import lightgbm as lgb
            self.lgbm_booster_ = lgb.Booster(model_file=lgbm_p)
        else:
            self.lgbm_booster_ = None

        # CatBoost
        cb_p = files.get("catboost")
        if cb_p and os.path.exists(cb_p):
            self.catboost_ = CatBoostRegressor()
            self.catboost_.load_model(cb_p, format="json")
        else:
            self.catboost_ = None

        # Sklearn models
        for name in ["ridge_", "knn_", "adaboost_", "gbr_", "meta_"]:
            p = files.get(name)
            if p and os.path.exists(p):
                setattr(self, name, joblib.load(p))
            else:
                setattr(self, name, None)