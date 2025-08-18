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
    return X


# ---------------------- Main Stacker --------------------------

class HypeStacker(BaseEstimator, RegressorMixin):
    """
    Full stacked regressor:
      - OneHotEncoder + StandardScaler
      - Base models: XGB, LGBM, CatBoost, Ridge, KNN, AdaBoost, GBR
      - Meta model: XGBRegressor
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

        # Base models
        self.xgb_: Optional[XGBRegressor] = None
        self.lgbm_: Optional[LGBMRegressor] = None
        self.catboost_: Optional[CatBoostRegressor] = None
        self.ridge_: Optional[Ridge] = None
        self.knn_: Optional[KNeighborsRegressor] = None
        self.adaboost_: Optional[AdaBoostRegressor] = None
        self.gbr_: Optional[GradientBoostingRegressor] = None

        # Boosters (portable prediction)
        self.xgb_booster_: Optional[xgb.Booster] = None
        self.lgbm_booster_: Optional[Any] = None

        # CatBoost indexing info
        self.cb_cols_: Optional[List[str]] = None
        self.cb_cat_idx_: Optional[List[int]] = None

        # Meta model
        self.meta_: Optional[XGBRegressor] = None

        # Artifact file map
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

    # ----------------- Fit / Predict API -----------------

    # (I won’t repeat the whole fit/predict you pasted — unchanged)

    # ----------------- Artifact save/load -----------------

    def _save_artifacts(self):
        files = {}

        if self.xgb_booster_ is not None:
            path = os.path.join(self.artifacts_dir, "xgb.json")
            self.xgb_booster_.save_model(path)
            files["xgb_booster"] = path

        if self.lgbm_booster_ is not None:
            path = os.path.join(self.artifacts_dir, "lgbm.txt")
            self.lgbm_booster_.save_model(path)
            files["lgbm_booster"] = path

        if self.catboost_ is not None:
            path = os.path.join(self.artifacts_dir, "catboost.json")
            self.catboost_.save_model(path, format="json")
            files["catboost"] = path

        for name in ["ridge_", "knn_", "adaboost_", "gbr_", "meta_"]:
            model = getattr(self, name)
            if model is not None:
                path = os.path.join(self.artifacts_dir, f"{name}.joblib")
                joblib.dump(model, path)
                files[name] = path

        if self.encoder_ is not None:
            path = os.path.join(self.artifacts_dir, "encoder.joblib")
            joblib.dump(self.encoder_, path)
            files["encoder"] = path
        if self.scaler_ is not None:
            path = os.path.join(self.artifacts_dir, "scaler.joblib")
            joblib.dump(self.scaler_, path)
            files["scaler"] = path

        self._artifact_files_ = files

    def load_artifacts(self, artifacts_dir: str = "artifacts"):
        """Load fitted encoder, scaler, and base/meta models from disk."""
        self.encoder_ = joblib.load(os.path.join(artifacts_dir, "encoder.joblib"))
        self.scaler_ = joblib.load(os.path.join(artifacts_dir, "scaler.joblib"))

        self.ridge_ = joblib.load(os.path.join(artifacts_dir, "ridge_.joblib"))
        self.knn_ = joblib.load(os.path.join(artifacts_dir, "knn_.joblib"))
        self.adaboost_ = joblib.load(os.path.join(artifacts_dir, "adaboost_.joblib"))
        self.gbr_ = joblib.load(os.path.join(artifacts_dir, "gbr_.joblib"))
        self.meta_ = joblib.load(os.path.join(artifacts_dir, "meta_.joblib"))

        self.xgb_booster_ = xgb.Booster()
        self.xgb_booster_.load_model(os.path.join(artifacts_dir, "xgb.json"))

        from lightgbm import Booster
        self.lgbm_booster_ = Booster(model_file=os.path.join(artifacts_dir, "lgbm.txt"))

        self.catboost_ = CatBoostRegressor()
        self.catboost_.load_model(os.path.join(artifacts_dir, "catboost.json"))