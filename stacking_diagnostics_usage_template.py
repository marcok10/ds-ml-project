
# --- Stacking diagnostics & meta-model training ---
from stacking_diagnostics import get_oof_predictions, correlation_matrix, fit_meta_xgb
import pandas as pd

# 1) Define your base and tuned model dicts using the names from your notebook
# Example (EDIT to match your variable names):
# base_models = {
#     "XGB": xgb_base,
#     "Ridge": ridge_base,
#     "KNN": knn_base,
#     "LGBM": lgbm_base,
#     "CatBoost": cat_base,
#     "AdaBoost": ada_base,
#     "GBR": gbr_base,
# }
# tuned_models = {
#     "XGB": xgb_tuned,
#     "Ridge": ridge_tuned,
#     "KNN": knn_tuned,
#     "LGBM": lgbm_tuned,
#     "CatBoost": cat_tuned,
#     "AdaBoost": ada_tuned,
#     "GBR": gbr_tuned,
# }

# 2) Choose your feature matrix and target vector (EDIT to match your variables)
# X_final = <your final preprocessed training matrix/DataFrame>
# y_final = <your target vector as a NumPy array>

# 3) OOF predictions for base models
oof_base, fold_preds_base, rmses_base, cv_idx = get_oof_predictions(base_models, X_final, y_final, n_splits=5, random_state=42)
# 4) OOF predictions for tuned models
oof_tuned, fold_preds_tuned, rmses_tuned, _ = get_oof_predictions(tuned_models, X_final, y_final, n_splits=5, random_state=42)

# 5) Correlation matrices
corr_base = correlation_matrix(oof_base)
corr_tuned = correlation_matrix(oof_tuned)

# 6) RMSE tables
rmse_base_df = pd.DataFrame.from_dict(rmses_base, orient='index', columns=['OOF_RMSE']).sort_values('OOF_RMSE')
rmse_tuned_df = pd.DataFrame.from_dict(rmses_tuned, orient='index', columns=['OOF_RMSE']).sort_values('OOF_RMSE')

display(corr_base)
display(corr_tuned)
display(rmse_base_df)
display(rmse_tuned_df)

# 7) Fit regularized meta-XGBoost on the OOF predictions
meta_base = fit_meta_xgb(oof_base, y_final)     # meta-model over base learners
meta_tuned = fit_meta_xgb(oof_tuned, y_final)   # meta-model over tuned learners

# Optional: evaluate meta-models with cross-validated meta-OOM predictions or a holdout set if you have one.
