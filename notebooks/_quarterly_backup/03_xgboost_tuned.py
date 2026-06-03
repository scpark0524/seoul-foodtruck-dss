# %% [markdown]
# # STEP 4 — XGBoost 학습
#
# 1. Time-based split: Train (Q1~Q8) / Val (Q9~Q10) / Test (Q11~Q12)
# 2. XGBoost 학습 (early stopping)
# 3. 학습 곡선 + Feature Importance
# 4. Val/Test 성능

# %%
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from src.config import (
    PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    KEY_DONG, KEY_TIME, LABEL, RANDOM_STATE,
)
from src.splits import time_holdout_split
from src.metrics import regression_metrics

plt.rcParams["figure.dpi"] = 110
import platform, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
_FONT = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "DejaVu Sans")
plt.rcParams["font.family"] = _FONT
plt.rcParams["axes.unicode_minus"] = False

# %% [markdown]
# ## 1) 피처 로드 + 타깃 결측 제거

# %%
features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
print(f"피처 매트릭스: {features.shape}")

features = features.dropna(subset=[LABEL]).reset_index(drop=True)
print(f"타깃 결측 제거 후: {features.shape}")

train, val, test = time_holdout_split(features)

# %% [markdown]
# ## 2) 피처 / 타깃 분리

# %%
EXCLUDE = {KEY_DONG, KEY_TIME, LABEL, "sales_ft"}
feature_cols = [c for c in features.columns
                if c not in EXCLUDE and pd.api.types.is_numeric_dtype(features[c])]
print(f"학습 피처 수: {len(feature_cols)}")

X_train, y_train = train[feature_cols], train[LABEL]
X_val,   y_val   = val[feature_cols],   val[LABEL]
X_test,  y_test  = test[feature_cols],  test[LABEL]
print(f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

# %% [markdown]
# ## 3) XGBoost 학습 (early stopping)

# %%
model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    objective="reg:squarederror",
    eval_metric=["rmse", "mae"],
    early_stopping_rounds=50,
    tree_method="hist",
)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100,
)
print(f"\nBest iteration: {model.best_iteration}")

# %% [markdown]
# ## 4) 학습 곡선

# %%
res = model.evals_result()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(res["validation_0"]["rmse"], label="train RMSE", color="#3498db")
ax.plot(res["validation_1"]["rmse"], label="val RMSE",   color="#e74c3c")
ax.axvline(model.best_iteration, linestyle="--", color="#888", alpha=0.6,
           label=f"best iter={model.best_iteration}")
ax.set_xlabel("boosting round"); ax.set_ylabel("RMSE")
ax.set_title("XGBoost 학습 곡선")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
out = RESULTS_DIR / "03_xgb_train_curve.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 5) Val/Test 성능

# %%
pred_val  = model.predict(X_val)
pred_test = model.predict(X_test)
val_metrics  = regression_metrics(y_val,  pred_val)
test_metrics = regression_metrics(y_test, pred_test)
print(f"\nVal  : {val_metrics}")
print(f"Test : {test_metrics}")

# %% [markdown]
# ## 6) Feature Importance Top 20

# %%
fi = pd.Series(model.feature_importances_, index=feature_cols)
fi_top = fi.sort_values(ascending=True).tail(20)
fig, ax = plt.subplots(figsize=(8, 8))
fi_top.plot.barh(ax=ax, color="#2c3e50")
ax.set_title("XGBoost Feature Importance (Top 20)")
ax.set_xlabel("Gain"); ax.grid(alpha=0.3)
plt.tight_layout()
out = RESULTS_DIR / "03_xgb_feature_importance.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 7) 저장

# %%
out_model = MODELS_DIR / "xgb_best.json"
model.save_model(out_model); print(f"[SAVED] {out_model}")

# SHAP 호환을 위해 pickle 도 함께 저장 (XGBoost 3.2 + SHAP 0.49 호환 이슈 우회)
import pickle
with open(MODELS_DIR / "xgb_best.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"[SAVED] {MODELS_DIR / 'xgb_best.pkl'}")

pred_df = test[[KEY_DONG, KEY_TIME, LABEL]].copy()
pred_df["y_pred"] = pred_test
pred_df.to_parquet(PROCESSED_DIR / "predictions_test.parquet", index=False)
print(f"[SAVED] processed/predictions_test.parquet")

with (RESULTS_DIR / "03_xgb_metrics.json").open("w", encoding="utf-8") as f:
    json.dump({"val": val_metrics, "test": test_metrics,
               "best_iteration": int(model.best_iteration),
               "n_features": len(feature_cols)}, f, indent=2, ensure_ascii=False)
print(f"[SAVED] results/03_xgb_metrics.json")
print("\n[OK] 다음: notebooks/04_evaluation_backtest.py")
