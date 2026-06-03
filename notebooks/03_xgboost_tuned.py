# %% [markdown]
# # STEP 3 — XGBoost 학습  (★ 일자 DAILY 버전)
#
# 1. 일자 기반 split: Train(~2024-12-31) / Val(~2025-06-30) / Test(2025-07-01~)
# 2. XGBoost early stopping
# 3. 학습 곡선 + Feature Importance + Val/Test 성능

# %%
from __future__ import annotations
import sys, json, gc, pickle
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from src.config import (PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, SALES_DAILY_VERSION,
                        KEY_DONG, KEY_DATE, KEY_YMD, LABEL, RANDOM_STATE)
from src.splits import date_holdout_split
from src.metrics import regression_metrics

plt.rcParams["figure.dpi"] = 110
import platform, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
_FONT = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "DejaVu Sans")
plt.rcParams["font.family"] = _FONT
plt.rcParams["axes.unicode_minus"] = False

VERSION = SALES_DAILY_VERSION
# 모델에서 제외할 비(非)피처 컬럼 (키/타깃/원매출/broadcast 키)
EXCLUDE = {KEY_DONG, KEY_DATE, KEY_YMD, LABEL, "sales_ft", "sales_cnt",
           "dong_nm", "yyqu", "yyyymm"}

# %% [markdown]
# ## 1) 피처 로드 + 일자 split

# %%
feat = pd.read_parquet(PROCESSED_DIR / f"features_daily_{VERSION}.parquet")
feat = feat.dropna(subset=[LABEL]).reset_index(drop=True)
print(f"피처 매트릭스: {feat.shape}")

feature_cols = [c for c in feat.columns
                if c not in EXCLUDE and pd.api.types.is_numeric_dtype(feat[c])]
print(f"학습 피처 수: {len(feature_cols)}")

train, val, test = date_holdout_split(feat)
del feat; gc.collect()

X_train, y_train = train[feature_cols], train[LABEL]
X_val,   y_val   = val[feature_cols],   val[LABEL]
X_test,  y_test  = test[feature_cols],  test[LABEL]

# %% [markdown]
# ## 2) XGBoost 학습 (early stopping)

# %%
model = xgb.XGBRegressor(
    n_estimators=1000, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    random_state=RANDOM_STATE, objective="reg:squarederror",
    eval_metric="rmse", early_stopping_rounds=40, tree_method="hist")
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=100)
print(f"\nBest iteration: {model.best_iteration}")

# %% [markdown]
# ## 3) 학습 곡선

# %%
res = model.evals_result()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(res["validation_0"]["rmse"], label="train RMSE", color="#3498db")
ax.plot(res["validation_1"]["rmse"], label="val RMSE", color="#e74c3c")
ax.axvline(model.best_iteration, ls="--", color="#888", alpha=0.6, label=f"best={model.best_iteration}")
ax.set_xlabel("boosting round"); ax.set_ylabel("RMSE"); ax.set_title(f"XGBoost 학습 곡선 ({VERSION}, 일 단위)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); out = RESULTS_DIR / "03_xgb_train_curve.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 4) Val / Test 성능

# %%
pred_val, pred_test = model.predict(X_val), model.predict(X_test)
val_metrics  = regression_metrics(y_val, pred_val)
test_metrics = regression_metrics(y_test, pred_test)
print(f"\nVal  : {val_metrics}")
print(f"Test : {test_metrics}")

# %% [markdown]
# ## 5) Feature Importance Top 20

# %%
fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(20)
fig, ax = plt.subplots(figsize=(8, 8))
fi.plot.barh(ax=ax, color="#2c3e50")
ax.set_title(f"XGBoost Feature Importance Top 20 ({VERSION})"); ax.set_xlabel("Gain"); ax.grid(alpha=0.3)
plt.tight_layout(); out = RESULTS_DIR / "03_xgb_feature_importance.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 6) 저장

# %%
model.save_model(MODELS_DIR / "xgb_best.json")
with open(MODELS_DIR / "xgb_best.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"[SAVED] models/xgb_best.json + .pkl")

pred_df = test[[KEY_DONG, KEY_DATE, KEY_YMD, LABEL]].copy(); pred_df["y_pred"] = pred_test
pred_df.to_parquet(PROCESSED_DIR / "predictions_test.parquet", index=False)
with (RESULTS_DIR / "03_xgb_metrics.json").open("w", encoding="utf-8") as f:
    json.dump({"version": VERSION, "val": val_metrics, "test": test_metrics,
               "best_iteration": int(model.best_iteration),
               "n_features": len(feature_cols)}, f, indent=2, ensure_ascii=False)
print("[SAVED] processed/predictions_test.parquet + results/03_xgb_metrics.json")
print("\n[OK] 다음: notebooks/04_evaluation_backtest.py")
