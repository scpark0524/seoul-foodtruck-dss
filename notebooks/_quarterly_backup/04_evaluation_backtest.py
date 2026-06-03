# %% [markdown]
# # STEP 5 — Walk-Forward 백테스팅
#
# 12분기를 시간 순서대로 expanding window 로 나눠 4 fold 평가:
# ```
# Fold 1: train Q1~Q4   test Q5~Q6
# Fold 2: train Q1~Q6   test Q7~Q8
# Fold 3: train Q1~Q8   test Q9~Q10
# Fold 4: train Q1~Q10  test Q11~Q12
# ```
# 각 fold 마다 학습 → 회귀 메트릭(RMSE/MAE/R²) + Top-K 메트릭(Recall@K, NDCG@K)

# %%
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from src.config import (
    PROCESSED_DIR, RESULTS_DIR,
    KEY_DONG, KEY_TIME, LABEL, RANDOM_STATE,
)
from src.splits import walk_forward_folds
from src.metrics import regression_metrics, topk_metrics

plt.rcParams["figure.dpi"] = 110
import platform, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
_FONT = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "DejaVu Sans")
plt.rcParams["font.family"] = _FONT
plt.rcParams["axes.unicode_minus"] = False

# %% [markdown]
# ## 1) 피처 로드

# %%
features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
features = features.dropna(subset=[LABEL]).reset_index(drop=True)

EXCLUDE = {KEY_DONG, KEY_TIME, LABEL, "sales_ft"}
feature_cols = [c for c in features.columns
                if c not in EXCLUDE and pd.api.types.is_numeric_dtype(features[c])]
print(f"피처: {features.shape}, n_features={len(feature_cols)}")

# %% [markdown]
# ## 2) Walk-Forward 루프

# %%
params = dict(
    n_estimators=800, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=RANDOM_STATE, objective="reg:squarederror",
    eval_metric="rmse", tree_method="hist",
)

fold_results, fold_predictions = [], []

for tr, te, name in walk_forward_folds(features, min_train_qu=4, test_window=2):
    if len(tr) == 0 or len(te) == 0:
        continue
    Xtr, ytr = tr[feature_cols], tr[LABEL]
    Xte, yte = te[feature_cols], te[LABEL]

    m = xgb.XGBRegressor(**params)
    m.fit(Xtr, ytr, verbose=False)
    pred = m.predict(Xte)

    reg = regression_metrics(yte, pred)
    te_out = te[[KEY_DONG, KEY_TIME, LABEL]].copy(); te_out["y_pred"] = pred
    topk = topk_metrics(te_out, "y_pred", LABEL, KEY_TIME, ks=(3, 5, 10))

    row = {"fold": name, "n_train": len(tr), "n_test": len(te), **reg, **topk}
    fold_results.append(row); fold_predictions.append(te_out)
    print(f"{name}\n  RMSE={reg['rmse']:.4f} R2={reg['r2']:.4f}  Recall@5={topk['recall@5']:.3f} NDCG@10={topk['ndcg@10']:.3f}")

results_df = pd.DataFrame(fold_results)
all_preds  = pd.concat(fold_predictions, ignore_index=True)

# %% [markdown]
# ## 3) Fold 별 성능

# %%
print("\n=== Walk-Forward 결과 ===")
print(results_df.round(4))
print("\n=== 평균 ===")
print(results_df.select_dtypes(include=[np.number]).mean().round(4))

# %% [markdown]
# ## 4) 시각화

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
x = np.arange(len(results_df))
labels = [r.split("test")[-1].lstrip("_") for r in results_df["fold"]]

ax = axes[0]; ax2 = ax.twinx()
ax.bar(x - 0.2, results_df["rmse"], width=0.4, label="RMSE", color="#3498db")
ax.bar(x + 0.2, results_df["mae"],  width=0.4, label="MAE",  color="#9b59b6")
ax2.plot(x, results_df["r2"], "o-", color="#e74c3c", label="R²")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30)
ax.set_ylabel("RMSE / MAE"); ax2.set_ylabel("R²")
ax.set_title("Fold별 회귀 메트릭")
ax.legend(loc="upper left"); ax2.legend(loc="upper right"); ax.grid(alpha=0.3)

ax = axes[1]
for k in (3, 5, 10):
    ax.plot(x, results_df[f"recall@{k}"], "o-", label=f"Recall@{k}")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30)
ax.set_ylabel("Recall@K"); ax.set_ylim(0, 1.05)
ax.set_title("Fold별 Top-K 입지 추천 성능")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
out = RESULTS_DIR / "04_backtest_summary.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 5) Residual 분석

# %%
all_preds["residual"] = all_preds[LABEL] - all_preds["y_pred"]
all_preds["abs_resid"] = all_preds["residual"].abs()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(all_preds["residual"], bins=60, color="#7f8c8d")
axes[0].axvline(0, color="red", ls="--", alpha=0.7)
axes[0].set_title("Residual (y - y_pred) 분포"); axes[0].grid(alpha=0.3)

axes[1].scatter(all_preds[LABEL], all_preds["y_pred"], alpha=0.3, s=8, color="#2c3e50")
mn, mx = all_preds[LABEL].min(), all_preds[LABEL].max()
axes[1].plot([mn, mx], [mn, mx], "r--", linewidth=1)
axes[1].set_xlabel("실제 y_log_sales"); axes[1].set_ylabel("예측 y_log_sales")
axes[1].set_title("실제 vs 예측"); axes[1].grid(alpha=0.3)
plt.tight_layout()
out = RESULTS_DIR / "04_residual_analysis.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

worst = all_preds.nlargest(10, "abs_resid")
print("\n가장 못 맞춘 Top 10:")
print(worst[[KEY_DONG, KEY_TIME, LABEL, "y_pred", "residual"]].round(3))

# %% [markdown]
# ## 6) 저장

# %%
results_df.to_csv(RESULTS_DIR / "04_backtest_metrics.csv", index=False)
print(f"[SAVED] results/04_backtest_metrics.csv")
all_preds.to_parquet(PROCESSED_DIR / "backtest_predictions.parquet", index=False)
print(f"[SAVED] processed/backtest_predictions.parquet")
print("\n[OK] 다음: notebooks/05_shap_interpretation.py")
