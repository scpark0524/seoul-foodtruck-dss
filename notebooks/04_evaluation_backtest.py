# %% [markdown]
# # STEP 4 — Walk-Forward 백테스팅 + v1/v2 비교  (★ 일자 DAILY 버전)
#
# 1. **Walk-forward** (expanding window, 90일 test × 4 fold) — 회귀 + Top-K 입지추천
# 2. **v1(균등 1/n) vs v2(요일 트렌드)** 분해 방식별 모델 성능 비교
#    → "분기매출을 어떻게 일자로 쪼개느냐"가 모델 학습에 주는 영향 정량화

# %%
from __future__ import annotations
import sys, gc, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from src.config import (PROCESSED_DIR, RESULTS_DIR, KEY_DONG, KEY_DATE, KEY_YMD,
                        LABEL, RANDOM_STATE)
from src.splits import walk_forward_daily
from src.metrics import regression_metrics, topk_metrics
from src.features_daily import build_feature_matrix_daily

plt.rcParams["figure.dpi"] = 110
import platform, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
_FONT = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "DejaVu Sans")
plt.rcParams["font.family"] = _FONT
plt.rcParams["axes.unicode_minus"] = False

EXCLUDE = {KEY_DONG, KEY_DATE, KEY_YMD, LABEL, "sales_ft", "sales_cnt",
           "dong_nm", "yyqu", "yyyymm"}
PARAMS = dict(n_estimators=400, max_depth=7, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
              random_state=RANDOM_STATE, objective="reg:squarederror",
              eval_metric="rmse", tree_method="hist")

# %% [markdown]
# ## 1) Walk-forward 함수 (한 버전에 대해)

# %%
def run_walk_forward(version: str):
    feat = build_feature_matrix_daily(version, save=True, verbose=True)
    feat = feat.dropna(subset=[LABEL]).reset_index(drop=True)
    fcols = [c for c in feat.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(feat[c])]
    rows, preds = [], []
    for tr, te, name in walk_forward_daily(feat, n_folds=4, test_days=90):
        m = xgb.XGBRegressor(**PARAMS)
        m.fit(tr[fcols], tr[LABEL], verbose=False)
        pred = m.predict(te[fcols])
        reg = regression_metrics(te[LABEL], pred)
        to = te[[KEY_DONG, KEY_YMD, LABEL]].copy(); to["y_pred"] = pred
        tk = topk_metrics(to, "y_pred", LABEL, KEY_YMD, ks=(5, 10))
        rows.append({"version": version, "fold": name, "n_train": len(tr),
                     "n_test": len(te), **reg, **tk})
        preds.append(to)
        print(f"  [{version}] {name}: RMSE={reg['rmse']:.4f} R2={reg['r2']:.4f} "
              f"Recall@5={tk['recall@5']:.3f}")
        del m; gc.collect()
    del feat; gc.collect()
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)

# %% [markdown]
# ## 2) v2 (기본) walk-forward

# %%
res_v2, preds_v2 = run_walk_forward("v2")
print("\n=== v2 fold 결과 ===")
print(res_v2.drop(columns=["version"]).round(4).to_string(index=False))

# %% [markdown]
# ## 3) v1 (균등 1/n) walk-forward

# %%
res_v1, _ = run_walk_forward("v1")
print("\n=== v1 fold 결과 ===")
print(res_v1.drop(columns=["version"]).round(4).to_string(index=False))

# %% [markdown]
# ## 4) v1 vs v2 비교 요약

# %%
def summary(df):
    n = df.select_dtypes(include=[np.number])
    return n.mean()[["rmse", "mae", "r2", "recall@5", "recall@10", "ndcg@10"]]

cmp = pd.DataFrame({"v1 (균등 1/n)": summary(res_v1), "v2 (요일 트렌드)": summary(res_v2)}).round(4)
print("\n=== v1 vs v2 평균 성능 ===")
print(cmp.to_string())

results_df = pd.concat([res_v1, res_v2], ignore_index=True)
results_df.to_csv(RESULTS_DIR / "04_backtest_metrics.csv", index=False)
print(f"\n[SAVED] results/04_backtest_metrics.csv")

# %% [markdown]
# ## 5) 시각화 — fold별 성능 + v1/v2 비교

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for df, lab, c in [(res_v1, "v1", "#9b59b6"), (res_v2, "v2", "#e74c3c")]:
    x = np.arange(len(df))
    axes[0].plot(x, df["r2"], "o-", color=c, label=lab)
    axes[1].plot(x, df["recall@5"], "o-", color=c, label=lab)
axes[0].set_title("Fold별 R²"); axes[0].set_xlabel("fold"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set_title("Fold별 Recall@5"); axes[1].set_xlabel("fold"); axes[1].set_ylim(0,1.05)
axes[1].legend(); axes[1].grid(alpha=0.3)
cmp.T[["rmse","mae"]].plot.bar(ax=axes[2], color=["#3498db","#9b59b6"])
axes[2].set_title("v1 vs v2 평균 오차"); axes[2].set_xticklabels(cmp.columns, rotation=0); axes[2].grid(alpha=0.3, axis="y")
plt.tight_layout(); out = RESULTS_DIR / "04_backtest_summary.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 6) Residual 분석 (v2)

# %%
preds_v2["residual"] = preds_v2[LABEL] - preds_v2["y_pred"]
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(preds_v2["residual"], bins=60, color="#7f8c8d")
axes[0].axvline(0, color="red", ls="--", alpha=0.7); axes[0].set_title("Residual 분포 (v2)"); axes[0].grid(alpha=0.3)
axes[1].scatter(preds_v2[LABEL], preds_v2["y_pred"], alpha=0.2, s=5, color="#2c3e50")
mn, mx = preds_v2[LABEL].min(), preds_v2[LABEL].max()
axes[1].plot([mn, mx], [mn, mx], "r--", lw=1)
axes[1].set_xlabel("실제 y_log_sales"); axes[1].set_ylabel("예측"); axes[1].set_title("실제 vs 예측 (v2)"); axes[1].grid(alpha=0.3)
plt.tight_layout(); out = RESULTS_DIR / "04_residual_analysis.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

preds_v2.to_parquet(PROCESSED_DIR / "backtest_predictions.parquet", index=False)
print("[SAVED] processed/backtest_predictions.parquet")
print("\n[OK] 다음: notebooks/05_shap_interpretation.py")
