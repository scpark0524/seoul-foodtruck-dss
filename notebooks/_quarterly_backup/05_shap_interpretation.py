# %% [markdown]
# # STEP 6 — SHAP 해석 (XAI)
#
# **방법**: XGBoost 내장 SHAP (`pred_contribs=True`).
# shap 라이브러리는 XGBoost 3.2+ 와 호환성 이슈가 있어, XGBoost 가 직접 계산하는
# SHAP 값을 사용하고 matplotlib 으로 시각화합니다.
#
# **출력**
# 1. Global Summary — Top 20 피처별 평균 |SHAP|
# 2. SHAP Beeswarm 유사 — Top 15 피처의 영향 분포
# 3. Dependence Plot — Top 6 피처 단변량 영향
# 4. 카테고리별 평균 |SHAP| — 어떤 데이터 그룹이 가장 기여?
# 5. Local — 매출 상/중/하 행정동 3개의 피처별 기여

# %%
from __future__ import annotations
import sys, pickle
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from src.config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, KEY_DONG, KEY_TIME, LABEL
from src.splits import time_holdout_split

plt.rcParams["figure.dpi"] = 110
import platform, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
_FONT = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "DejaVu Sans")
plt.rcParams["font.family"] = _FONT
plt.rcParams["axes.unicode_minus"] = False

# %% [markdown]
# ## 1) 모델 + 데이터 로드

# %%
features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
features = features.dropna(subset=[LABEL]).reset_index(drop=True)

EXCLUDE = {KEY_DONG, KEY_TIME, LABEL, "sales_ft"}
feature_cols = [c for c in features.columns
                if c not in EXCLUDE and pd.api.types.is_numeric_dtype(features[c])]

_, val, _ = time_holdout_split(features)
val = val.reset_index(drop=True)
X_val, y_val = val[feature_cols], val[LABEL]

pkl_path = MODELS_DIR / "xgb_best.pkl"
if pkl_path.exists():
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
else:
    model = xgb.XGBRegressor()
    model.load_model(MODELS_DIR / "xgb_best.json")
print(f"모델 로드 완료. n_features={len(feature_cols)}, val_size={len(X_val)}")

# %% [markdown]
# ## 2) XGBoost 내장 SHAP 계산
# `pred_contribs=True` 가 (n_samples, n_features+1) shape 반환.
# 마지막 컬럼이 expected_value (bias), 나머지가 피처별 기여.

# %%
booster = model.get_booster()
dval = xgb.DMatrix(X_val, label=y_val)
shap_full = booster.predict(dval, pred_contribs=True)  # (n, n_feat+1)
shap_values = shap_full[:, :-1]   # 피처별 SHAP
expected_value = float(shap_full[0, -1])
print(f"SHAP shape: {shap_values.shape}, expected_value={expected_value:.3f}")

mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
print(f"\nTop 10 mean |SHAP|:")
print(mean_abs.sort_values(ascending=False).head(10).round(4))

# %% [markdown]
# ## 3) Global Summary (Top 20 평균 절대 SHAP)

# %%
top20 = mean_abs.sort_values(ascending=True).tail(20)
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(range(len(top20)), top20.values, color="#2c3e50")
ax.set_yticks(range(len(top20))); ax.set_yticklabels(top20.index)
ax.set_xlabel("mean |SHAP value|")
ax.set_title("Feature Importance — mean |SHAP| (Top 20)")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
out = RESULTS_DIR / "05_shap_summary.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 4) Beeswarm 유사 — Top 15 피처 SHAP 분포 (피처값으로 색칠)

# %%
top15 = mean_abs.sort_values(ascending=False).head(15).index.tolist()
fig, ax = plt.subplots(figsize=(10, 8))
for i, feat in enumerate(reversed(top15)):
    fi = feature_cols.index(feat)
    s = shap_values[:, fi]
    fv = X_val[feat].values
    # 피처값을 0~1로 정규화 → 색
    if np.nanstd(fv) > 0:
        fv_norm = (fv - np.nanmin(fv)) / (np.nanmax(fv) - np.nanmin(fv))
    else:
        fv_norm = np.zeros_like(fv)
    jitter = np.random.uniform(-0.3, 0.3, len(s))
    sc = ax.scatter(s, [i] * len(s) + jitter, c=fv_norm, cmap="coolwarm",
                    s=10, alpha=0.5, edgecolors="none")
ax.set_yticks(range(len(top15))); ax.set_yticklabels(list(reversed(top15)))
ax.axvline(0, color="grey", linewidth=0.5)
ax.set_xlabel("SHAP value (음수=매출 낮춤, 양수=매출 높임)")
ax.set_title("Top 15 피처 SHAP 분포 — 색=피처값 (red=high, blue=low)")
cbar = plt.colorbar(sc, ax=ax, shrink=0.6, label="feature value (normalized)")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
out = RESULTS_DIR / "05_shap_beeswarm.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 5) Dependence Plot — Top 6

# %%
top6 = mean_abs.sort_values(ascending=False).head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, feat in zip(axes.flatten(), top6):
    fi = feature_cols.index(feat)
    ax.scatter(X_val[feat], shap_values[:, fi], alpha=0.4, s=8, color="#3498db")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel(feat); ax.set_ylabel(f"SHAP({feat})")
    ax.set_title(f"{feat}  (mean |SHAP|={mean_abs[feat]:.3f})")
    ax.grid(alpha=0.3)
plt.tight_layout()
out = RESULTS_DIR / "05_shap_dependence_top6.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 6) 카테고리별 평균 |SHAP|

# %%
def categorize(col: str) -> str:
    c = col.lower()
    if any(s in c for s in ["sales_lag", "sales_roll", "sales_growth"]):       return "A. 매출 lag/추세"
    if any(s in c for s in ["q_sin", "q_cos", "year_norm", "year", "q"]):      return "B. 시간"
    if "cons_" in c:                                                            return "C. 소비"
    if "trdar_" in c:                                                           return "D. 상권변화"
    if any(s in c for s in ["ft_store", "ft_franchise", "ft_open", "ft_close", "all_store"]):
                                                                                return "E. 점포"
    if "subway_" in c:                                                          return "F. 지하철"
    if "bus_" in c:                                                             return "G. 버스"
    if "weather_" in c:                                                         return "H. 기상"
    if any(s in c for s in ["holiday_", "weekend"]):                            return "I. 공휴일"
    if "air_" in c:                                                             return "J. 대기질"
    if any(s in c for s in ["store_", "fac_"]):                                 return "K. 시설/집객"
    return "기타"

cat_df = pd.DataFrame({"feature": feature_cols, "mean_abs": mean_abs.values})
cat_df["category"] = cat_df["feature"].apply(categorize)
cat_score = cat_df.groupby("category")["mean_abs"].sum().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(range(len(cat_score)), cat_score.values, color="#16a085")
ax.set_yticks(range(len(cat_score))); ax.set_yticklabels(cat_score.index)
ax.set_title("데이터 카테고리별 매출 예측 기여도 (Σ mean |SHAP|)")
ax.set_xlabel("Sum of mean |SHAP|"); ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
out = RESULTS_DIR / "05_shap_by_category.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

print("\n=== 카테고리별 기여도 ===")
total = cat_score.sum()
for cat, sc in cat_score.sort_values(ascending=False).items():
    print(f"  {cat:18s} {sc:.4f} ({sc/total*100:.1f}%)")

# %% [markdown]
# ## 7) Local 해석 — 매출 상/중/하 행정동 3개

# %%
sorted_idx = val[LABEL].sort_values().index
n = len(sorted_idx)
samples = {
    "high":   int(sorted_idx[int(n * 0.95)]),
    "median": int(sorted_idx[int(n * 0.50)]),
    "low":    int(sorted_idx[int(n * 0.05)]),
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (level, idx) in zip(axes, samples.items()):
    s = shap_values[idx]
    pred = float(booster.predict(xgb.DMatrix(X_val.iloc[[idx]]))[0])
    actual = float(y_val.iloc[idx])
    # Top 10 영향 피처 (절댓값 기준)
    order = np.argsort(-np.abs(s))[:10]
    feats = [feature_cols[i] for i in order]
    vals = s[order]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
    y_pos = range(len(feats))
    ax.barh(y_pos, vals, color=colors)
    ax.set_yticks(y_pos); ax.set_yticklabels(feats); ax.invert_yaxis()
    ax.axvline(0, color="grey", lw=0.5)
    info = val.iloc[idx]
    ax.set_title(f"[{level}] dong={info[KEY_DONG]} yyqu={info[KEY_TIME]}\n"
                 f"실제={actual:.2f}  예측={pred:.2f}")
    ax.set_xlabel("SHAP value (red=↑매출, blue=↓매출)")
    ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
out = RESULTS_DIR / "05_shap_local_3cases.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 8) 보고서 요약

# %%
top10 = mean_abs.sort_values(ascending=False).head(10)
print("\n=== Top 10 글로벌 피처 ===")
for i, (feat, v) in enumerate(top10.items(), 1):
    print(f"  {i:2d}. {feat:30s} mean|SHAP|={v:.4f}")
print("\n[OK] STEP 6 완료. 산출물: results/05_shap_*.png")
