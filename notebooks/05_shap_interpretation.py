# %% [markdown]
# # STEP 5 — SHAP 해석 (XAI)  (★ 일자 DAILY 버전)
#
# XGBoost 내장 SHAP(`pred_contribs=True`) → 외부 shap 라이브러리 불필요.
# 일자 모델의 카테고리: 매출lag / 캘린더 / 생활인구 / 기상 / 공휴일 / 대기질 /
#                      소비 / 상권변화 / 점포 / 지하철 / 버스 / 시설.

# %%
from __future__ import annotations
import sys, pickle, gc
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from src.config import (PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, SALES_DAILY_VERSION,
                        KEY_DONG, KEY_DATE, KEY_YMD, LABEL)
from src.splits import date_holdout_split

plt.rcParams["figure.dpi"] = 110
import platform, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
_FONT = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "DejaVu Sans")
plt.rcParams["font.family"] = _FONT
plt.rcParams["axes.unicode_minus"] = False

VERSION = SALES_DAILY_VERSION
EXCLUDE = {KEY_DONG, KEY_DATE, KEY_YMD, LABEL, "sales_ft", "sales_cnt",
           "dong_nm", "yyqu", "yyyymm"}
SAMPLE = 25000   # SHAP 시각화용 표본 (val 에서)

# %% [markdown]
# ## 1) 모델 + Val 데이터 로드 (val 표본)

# %%
feat = pd.read_parquet(PROCESSED_DIR / f"features_daily_{VERSION}.parquet")
feat = feat.dropna(subset=[LABEL]).reset_index(drop=True)
feature_cols = [c for c in feat.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(feat[c])]
_, val, _ = date_holdout_split(feat); del feat; gc.collect()
if len(val) > SAMPLE:
    val = val.sample(SAMPLE, random_state=42)
val = val.reset_index(drop=True)
X_val, y_val = val[feature_cols], val[LABEL]

pkl = MODELS_DIR / "xgb_best.pkl"
if pkl.exists():
    with open(pkl, "rb") as f: model = pickle.load(f)
else:
    model = xgb.XGBRegressor(); model.load_model(MODELS_DIR / "xgb_best.json")
print(f"모델 로드. n_features={len(feature_cols)}, SHAP 표본={len(X_val)}")

# %% [markdown]
# ## 2) 내장 SHAP 계산

# %%
booster = model.get_booster()
shap_full = booster.predict(xgb.DMatrix(X_val), pred_contribs=True)
shap_values = shap_full[:, :-1]
expected_value = float(shap_full[0, -1])
mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
print(f"SHAP shape: {shap_values.shape}, expected={expected_value:.3f}")
print("\nTop 10 mean |SHAP|:\n", mean_abs.sort_values(ascending=False).head(10).round(4))

# %% [markdown]
# ## 3) Global Summary (Top 20)

# %%
top20 = mean_abs.sort_values(ascending=True).tail(20)
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(range(len(top20)), top20.values, color="#2c3e50")
ax.set_yticks(range(len(top20))); ax.set_yticklabels(top20.index)
ax.set_xlabel("mean |SHAP value|"); ax.set_title(f"Feature Importance — mean |SHAP| Top 20 ({VERSION})")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout(); out = RESULTS_DIR / "05_shap_summary.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 4) Beeswarm 유사 (Top 15)

# %%
top15 = mean_abs.sort_values(ascending=False).head(15).index.tolist()
fig, ax = plt.subplots(figsize=(10, 8))
for i, feat_n in enumerate(reversed(top15)):
    fi = feature_cols.index(feat_n); s = shap_values[:, fi]; fv = X_val[feat_n].values.astype(float)
    if np.nanstd(fv) > 0:
        fv_norm = (fv - np.nanmin(fv)) / (np.nanmax(fv) - np.nanmin(fv))
    else:
        fv_norm = np.zeros_like(fv)
    jit = np.random.uniform(-0.3, 0.3, len(s))
    sc = ax.scatter(s, [i]*len(s)+jit, c=fv_norm, cmap="coolwarm", s=8, alpha=0.4, edgecolors="none")
ax.set_yticks(range(len(top15))); ax.set_yticklabels(list(reversed(top15)))
ax.axvline(0, color="grey", lw=0.5); ax.set_xlabel("SHAP value (음수=매출↓, 양수=매출↑)")
ax.set_title(f"Top 15 피처 SHAP 분포 — 색=피처값 (red=high) ({VERSION})")
plt.colorbar(sc, ax=ax, shrink=0.6, label="feature value (norm)"); ax.grid(alpha=0.3, axis="x")
plt.tight_layout(); out = RESULTS_DIR / "05_shap_beeswarm.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 5) Dependence Plot (Top 6)

# %%
top6 = mean_abs.sort_values(ascending=False).head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, feat_n in zip(axes.flatten(), top6):
    fi = feature_cols.index(feat_n)
    ax.scatter(X_val[feat_n], shap_values[:, fi], alpha=0.3, s=6, color="#3498db")
    ax.axhline(0, color="grey", lw=0.5); ax.set_xlabel(feat_n); ax.set_ylabel(f"SHAP")
    ax.set_title(f"{feat_n} (|SHAP|={mean_abs[feat_n]:.3f})"); ax.grid(alpha=0.3)
plt.tight_layout(); out = RESULTS_DIR / "05_shap_dependence_top6.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 6) 카테고리별 기여도

# %%
def categorize(col: str) -> str:
    c = col.lower()
    if any(s in c for s in ["sales_lag", "sales_roll", "sales_growth"]):       return "A. 매출 lag/추세"
    if any(s in c for s in ["month", "dow", "doy", "woy", "_sin", "_cos", "year", "is_weekend", "day", "q"]):
                                                                                return "B. 캘린더"
    if "livpop" in c:                                                           return "C. 생활인구"
    if "wx_" in c:                                                              return "D. 기상"
    if "hol_" in c:                                                             return "E. 공휴일"
    if "air_" in c:                                                             return "F. 대기질"
    if "cons_" in c:                                                            return "G. 소비"
    if "trdar_" in c:                                                           return "H. 상권변화"
    if any(s in c for s in ["ft_store", "ft_franchise", "ft_open", "ft_close", "all_store"]):
                                                                                return "I. 점포"
    if "subway_" in c:                                                          return "J. 지하철"
    if "bus_" in c:                                                             return "K. 버스"
    if any(s in c for s in ["store_", "fac_"]):                                 return "L. 시설/집객"
    return "기타"

cat_df = pd.DataFrame({"feature": feature_cols, "mean_abs": mean_abs.values})
cat_df["category"] = cat_df["feature"].apply(categorize)
cat_score = cat_df.groupby("category")["mean_abs"].sum().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(range(len(cat_score)), cat_score.values, color="#16a085")
ax.set_yticks(range(len(cat_score))); ax.set_yticklabels(cat_score.index)
ax.set_title(f"데이터 카테고리별 일매출 예측 기여도 (Σ mean|SHAP|, {VERSION})")
ax.set_xlabel("Sum of mean |SHAP|"); ax.grid(alpha=0.3, axis="x")
plt.tight_layout(); out = RESULTS_DIR / "05_shap_by_category.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

print("\n=== 카테고리별 기여도 ===")
total = cat_score.sum()
for cat, sc in cat_score.sort_values(ascending=False).items():
    print(f"  {cat:18s} {sc:.4f} ({sc/total*100:.1f}%)")

# %% [markdown]
# ## 7) Local 해석 — 매출 상/중/하 3개

# %%
sidx = val[LABEL].sort_values().index.tolist(); n = len(sidx)
samples = {"high": sidx[int(n*0.95)], "median": sidx[int(n*0.50)], "low": sidx[int(n*0.05)]}
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (lvl, idx) in zip(axes, samples.items()):
    s = shap_values[val.index.get_loc(idx)]
    order = np.argsort(-np.abs(s))[:10]
    feats = [feature_cols[i] for i in order]; vals = s[order]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
    ax.barh(range(len(feats)), vals, color=colors)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats); ax.invert_yaxis()
    ax.axvline(0, color="grey", lw=0.5)
    info = val.loc[idx]
    ax.set_title(f"[{lvl}] dong={info[KEY_DONG]} {info[KEY_DATE].date()}\n실제={info[LABEL]:.2f}")
    ax.set_xlabel("SHAP (red=↑, blue=↓)"); ax.grid(alpha=0.3, axis="x")
plt.tight_layout(); out = RESULTS_DIR / "05_shap_local_3cases.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

print("\n[OK] STEP 5 완료. 산출물: results/05_shap_*.png")
