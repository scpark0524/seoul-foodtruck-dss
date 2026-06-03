# %% [markdown]
# # STEP 6 — 서비스 코어 모델: 입지 × 업종 매출 예측  (★ 신규)
#
# **목적**: 실시간 웹 서비스 — 입지 입력 → **추천 메뉴 + 예상 매출**.
# **핵심 설계**:
# - 타깃: log1p(업종별 분기매출).  한 모델이 (입지, 업종) 조합의 매출을 예측.
# - **self-lag 없음** → 기록 없는 신규 후보 입지도 입지속성만으로 예측 → 최적입지 추천 가능.
# - 검증: **시간 holdout(2025)** + **leave-dong-out 공간 CV**("안 본 동네" 일반화).
# - SHAP: 어떤 입지 변수가 매출을 좌우하는가(=발표 핵심).

# %%
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, matplotlib.pyplot as plt, xgboost as xgb
from src.config import ALL_QU_AVAILABLE, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, KEY_DONG, KEY_TIME, RANDOM_STATE
from src.features_location import build_location_matrix, location_feature_cols, CAT_LABEL
from src.splits import leave_dong_out_cv, time_holdout_quarters
from src.metrics import regression_metrics, topk_metrics

plt.rcParams["figure.dpi"] = 110
import platform, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = {"Darwin":"AppleGothic","Windows":"Malgun Gothic"}.get(platform.system(),"DejaVu Sans")
plt.rcParams["axes.unicode_minus"] = False

# %% [markdown]
# ## 1) 피처 빌드 (입지×업종×분기, self-lag 없음)

# %%
feat = build_location_matrix(ALL_QU_AVAILABLE, save=True, verbose=True)
feat = feat.dropna(subset=[CAT_LABEL]).reset_index(drop=True)
fcols = location_feature_cols(feat)
print(f"피처 매트릭스: {feat.shape}  | 모델 입력 피처: {len(fcols)} (self-lag 없음)")

PARAMS = dict(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
              random_state=RANDOM_STATE, objective="reg:squarederror", tree_method="hist")

# %% [markdown]
# ## 2) 검증 ① 시간 holdout (train 2021~2024, test 2025)

# %%
train_qu = [q for q in ALL_QU_AVAILABLE if q < 20251]
test_qu  = [q for q in ALL_QU_AVAILABLE if q >= 20251]
tr, te = time_holdout_quarters(feat, train_qu, test_qu)
m_time = xgb.XGBRegressor(**PARAMS); m_time.fit(tr[fcols], tr[CAT_LABEL], verbose=False)
time_metrics = regression_metrics(te[CAT_LABEL], m_time.predict(te[fcols]))
print("[시간 holdout 2025]", {k: round(v,4) for k,v in time_metrics.items()})
# 입지추천 품질: 각 분기 업종별 매출 Top-K 동
to = te[[KEY_DONG, KEY_TIME, CAT_LABEL]].copy(); to["y_pred"] = m_time.predict(te[fcols])
print("Top-K 입지:", {k: round(v,3) for k,v in topk_metrics(to,"y_pred",CAT_LABEL,KEY_TIME,ks=(5,10)).items()})

# %% [markdown]
# ## 3) 검증 ② leave-dong-out 공간 CV (안 본 행정동 일반화)

# %%
sp = []
for trf, tef, name in leave_dong_out_cv(feat, n_splits=5):
    mm = xgb.XGBRegressor(**PARAMS); mm.fit(trf[fcols], trf[CAT_LABEL], verbose=False)
    r = regression_metrics(tef[CAT_LABEL], mm.predict(tef[fcols]))
    sp.append({"fold": name, **r}); print(f"  {name}: R2={r['r2']:.4f} RMSE={r['rmse']:.4f}")
sp_df = pd.DataFrame(sp)
print(f"\n[공간 CV 평균] R2(안 본 동네)={sp_df['r2'].mean():.4f}  RMSE={sp_df['rmse'].mean():.4f}")

# %% [markdown]
# ## 4) 최종 모델 학습 (전체 데이터) → 서빙용 저장

# %%
model = xgb.XGBRegressor(**PARAMS); model.fit(feat[fcols], feat[CAT_LABEL], verbose=False)
model.save_model(MODELS_DIR / "xgb_location.json")
with open(MODELS_DIR / "xgb_location.pkl", "wb") as f: pickle.dump(model, f)
with open(MODELS_DIR / "location_feature_cols.json", "w", encoding="utf-8") as f:
    json.dump(fcols, f, ensure_ascii=False, indent=2)
metrics_out = {"time_holdout_2025": time_metrics,
               "spatial_cv_mean_r2": float(sp_df["r2"].mean()),
               "spatial_cv_folds": sp, "n_features": len(fcols)}
with open(RESULTS_DIR / "06_location_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, ensure_ascii=False, indent=2)
print("[SAVED] models/xgb_location.* + location_feature_cols.json + results/06_location_metrics.json")

# %% [markdown]
# ## 5) SHAP — 어떤 입지 변수가 매출을 좌우하는가

# %%
booster = model.get_booster()
samp = feat.sample(min(20000, len(feat)), random_state=42).reset_index(drop=True)
shap_full = booster.predict(xgb.DMatrix(samp[fcols]), pred_contribs=True)
shap_values = shap_full[:, :-1]
mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=fcols).sort_values(ascending=False)
print("\nTop 15 변수(mean|SHAP|):\n", mean_abs.head(15).round(4).to_string())

top20 = mean_abs.sort_values(ascending=True).tail(20)
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(range(len(top20)), top20.values, color="#2c3e50")
ax.set_yticks(range(len(top20))); ax.set_yticklabels(top20.index)
ax.set_xlabel("mean |SHAP|"); ax.set_title("입지×업종 매출 — 핵심 변수 Top 20 (self-lag 없음)")
ax.grid(alpha=0.3, axis="x"); plt.tight_layout()
out = RESULTS_DIR / "06_location_shap_top20.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 6) 카테고리별 변수군 기여도

# %%
def cat_of(c):
    cl = c.lower()
    if c.startswith("cat_is_"):                                   return "업종 정체성"
    if c.startswith("cat_"):                                      return "업종 경쟁/점포"
    if "livpop" in cl:                                            return "생활인구"
    if cl.startswith(("cons_",)):                                 return "소비력"
    if "trdar" in cl:                                             return "상권변화"
    if "subway" in cl or "bus" in cl:                             return "대중교통 유동"
    if cl.startswith("fac_"):                                     return "집객시설"
    if cl.startswith("store_") or cl=="all_store_cnt":            return "상권 밀도"
    if cl in ("year","q","q_sin","q_cos","year_norm"):           return "시간"
    return "기타"
cdf = pd.DataFrame({"f": fcols, "v": mean_abs.reindex(fcols).values}); cdf["g"] = cdf["f"].map(cat_of)
cs = cdf.groupby("g")["v"].sum().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(range(len(cs)), cs.values, color="#16a085")
ax.set_yticks(range(len(cs))); ax.set_yticklabels(cs.index)
ax.set_title("변수군별 기여도 (Σ mean|SHAP|)"); ax.set_xlabel("기여도"); ax.grid(alpha=0.3, axis="x")
plt.tight_layout(); out = RESULTS_DIR / "06_location_shap_by_group.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()
print("\n변수군 기여:")
tot = cs.sum()
for g, v in cs.sort_values(ascending=False).items(): print(f"  {g:14s} {v/tot*100:5.1f}%")
print("\n[OK] 다음: notebooks/07_service_demo.py")
