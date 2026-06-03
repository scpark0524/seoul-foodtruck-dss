# %% [markdown]
# # STEP 6 — 서비스 코어 모델: 입지 × 업종 × **일자** 매출 예측  (★ DAILY)
#
# **목표**: 실시간 웹 서비스 — 입지(+날짜) 입력 → 추천 메뉴 + **일 예상매출**.
# **타깃**: log1p(업종별 **일**매출).  분기 업종매출을 v2 일별 분배비율로 일자화.
# **설계**: self-lag 없음(신규 입지 가능) / 입지구조 + 캘린더 + 일별외부(기상·공휴일·생활인구·대기질).
# **검증**: 시간 holdout(2025) + leave-dong-out 공간 CV(안 본 동네 일반화).

# %%
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, xgboost as xgb
from src.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, KEY_DONG, KEY_TIME, RANDOM_STATE
from src.features_location import (build_location_daily_matrix, location_daily_feature_cols, DAILY_CAT_LABEL)
from src.splits import leave_dong_out_cv
from src.metrics import regression_metrics, topk_metrics
plt.rcParams["figure.dpi"]=110
import platform, logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"]={"Darwin":"AppleGothic","Windows":"Malgun Gothic"}.get(platform.system(),"DejaVu Sans")
plt.rcParams["axes.unicode_minus"]=False

# %% [markdown]
# ## 1) 일자 입지×업종 피처 빌드 (self-lag 없음)
# %%
feat = build_location_daily_matrix(n_dates_per_quarter=12, version="v2", save=True, verbose=True)
fc = location_daily_feature_cols(feat)
print(f"매트릭스: {feat.shape}  | 피처: {len(fc)} (self-lag 없음)")
PARAMS = dict(n_estimators=400, max_depth=8, learning_rate=0.07, subsample=0.8,
              colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
              random_state=RANDOM_STATE, objective="reg:squarederror", tree_method="hist")

# %% [markdown]
# ## 2) 검증 ① 시간 holdout (train ~2024 / test 2025)
# %%
tr = feat[feat[KEY_TIME] < 20251]; te = feat[feat[KEY_TIME] >= 20251]
m = xgb.XGBRegressor(**PARAMS); m.fit(tr[fc], tr[DAILY_CAT_LABEL], verbose=False)
pred = m.predict(te[fc])
tmetrics = regression_metrics(te[DAILY_CAT_LABEL], pred)
print("[시간 holdout 2025]", {k: round(v,4) for k,v in tmetrics.items()})
to = te[[KEY_DONG, "ymd", DAILY_CAT_LABEL]].copy(); to["yp"]=pred
print("[일자 입지추천 Top-K]", {k: round(v,3) for k,v in topk_metrics(to,"yp",DAILY_CAT_LABEL,"ymd",ks=(5,10)).items()})

# %% [markdown]
# ## 3) 검증 ② leave-dong-out 공간 CV (안 본 행정동)
# %%
sp=[]
for trf,tef,nm in leave_dong_out_cv(feat, n_splits=5):
    mm=xgb.XGBRegressor(**PARAMS); mm.fit(trf[fc],trf[DAILY_CAT_LABEL],verbose=False)
    r=regression_metrics(tef[DAILY_CAT_LABEL],mm.predict(tef[fc])); sp.append(r["r2"])
    print(f"  {nm}: R2={r['r2']:.4f}")
print(f"[공간 CV 평균] R2={np.mean(sp):.4f}")

# %% [markdown]
# ## 4) 최종 모델(전체) 저장
# %%
model = xgb.XGBRegressor(**PARAMS); model.fit(feat[fc], feat[DAILY_CAT_LABEL], verbose=False)
model.save_model(MODELS_DIR/"xgb_location_daily.json")
with open(MODELS_DIR/"xgb_location_daily.pkl","wb") as f: pickle.dump(model,f)
with open(MODELS_DIR/"location_daily_feature_cols.json","w",encoding="utf-8") as f: json.dump(fc,f,ensure_ascii=False)
json.dump({"time_holdout_2025":{k:round(v,4) for k,v in tmetrics.items()},
           "spatial_cv_mean_r2":round(float(np.mean(sp)),4),"n_features":len(fc),"n_rows":len(feat)},
          open(RESULTS_DIR/"06_location_daily_metrics.json","w"), ensure_ascii=False, indent=2)
print("[SAVED] models/xgb_location_daily.* + 06_location_daily_metrics.json")

# %% [markdown]
# ## 5) SHAP — 일매출을 좌우하는 변수
# %%
booster=model.get_booster()
samp=feat.sample(min(15000,len(feat)),random_state=42)
sv=booster.predict(xgb.DMatrix(samp[fc]),pred_contribs=True)[:,:-1]
ma=pd.Series(np.abs(sv).mean(0),index=fc).sort_values(ascending=False)
print("Top15:\n", ma.head(15).round(4).to_string())
t20=ma.sort_values().tail(20)
fig,ax=plt.subplots(figsize=(8,8)); ax.barh(range(len(t20)),t20.values,color="#2c3e50")
ax.set_yticks(range(len(t20))); ax.set_yticklabels(t20.index); ax.set_xlabel("mean |SHAP|")
ax.set_title("일 입지×업종 매출 — 핵심 변수 Top20"); ax.grid(alpha=.3,axis="x")
plt.tight_layout(); out=RESULTS_DIR/"06_location_daily_shap_top20.png"
plt.savefig(out,bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()
print("\n[OK] 다음: notebooks/07_service_demo.py")
