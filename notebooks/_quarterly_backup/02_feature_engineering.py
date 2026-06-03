# %% [markdown]
# # STEP 3 — Feature Engineering (14종 결합)
#
# 마스터 패널(424×12) 위에 다음 피처들을 결합:
# - **A.** 매출 lag/rolling/시간 (필수, 모든 분기 가능)
# - **B.** 소비 (분기, 행정동) — 12/12 분기
# - **C.** 상권변화 (분기, 행정동) — 12/12 분기
# - **D.** 점포 (분기, 행정동) — 2020-2024만 (2025 결측 → NaN)
# - **E.** 지하철·버스 (월별, 행정동) — 12/12 분기
# - **F.** 기상·공휴일 (일별, 서울) — 2020-2024만 (2025 결측)
# - **G.** 대기질 (일별, 구별) — 2020-2024만
# - **H.** 시설/집객 (정적, 행정동)
#
# 결측은 NaN 그대로 둠 — XGBoost 가 자동 처리.

# %%
from __future__ import annotations
import sys, warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)

from src.config import PROCESSED_DIR, RESULTS_DIR, ALL_QU, KEY_DONG, KEY_TIME, LABEL
from src.data_loader import load_one
from src.features import (
    add_lag_features, add_time_features,
    aggregate_consumption, aggregate_trdar_change, aggregate_storecount,
    aggregate_subway, aggregate_bus,
    aggregate_weather, aggregate_holidays, aggregate_air,
    normalize_facility, dong_to_gu_map, build_dong_name_to_cd,
    merge_all_features,
)
from src.data_loader import load_one, standardize_sales

plt.rcParams["figure.dpi"] = 110
import platform, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
_FONT = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "DejaVu Sans")
plt.rcParams["font.family"] = _FONT
plt.rcParams["axes.unicode_minus"] = False

# %% [markdown]
# ## 1) 마스터 패널 로드

# %%
panel = pd.read_parquet(PROCESSED_DIR / "panel_master.parquet")
print(f"마스터 패널: {panel.shape}")

# 지하철·버스 매핑용: 매출에서 (행정동명 → 8자리 adstr_cd) 사전 만들기
# (지하철·버스는 7자리 다른 코드 체계라 행정동명으로 join 해야 함)
sales_raw = load_one("sales_dong")
sales_std = standardize_sales(sales_raw)
DONG_NAME_TO_CD = build_dong_name_to_cd(sales_std)
print(f"행정동명→코드 매핑: {len(DONG_NAME_TO_CD)}개")
del sales_raw, sales_std

# %% [markdown]
# ## 2) A. 매출 lag + 시간 피처 (필수, 가장 강력한 단일 피처군)

# %%
features = add_lag_features(panel)
features = add_time_features(features)
print(f"lag + 시간 추가 후: {features.shape}")
print(f"추가된 컬럼: {[c for c in features.columns if c not in panel.columns]}")

# %% [markdown]
# ## 3) B/C/D. 분기-행정동 단위 raw 데이터 집계

# %%
# 소비
try:
    cons_raw = load_one("consumption")
    cons_q = aggregate_consumption(cons_raw)
    cons_q = cons_q[cons_q[KEY_TIME].isin(ALL_QU)]
    print(f"  소비 집계: {cons_q.shape}")
except Exception as e:
    cons_q = None; print(f"  [SKIP] 소비: {e}")

# 상권변화
try:
    trdar_raw = load_one("trdar_change")
    trdar_q = aggregate_trdar_change(trdar_raw)
    trdar_q = trdar_q[trdar_q[KEY_TIME].isin(ALL_QU)]
    print(f"  상권변화 집계: {trdar_q.shape}")
except Exception as e:
    trdar_q = None; print(f"  [SKIP] 상권변화: {e}")

# 점포 정보 (푸드트럭 적합 업종만)
try:
    store_raw = load_one("storecount")
    store_q = aggregate_storecount(store_raw)
    store_q = store_q[store_q[KEY_TIME].isin(ALL_QU)]
    print(f"  점포 집계: {store_q.shape}  (2025 결측 정상)")
except Exception as e:
    store_q = None; print(f"  [SKIP] 점포: {e}")

# %% [markdown]
# ## 4) E. 지하철·버스 (월별 → 분기 평균)

# %%
try:
    subway_raw = load_one("subway")
    subway_q = aggregate_subway(subway_raw, dong_name_to_cd=DONG_NAME_TO_CD)
    subway_q = subway_q[subway_q[KEY_TIME].isin(ALL_QU)]
    print(f"  지하철 집계: {subway_q.shape}")
except Exception as e:
    subway_q = None; print(f"  [SKIP] 지하철: {e}")

try:
    bus_raw = load_one("bus")
    bus_q = aggregate_bus(bus_raw, dong_name_to_cd=DONG_NAME_TO_CD)
    bus_q = bus_q[bus_q[KEY_TIME].isin(ALL_QU)]
    print(f"  버스 집계: {bus_q.shape}")
except Exception as e:
    bus_q = None; print(f"  [SKIP] 버스: {e}")

# %% [markdown]
# ## 5) F/G. 기상·공휴일·대기질 (일별 → 분기 통계)

# %%
try:
    weather_raw = load_one("weather")
    weather_q = aggregate_weather(weather_raw)
    print(f"  기상 집계: {weather_q.shape}  (2025 결측 정상)")
except Exception as e:
    weather_q = None; print(f"  [SKIP] 기상: {e}")

try:
    hol_raw = load_one("holidays")
    hol_q = aggregate_holidays(hol_raw)
    print(f"  공휴일 집계: {hol_q.shape}  (2025 결측 정상)")
except Exception as e:
    hol_q = None; print(f"  [SKIP] 공휴일: {e}")

# %% [markdown]
# ## 6) H. 시설 (정적) + 대기질(구 단위)

# %%
fac_raw = fac_static = dong2gu = None
try:
    fac_raw = load_one("facility")
    fac_static = normalize_facility(fac_raw)
    dong2gu = dong_to_gu_map(fac_raw)
    print(f"  시설 정적: {fac_static.shape}, dong→gu 매핑 {len(dong2gu)}개")
except Exception as e:
    print(f"  [SKIP] 시설: {e}")

air_q = None
try:
    air_raw = load_one("air")
    air_q = aggregate_air(air_raw, dong_to_gu=dong2gu)
    air_q = air_q[air_q[KEY_TIME].isin(ALL_QU)] if KEY_TIME in air_q.columns else air_q
    print(f"  대기질 집계: {air_q.shape}  (2025 결측 정상)")
except Exception as e:
    print(f"  [SKIP] 대기질: {e}")

# %% [markdown]
# ## 7) 모든 피처 결합 → features.parquet

# %%
features_full = merge_all_features(
    features,
    consumption_q=cons_q, trdar_q=trdar_q, storecount_q=store_q,
    subway_q=subway_q, bus_q=bus_q,
    weather_q=weather_q, holidays_q=hol_q, air_q=air_q,
    facility_static=fac_static,
)
print(f"\n최종 피처 매트릭스 shape: {features_full.shape}")

# %% [markdown]
# ## 8) 결측치 점검 + 시각화

# %%
miss = features_full.isnull().mean().sort_values(ascending=False)
miss_nonzero = miss[miss > 0]
print(f"\n결측치 있는 컬럼 ({len(miss_nonzero)}/{len(features_full.columns)}):")
print(miss_nonzero.round(3))

# 결측 패턴 시각화
if len(miss_nonzero) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(miss_nonzero)*0.25)))
    miss_nonzero.plot.barh(ax=ax, color="#e74c3c")
    ax.set_title(f"피처별 결측률 ({len(miss_nonzero)}개)")
    ax.set_xlabel("결측 비율")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "02_missing_rate.png"
    plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 9) 타깃과 가장 상관 큰 Top 20 피처

# %%
num_cols = features_full.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in (KEY_TIME, "sales_ft")]
corr_y = features_full[num_cols].corr()[LABEL].drop(LABEL).abs().sort_values(ascending=False)
print("\nTop 20 |corr(피처, y_log_sales)|:")
print(corr_y.head(20).round(3))

fig, ax = plt.subplots(figsize=(8, 8))
corr_y.head(20).sort_values().plot.barh(ax=ax, color="#16a085")
ax.set_title("타깃 상관도 Top 20")
ax.set_xlabel("|correlation|"); ax.grid(alpha=0.3)
plt.tight_layout()
out = RESULTS_DIR / "02_top_correlation.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 10) 저장

# %%
out_features = PROCESSED_DIR / "features.parquet"
features_full.to_parquet(out_features, index=False)
print(f"[SAVED] {out_features}  shape={features_full.shape}")
print("\n[OK] 다음: notebooks/03_xgboost_tuned.py")
