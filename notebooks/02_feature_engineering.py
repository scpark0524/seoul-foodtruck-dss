# %% [markdown]
# # STEP 2 — Feature Engineering  (★ 일자 DAILY 버전)
#
# 마스터 패널(행정동×일자) 위에 다음을 결합 → features_daily_{VERSION}.parquet
#
# | 군 | 내용 | 결합 키 | 가용 |
# |---|---|---|---|
# | A 캘린더 | year/month/dow/주말/계절 sin·cos | date 파생 | 전 기간 |
# | B 매출 lag | lag1/7/28/364, roll7/28, growth | dong | 전 기간 |
# | C 기상 | 기온/강수/습도/맑은날 | date | 2020~2024 |
# | D 공휴일 | 공휴일/연휴/샌드위치데이 | date | 2020~2024 |
# | E 대기질 | PM10/PM25/O3/NO2 | gu→dong, date | 2020~2024 |
# | F 생활인구 | 내국인/외국인/점심·저녁/20·30대 | dong, ymd | 2023~2025 |
# | G 소비·상권변화·점포 | 분기→broadcast | dong, yyqu | |
# | H 지하철·버스 | 월→broadcast | dong, yyyymm | |
# | I 시설/집객 | 점포종류/집객시설 | dong(정적) | |
#
# 결측은 NaN 유지 (XGBoost 자동 처리). float32 다운캐스팅으로 메모리 절약.

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

from src.config import (PROCESSED_DIR, RESULTS_DIR, SALES_DAILY_VERSION,
                        KEY_DONG, KEY_DATE, KEY_YMD, LABEL)
from src.data_loader import (load_sales_daily, load_living_pop_daily, load_one, standardize_sales)
from src.features import (aggregate_consumption, aggregate_trdar_change, aggregate_storecount,
                          normalize_facility, dong_to_gu_map, build_dong_name_to_cd)
from src.features_daily import (
    build_master_panel_daily, add_calendar_features, add_lag_features_daily,
    aggregate_weather_daily, aggregate_holidays_daily, aggregate_air_daily,
    aggregate_subway_monthly, aggregate_bus_monthly, merge_all_features_daily)

plt.rcParams["figure.dpi"] = 110
import platform, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
_FONT = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "DejaVu Sans")
plt.rcParams["font.family"] = _FONT
plt.rcParams["axes.unicode_minus"] = False

VERSION = SALES_DAILY_VERSION
print(f"[INFO] 일자 매출 버전 = {VERSION}")

# %% [markdown]
# ## 1) 패널 + 캘린더(A) + 매출 lag(B)

# %%
sd = load_sales_daily(VERSION)
panel = build_master_panel_daily(sd); del sd; gc.collect()
panel = add_calendar_features(panel)
panel = add_lag_features_daily(panel)
print(f"패널 + 캘린더 + lag: {panel.shape}")

# %% [markdown]
# ## 2) 일별 외부 — 기상(C)·공휴일(D)·생활인구(F)

# %%
wx  = aggregate_weather_daily(load_one("weather"));   print(f"  기상(일): {wx.shape}")
hol = aggregate_holidays_daily(load_one("holidays")); print(f"  공휴일(일): {hol.shape}")
lp  = load_living_pop_daily()

# %% [markdown]
# ## 3) 시설(I) + 대기질(E, gu→dong)

# %%
fac_raw = load_one("facility")
fac_static = normalize_facility(fac_raw)
dong2gu = dong_to_gu_map(fac_raw)
print(f"  시설 정적: {fac_static.shape}, dong→gu {len(dong2gu)}개")
air = aggregate_air_daily(load_one("air"), dong_to_gu=dong2gu)
print(f"  대기질(일): {air.shape}")
del fac_raw; gc.collect()

# %% [markdown]
# ## 4) 분기 broadcast — 소비/상권변화/점포(G)

# %%
cons  = aggregate_consumption(load_one("consumption"))
trdar = aggregate_trdar_change(load_one("trdar_change"))
store = aggregate_storecount(load_one("storecount"))
print(f"  소비 {cons.shape} | 상권변화 {trdar.shape} | 점포 {store.shape}")

# %% [markdown]
# ## 5) 월 broadcast — 지하철/버스(H)  (행정동명→코드 매핑 필요)

# %%
sales_std = standardize_sales(load_one("sales_dong"))
name2cd = build_dong_name_to_cd(sales_std); del sales_std; gc.collect()
sub = aggregate_subway_monthly(load_one("subway"), name2cd); print(f"  지하철(월): {sub.shape}")
bus = aggregate_bus_monthly(load_one("bus"), name2cd);       print(f"  버스(월): {bus.shape}")

# %% [markdown]
# ## 6) 전체 결합 + float32 + 저장

# %%
feat = merge_all_features_daily(
    panel, weather_d=wx, holiday_d=hol, air_d=air, livpop_d=lp,
    consumption_q=cons, trdar_q=trdar, storecount_q=store,
    subway_m=sub, bus_m=bus, facility_static=fac_static)
del panel, wx, hol, lp, air, cons, trdar, store, sub, bus, fac_static; gc.collect()

fcast = [c for c in feat.columns if feat[c].dtype == "float64"]
feat[fcast] = feat[fcast].astype("float32")
print(f"\n최종 피처 매트릭스: {feat.shape}  mem={feat.memory_usage(deep=True).sum()/1e6:.0f}MB")

# %% [markdown]
# ## 7) 결측 점검 + 타깃 상관 Top 20

# %%
miss = feat.isnull().mean().sort_values(ascending=False)
miss_nz = miss[miss > 0]
print(f"결측 있는 컬럼 ({len(miss_nz)}/{feat.shape[1]}):")
print(miss_nz.head(20).round(3))
if len(miss_nz) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(miss_nz)*0.22)))
    miss_nz.plot.barh(ax=ax, color="#e74c3c"); ax.invert_yaxis()
    ax.set_title(f"피처별 결측률 ({len(miss_nz)}개)"); ax.set_xlabel("결측 비율"); ax.grid(alpha=0.3)
    plt.tight_layout(); out = RESULTS_DIR / "02_missing_rate.png"
    plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

num = feat.select_dtypes(include=[np.number]).columns.tolist()
num = [c for c in num if c not in (KEY_YMD, "sales_ft", "sales_cnt", "yyqu", "yyyymm")]
corr_y = feat[num].corr()[LABEL].drop(LABEL).abs().sort_values(ascending=False)
print("\nTop 20 |corr(피처, y_log_sales)|:\n", corr_y.head(20).round(3))
fig, ax = plt.subplots(figsize=(8, 8))
corr_y.head(20).sort_values().plot.barh(ax=ax, color="#16a085")
ax.set_title("타깃 상관도 Top 20"); ax.set_xlabel("|correlation|"); ax.grid(alpha=0.3)
plt.tight_layout(); out = RESULTS_DIR / "02_top_correlation.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 8) 저장

# %%
out_features = PROCESSED_DIR / f"features_daily_{VERSION}.parquet"
feat.to_parquet(out_features, index=False)
print(f"[SAVED] {out_features}  shape={feat.shape}")
print("\n[OK] 다음: notebooks/03_xgboost_tuned.py")
