# %% [markdown]
# # STEP 1 — EDA + Naive Baseline  (★ 일자 DAILY 버전)
#
# **데이터**: backtest_data/5. 추정매출/sales_v2_full_weekday_trend.csv
#            (분기 추정매출을 요일 트렌드로 일자 분해한 파일)
# **단위**: 행정동 × 일자 (2021-01-01 ~ 2025-12-31, 약 425동 × 1,826일)
# **타깃**: 그 날의 푸드트럭 적합 업종 일매출 → log1p
#
# > 분기 버전 노트북은 notebooks/_quarterly_backup/ 에 보관.

# %%
from __future__ import annotations
import sys, json, gc
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (DATA_DIR, PROCESSED_DIR, RESULTS_DIR,
                        SALES_DAILY_VERSION, KEY_DONG, KEY_DATE, KEY_YMD, LABEL)
from src.data_loader import load_sales_daily
from src.features_daily import build_master_panel_daily
from src.metrics import regression_metrics

plt.rcParams["figure.dpi"] = 110
import platform, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
_FONT = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "DejaVu Sans")
plt.rcParams["font.family"] = _FONT
plt.rcParams["axes.unicode_minus"] = False

VERSION = SALES_DAILY_VERSION   # 'v2' (요일 트렌드) 기본. v1 비교는 STEP 4.
print(f"[INFO] DATA_DIR exists = {DATA_DIR.exists()}  | 일자 매출 버전 = {VERSION}")

# %% [markdown]
# ## 1) 일자 매출 로드 + 마스터 패널 (타깃 = log1p 일매출)

# %%
sd = load_sales_daily(VERSION)
panel = build_master_panel_daily(sd)   # leakage(요일/시간대/성별/연령) 컬럼 제거 + 타깃 생성
del sd; gc.collect()
print(f"\n마스터 패널 shape: {panel.shape}")
print(f"기간: {panel[KEY_DATE].min().date()} ~ {panel[KEY_DATE].max().date()}")
print(f"행정동 수: {panel[KEY_DONG].nunique()}  | 일자 수: {panel[KEY_DATE].nunique()}")
print(f"타깃 결측: {panel[LABEL].isna().sum()}")
print(f"\ny_log_sales 분포:\n{panel[LABEL].describe().round(3)}")

# %% [markdown]
# ## 2) 타깃 분포 (raw vs log1p)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(panel["sales_ft"].dropna() / 1e6, bins=80, color="#444", alpha=0.85)
axes[0].set_title("raw 일매출 분포 (롱테일)"); axes[0].set_xlabel("일매출 (백만원)")
axes[1].hist(panel[LABEL].dropna(), bins=80, color="#c0392b", alpha=0.85)
axes[1].set_title("y_log_sales 분포 (log1p)"); axes[1].set_xlabel("log(1+일매출)")
for ax in axes: ax.grid(alpha=0.3)
plt.tight_layout()
out = RESULTS_DIR / "01_target_distribution.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 3) 일별 매출 추세 + 요일 패턴 (v2 의 핵심)

# %%
daily_tot = panel.groupby(KEY_DATE)["sales_ft"].sum() / 1e8   # 억원
fig, axes = plt.subplots(2, 1, figsize=(13, 7))
axes[0].plot(daily_tot.index, daily_tot.values, linewidth=0.6, color="#2c3e50")
axes[0].set_title("서울 푸드트럭 적합 업종 일별 매출 총합 (억원)")
axes[0].grid(alpha=0.3)
# 요일별 평균
panel["_dow"] = panel[KEY_DATE].dt.dayofweek
dow_mean = panel.groupby("_dow")["sales_ft"].mean() / 1e6
dow_lbl = ["월","화","수","목","금","토","일"]
axes[1].bar(dow_lbl, dow_mean.values, color="#16a085", alpha=0.85)
axes[1].set_title("요일별 평균 일매출 (백만원) — v2 weekday trend 반영")
axes[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
out = RESULTS_DIR / "01_quarterly_trend.png"   # 파일명 호환 유지
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 4) Naive Baseline (일 단위)
# - lag1 (어제),  lag7 (지난주 같은 요일),  group mean (행정동 평균)

# %%
ps = panel.sort_values([KEY_DONG, KEY_DATE]).reset_index(drop=True)
g = ps.groupby(KEY_DONG)[LABEL]
ps["pred_lag1"] = g.shift(1)
ps["pred_lag7"] = g.shift(7)
ps["pred_groupmean"] = ps.groupby(KEY_DONG)[LABEL].transform("mean")
ev = ps.dropna(subset=[LABEL, "pred_lag7"])
print(f"평가 가능 행: {len(ev):,}")

baselines = {
    "Naive lag1 (어제)":        regression_metrics(ev[LABEL], ev["pred_lag1"]),
    "Naive lag7 (지난주 동요일)": regression_metrics(ev[LABEL], ev["pred_lag7"]),
    "Group mean (행정동 평균)":  regression_metrics(ev[LABEL], ev["pred_groupmean"]),
}
print("\n=== Naive Baseline 성능 (일 단위) ===")
print(pd.DataFrame(baselines).T.round(4))

# %% [markdown]
# ## 5) 저장

# %%
panel = panel.drop(columns=[c for c in ["_dow"] if c in panel.columns])
out_panel = PROCESSED_DIR / "panel_master_daily.parquet"
panel.to_parquet(out_panel, index=False)
print(f"[SAVED] {out_panel}  ({len(panel):,} rows)")
with (RESULTS_DIR / "01_naive_baseline_metrics.json").open("w", encoding="utf-8") as f:
    json.dump(baselines, f, indent=2, ensure_ascii=False)
print("[SAVED] results/01_naive_baseline_metrics.json")
print("\n[OK] 다음: notebooks/02_feature_engineering.py")
