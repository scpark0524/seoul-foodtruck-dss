# %% [markdown]
# # STEP 2 — EDA + Naive Baseline
#
# **데이터**: backtest_data/5. 추정매출 (행정동, 4개년 파일)
# **기간**: 2023Q1 ~ 2025Q4 (12분기 연속)
# **타깃**: 푸드트럭 적합 10업종 매출 합 → log1p
#
# **실행**: VSCode Jupyter Interactive 에서 셀별 (Shift+Enter)
#         또는 터미널: `python notebooks/01_eda_baseline.py`

# %%
from __future__ import annotations
import sys, json
from pathlib import Path

# 프로젝트 루트를 import path 에 추가
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    DATA_DIR, PROCESSED_DIR, RESULTS_DIR,
    ALL_QU, KEY_DONG, KEY_TIME, LABEL, FT_CATEGORIES,
)
from src.data_loader import load_one, standardize_sales
from src.features import build_master_panel
from src.metrics import regression_metrics, naive_lag, naive_group_mean

plt.rcParams["figure.dpi"] = 110
import platform, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
_FONT = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "DejaVu Sans")
plt.rcParams["font.family"] = _FONT
plt.rcParams["axes.unicode_minus"] = False

print(f"[INFO] DATA_DIR exists = {DATA_DIR.exists()}")
print(f"[INFO] 모델링 12분기 = {ALL_QU}")

# %% [markdown]
# ## 1) 추정매출(행정동) 로드 + 컬럼 표준화

# %%
sales_raw = load_one("sales_dong")
sales = standardize_sales(sales_raw)
print(f"\n표준화 후 shape: {sales.shape}")
print(f"분기 unique ({sales[KEY_TIME].nunique()}개): {sorted(sales[KEY_TIME].dropna().unique().tolist())}")
print(f"행정동 unique: {sales[KEY_DONG].nunique()}")
print(f"업종 unique: {sales['category'].nunique()}")

# %% [markdown]
# ## 2) 12분기 + 푸드트럭 적합 10업종 필터

# %%
sales12 = sales[sales[KEY_TIME].isin(ALL_QU)].copy()
sales_ft = sales12[sales12["category"].isin(FT_CATEGORIES)].copy()
print(f"12분기 필터: {sales12.shape}  → 푸드트럭 업종 {len(FT_CATEGORIES)}종 필터: {sales_ft.shape}")
print(f"\n업종별 누적 매출 (조원, 12분기 합):")
agg = sales_ft.groupby("category")["sales_amt"].sum() / 1e12
print(agg.sort_values(ascending=False).round(2))

# %% [markdown]
# ## 3) 행정동 × 분기 마스터 패널 생성 (타깃 = log1p 매출 합)

# %%
panel = build_master_panel(sales, ALL_QU)
print(f"\n마스터 패널 shape: {panel.shape}  (기대 424×12 = 5,088)")
print(f"결측 행: {panel[LABEL].isna().sum()} ({panel[LABEL].isna().mean()*100:.2f}%)")
print(f"\ny_log_sales 분포:")
print(panel[LABEL].describe().round(3))

# %% [markdown]
# ## 4) 타깃 분포 시각화

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(panel["sales_ft"].dropna() / 1e8, bins=80, color="#444", alpha=0.85)
axes[0].set_title("raw 매출 분포 (롱테일)")
axes[0].set_xlabel("매출 (억원)")
axes[1].hist(panel[LABEL].dropna(), bins=80, color="#c0392b", alpha=0.85)
axes[1].set_title("y_log_sales 분포 (log1p, 정규분포 비슷)")
axes[1].set_xlabel("log(1 + sales_ft)")
for ax in axes: ax.grid(alpha=0.3)
plt.tight_layout()
out = RESULTS_DIR / "01_target_distribution.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 5) 분기별 매출 총합 추세

# %%
qtot = panel.groupby(KEY_TIME)["sales_ft"].sum() / 1e8
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(qtot.index.astype(str), qtot.values, marker="o", linewidth=2, color="#2c3e50")
ax.set_title("서울시 푸드트럭 적합 업종 분기별 매출 총합 (단위: 억원)")
ax.set_xlabel("yyqu (YYYYQ)"); ax.set_ylabel("매출 (억원)")
ax.grid(alpha=0.3)
plt.tight_layout()
out = RESULTS_DIR / "01_quarterly_trend.png"
plt.savefig(out, bbox_inches="tight"); print(f"[SAVED] {out}"); plt.show()

# %% [markdown]
# ## 6) Naive Baseline 3종

# %%
panel_sorted = panel.sort_values([KEY_DONG, KEY_TIME]).reset_index(drop=True)
panel_sorted["pred_lag1"] = naive_lag(panel_sorted, KEY_DONG, KEY_TIME, LABEL, lag=1)
panel_sorted["pred_lag4"] = naive_lag(panel_sorted, KEY_DONG, KEY_TIME, LABEL, lag=4)
panel_sorted["pred_groupmean"] = naive_group_mean(panel_sorted, KEY_DONG, LABEL)

eval_df = panel_sorted.dropna(subset=[LABEL, "pred_lag4"]).copy()
print(f"평가 가능 행: {len(eval_df):,} (lag4 가 생성되는 5분기째부터)")

baselines = {
    "Naive lag1 (전기)":     regression_metrics(eval_df[LABEL], eval_df["pred_lag1"]),
    "Naive lag4 (작년 동분기)": regression_metrics(eval_df[LABEL], eval_df["pred_lag4"]),
    "Group mean (행정동 평균)": regression_metrics(eval_df[LABEL], eval_df["pred_groupmean"]),
}
print("\n=== Naive Baseline 성능 ===")
print(pd.DataFrame(baselines).T.round(4))

# %% [markdown]
# ## 7) 저장

# %%
out_panel = PROCESSED_DIR / "panel_master.parquet"
panel.to_parquet(out_panel, index=False)
print(f"[SAVED] {out_panel}  ({len(panel):,} rows)")

with (RESULTS_DIR / "01_naive_baseline_metrics.json").open("w", encoding="utf-8") as f:
    json.dump(baselines, f, indent=2, ensure_ascii=False)
print(f"[SAVED] results/01_naive_baseline_metrics.json")
print("\n[OK] 다음: notebooks/02_feature_engineering.py")
