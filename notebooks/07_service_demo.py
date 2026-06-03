# %% [markdown]
# # STEP 7 — 서비스 데모 (★ DAILY): 날짜·요일별 추천메뉴 + 일 예상매출
#
# `src/serve.py`(일자+실시간) 호출 → 요일별 예측을 웹 데모용으로 export.
# 산출물:
#   - results/07_daily_predictions.csv          (요일×행정동×업종 일 예상매출)
#   - service_demo/predictions_daily.js         (정적 웹 데모용; 요일 선택)

# %%
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from src.config import RESULTS_DIR, FT_CATEGORIES, FOODTRUCK_CATEGORIES, KEY_DONG
from src import serve

# 대표 1주 (월~일) — 요일 효과 시연용
WEEK = {"월":"2025-06-09","화":"2025-06-10","수":"2025-06-11","목":"2025-06-12",
        "금":"2025-06-13","토":"2025-06-14","일":"2025-06-15"}

# %% [markdown]
# ## 1) 예시 — 평일(금) vs 주말(일)
# %%
for d in ["2025-06-13","2025-06-15"]:
    rep = serve.location_report("11680640", d)
    print(f"[역삼1동 {d}({rep['요일']})] " +
          ", ".join(f"{m['category']}({m['예상_일매출_백만원']:.0f}백만/일)" for m in rep["추천메뉴"]))

# %% [markdown]
# ## 2) 요일별 전체 예측 → CSV + 웹 데모 JSON
# %%
rows = []
payload = {"categories": FOODTRUCK_CATEGORIES, "weekdays": list(WEEK.keys()), "byday": {}}
for wd, d in WEEK.items():
    g = serve._predict_grid(d)[[KEY_DONG, "dong_nm", "category", "pred_sales"]].copy()
    g = g[g["category"].isin(FOODTRUCK_CATEGORIES)]
    g["weekday"] = wd; g["date"] = d
    rows.append(g)
    day = {}
    for code, sub in g.groupby(KEY_DONG):
        day[code] = {"name": str(sub["dong_nm"].iloc[0]),
                     "menus": [{"category": r["category"], "sales_mil": round(r["pred_sales"]/1e6, 2)}
                               for _, r in sub.iterrows()]}
    payload["byday"][wd] = day
allrows = pd.concat(rows, ignore_index=True)
allrows["일예상매출_백만원"] = (allrows["pred_sales"]/1e6).round(2)
allrows.to_csv(RESULTS_DIR/"07_daily_predictions.csv", index=False, encoding="utf-8-sig")
print(f"[SAVED] results/07_daily_predictions.csv ({len(allrows):,}행)")

demo = ROOT/"service_demo"; demo.mkdir(exist_ok=True)
with open(demo/"predictions_daily.js","w",encoding="utf-8") as f:
    f.write("window.PRED="+json.dumps(payload,ensure_ascii=False)+";")
print(f"[SAVED] service_demo/predictions_daily.js (요일 {len(WEEK)} × 행정동)")
print("\n[OK] 웹 데모: service_demo/index.html (요일 선택 가능)")
