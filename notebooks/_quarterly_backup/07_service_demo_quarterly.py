# %% [markdown]
# # STEP 7 — 서비스 데모: 추천 메뉴 + 예상 매출 + 최적입지  (★ 신규)
#
# `src/serve.py` 를 호출해 실시간 서비스 응답을 재현하고, 웹 데모용 데이터를 export.
# 산출물:
#   - results/07_serving_predictions.csv  (행정동×업종 전체 예측)
#   - results/07_best_menu_by_dong.csv    (행정동별 추천 메뉴 1~3위)
#   - service_demo/predictions.json       (정적 웹 데모용)

# %%
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from src.config import RESULTS_DIR, FT_CATEGORIES, KEY_DONG
from src import serve

# %% [markdown]
# ## 1) 예시 질의 — 입지 입력 → 추천 메뉴

# %%
for d in ["11110530", "11680640", "11440660"]:   # 사직동, 역삼1동, 서교동
    rep = serve.location_report(d, top_n=3)
    print(f"[{rep['행정동명']}({d})] 추천메뉴:",
          ", ".join(f"{m['category']}({m['예상매출_분기_백만원']:.0f}백만)" for m in rep["추천메뉴"]))

# %% [markdown]
# ## 2) 업종별 최적입지

# %%
for c in ["커피-음료", "한식음식점", "분식전문점"]:
    top = serve.rank_locations(c, top_k=5)
    print(f"\n[{c}] 최적입지 Top5:")
    print(top.to_string(index=False))

# %% [markdown]
# ## 3) 전체 예측 테이블 export (CSV)

# %%
t = serve._table()[[KEY_DONG, "dong_nm", "category", "pred_sales"]].copy()
t["pred_sales_백만원"] = (t["pred_sales"] / 1e6).round(1)
t = t.sort_values([KEY_DONG, "pred_sales"], ascending=[True, False])
out_csv = RESULTS_DIR / "07_serving_predictions.csv"
t.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"[SAVED] {out_csv}  ({len(t):,} 행)")

# 행정동별 추천 1~3위
best = (t.groupby(KEY_DONG).head(3)
          .assign(rank=lambda d: d.groupby(KEY_DONG).cumcount() + 1))
best_out = RESULTS_DIR / "07_best_menu_by_dong.csv"
best.to_csv(best_out, index=False, encoding="utf-8-sig")
print(f"[SAVED] {best_out}")

# %% [markdown]
# ## 4) 정적 웹 데모용 JSON

# %%
demo_dir = ROOT / "service_demo"; demo_dir.mkdir(exist_ok=True)
payload = {"categories": FT_CATEGORIES, "dongs": []}
for code, g in t.groupby(KEY_DONG):
    payload["dongs"].append({
        "code": code, "name": str(g["dong_nm"].iloc[0]),
        "menus": [{"category": r["category"], "sales_mil": float(round(r["pred_sales"]/1e6, 1))}
                  for _, r in g.iterrows()],
    })
with open(demo_dir / "predictions.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False)
with open(demo_dir / "predictions.js", "w", encoding="utf-8") as f:
    f.write("window.PRED=" + json.dumps(payload, ensure_ascii=False) + ";")  # file:// 직접열기용
print(f"[SAVED] {demo_dir/'predictions.json'}  (행정동 {len(payload['dongs'])}개)")
print("\n[OK] 웹 데모: service_demo/index.html 을 브라우저로 열기")
