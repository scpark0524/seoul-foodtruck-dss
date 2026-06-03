"""
서비스 서빙 계층 (★ 일자 DAILY + 실시간) — 입지(+날짜) → 추천 메뉴 + 일 예상매출.

웹/API 호출 함수:
  - recommend_menu(dong, date)      : 한 입지·날짜의 업종별 일 예상매출 → 추천 메뉴
  - predict_sales(dong, cat, date)  : (입지, 업종, 날짜) 일 예상매출(원)
  - rank_locations(cat, date, k)    : 한 업종·날짜의 일 예상매출 Top-K 입지
  - location_report(dong, date)     : 추천메뉴 + 예상매출 + 실시간 현황(rt_*)

설계:
  - 모델: models/xgb_location_daily.*  (입지×업종, self-lag 없음)
  - 입력 피처 = 입지 구조(최신 분기) + 캘린더(날짜) + 일별 외부(기상/공휴일/대기질/생활인구)
  - 실시간(rt_*)은 processed/realtime_dong_daily.parquet 에서 '현황'으로 함께 제공
    (모델 피처로의 결합은 누적 충분 후 재학습 단계에서)
"""
from __future__ import annotations
import json, pickle, functools
from pathlib import Path
import numpy as np
import pandas as pd

from .config import MODELS_DIR, PROCESSED_DIR, FT_CATEGORIES, FOODTRUCK_CATEGORIES, KEY_DONG, KEY_TIME

_CAT_REAL = ["cat_store_cnt", "cat_similar_cnt", "cat_open_rate",
             "cat_close_rate", "cat_franchise_cnt"]


@functools.lru_cache(maxsize=1)
def _load():
    with open(MODELS_DIR / "xgb_location_daily.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "location_daily_feature_cols.json", encoding="utf-8") as f:
        fcols = json.load(f)
    struct = _structural_base(fcols)              # (dong×업종) 구조 피처 + dong_nm
    ext = _external_sources()                     # 일별 외부 캐시
    rt = _realtime_table()                        # 실시간 현황
    return model, fcols, struct, ext, rt


# ---- 구조 피처 베이스 (최신 분기, dong×10업종 격자) ----
def _structural_base(fcols):
    feat = pd.read_parquet(PROCESSED_DIR / "features_location.parquet")
    q = int(feat[KEY_TIME].max())
    fq = feat[feat[KEY_TIME] == q].copy()
    struct_cols = [c for c in fcols if c in fq.columns]   # 분기 구조 피처만
    cat_cols = [c for c in struct_cols if c.startswith("cat_")]
    dong_cols = [c for c in struct_cols if c not in cat_cols]
    dong_attrs = fq[[KEY_DONG, "dong_nm"] + dong_cols].drop_duplicates(KEY_DONG).set_index(KEY_DONG)
    grid = pd.MultiIndex.from_product([dong_attrs.index.tolist(), FT_CATEGORIES],
                                      names=[KEY_DONG, "category"]).to_frame(index=False)
    grid = grid.merge(dong_attrs.reset_index(), on=KEY_DONG, how="left")
    real = [c for c in _CAT_REAL if c in fq.columns]
    grid = grid.merge(fq[[KEY_DONG, "category"] + real], on=[KEY_DONG, "category"], how="left")
    for c in real: grid[c] = grid[c].fillna(0)
    for c in FT_CATEGORIES:
        col = f"cat_is_{c}"
        if col in struct_cols: grid[col] = (grid["category"] == c).astype(int)
    return grid


# ---- 일별 외부 소스 (캐시) ----
def _external_sources():
    from .data_loader import load_one, load_living_pop_daily
    from .features import normalize_facility, dong_to_gu_map
    from .features_daily import (aggregate_weather_daily, aggregate_holidays_daily, aggregate_air_daily)
    wx = aggregate_weather_daily(load_one("weather"))
    hol = aggregate_holidays_daily(load_one("holidays"))
    d2g = dong_to_gu_map(load_one("facility"))
    air = aggregate_air_daily(load_one("air"), d2g)
    lp = load_living_pop_daily()
    return {"wx": wx, "hol": hol, "air": air, "lp": lp}


def _realtime_table():
    p = PROCESSED_DIR / "realtime_dong_daily.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ---- 캘린더 피처 (단일 날짜) ----
def _calendar_row(date: pd.Timestamp) -> dict:
    d = pd.Timestamp(date)
    return {
        "year": d.year, "month": d.month, "day": d.day, "dow": d.dayofweek,
        "is_weekend": int(d.dayofweek >= 5), "doy": d.dayofyear,
        "woy": int(d.isocalendar().week), "q": d.quarter, "year_norm": d.year - 2021,
        "month_sin": np.sin(2*np.pi*d.month/12), "month_cos": np.cos(2*np.pi*d.month/12),
        "dow_sin": np.sin(2*np.pi*d.dayofweek/7), "dow_cos": np.cos(2*np.pi*d.dayofweek/7),
        "doy_sin": np.sin(2*np.pi*d.dayofyear/365), "doy_cos": np.cos(2*np.pi*d.dayofyear/365),
    }


def _predict_grid(date) -> pd.DataFrame:
    model, fcols, struct, ext, rt = _load()
    d = pd.Timestamp(date); ymd = int(d.strftime("%Y%m%d"))
    g = struct.copy()
    for k, v in _calendar_row(d).items(): g[k] = v
    # 일별 외부 (해당 날짜) — 없으면 NaN (모델이 결측 처리)
    wx = ext["wx"][ext["wx"]["date"] == d.normalize()]
    if len(wx): 
        for c in [x for x in wx.columns if x != "date"]: g[c] = wx[c].iloc[0]
    hol = ext["hol"][ext["hol"]["date"] == d.normalize()]
    if len(hol):
        for c in [x for x in hol.columns if x != "date"]: g[c] = hol[c].iloc[0]
    air = ext["air"][ext["air"]["date"] == d.normalize()][[KEY_DONG] + [c for c in ext["air"].columns if c.startswith("air_")]]
    if len(air): g = g.merge(air, on=KEY_DONG, how="left")
    lp = ext["lp"][ext["lp"]["ymd"] == ymd][[KEY_DONG] + [c for c in ext["lp"].columns if c.startswith("livpop")]]
    if len(lp): g = g.merge(lp, on=KEY_DONG, how="left")
    # feature_cols 순서 정렬 (없는 컬럼은 NaN)
    X = g.reindex(columns=fcols)
    g["pred_sales"] = np.expm1(model.predict(X))
    return g


def predict_sales(dong: str, category: str, date) -> float:
    g = _predict_grid(date); dong = str(dong).zfill(8)
    r = g[(g[KEY_DONG] == dong) & (g["category"] == category)]
    if r.empty: raise ValueError(f"없음: {dong}/{category}")
    return float(r["pred_sales"].iloc[0])


def recommend_menu(dong: str, date, top_n: int = 3) -> pd.DataFrame:
    g = _predict_grid(date); dong = str(dong).zfill(8)
    r = g[(g[KEY_DONG] == dong) & (g["category"].isin(FOODTRUCK_CATEGORIES))]
    r = r.sort_values("pred_sales", ascending=False).head(top_n).reset_index(drop=True)
    r["예상_일매출_백만원"] = (r["pred_sales"] / 1e6).round(2)
    r["rank"] = r.index + 1
    return r[["rank", "category", "예상_일매출_백만원"]]


def rank_locations(category: str, date, top_k: int = 10) -> pd.DataFrame:
    g = _predict_grid(date)
    r = g[g["category"] == category].sort_values("pred_sales", ascending=False).head(top_k).reset_index(drop=True)
    r["예상_일매출_백만원"] = (r["pred_sales"] / 1e6).round(2)
    r["rank"] = r.index + 1
    return r[["rank", KEY_DONG, "dong_nm", "예상_일매출_백만원"]]


def realtime_status(dong: str) -> dict | None:
    _, _, _, _, rt = _load()
    if rt is None: return None
    dong = str(dong).zfill(8)
    r = rt[rt[KEY_DONG] == dong]
    if r.empty: return None
    last = r.sort_values("date").iloc[-1]
    out = {"기준일": str(pd.Timestamp(last["date"]).date())}
    for c in ["rt_cmrcl_lvl", "rt_pay_cnt", "rt_pay_amt"]:
        if c in last and pd.notna(last[c]): out[c] = float(last[c])
    return out


def location_report(dong: str, date=None, top_n: int = 3) -> dict:
    d = pd.Timestamp(date) if date is not None else pd.Timestamp.today().normalize()
    dong = str(dong).zfill(8)
    rec = recommend_menu(dong, d, top_n)
    g = _predict_grid(d); name = g[g[KEY_DONG] == dong]["dong_nm"].iloc[0] if (g[KEY_DONG]==dong).any() else dong
    return {
        "행정동코드": dong, "행정동명": str(name), "날짜": str(d.date()),
        "요일": ["월","화","수","목","금","토","일"][d.dayofweek],
        "추천메뉴": rec.to_dict(orient="records"),
        "최고_예상일매출_백만원": float(rec["예상_일매출_백만원"].iloc[0]),
        "실시간현황": realtime_status(dong),
    }
