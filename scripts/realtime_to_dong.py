# -*- coding: utf-8 -*-
"""
실시간 도시데이터(spot 단위) → 행정동(adstr_cd) 단위 일자 집계.

입력 : raw/realtime/{category}/{YYYY-MM-DD}.csv  (collect_realtime.py 누적분)
       - cmrcl       : 상권 결제(AREA_SH_PAYMENT_*), 혼잡도(AREA_CMRCL_LVL), 결제자 인구통계
       - cmrcl_rsb   : 업종(RSB_MID_CTGR)별 결제·가맹점수
       - ppltn       : 실시간 인구(LIVE_PPLTN)
매핑 : scripts/spot_to_dong.csv (AREA_NM → adstr_cd)
출력 : processed/realtime_dong_daily.parquet      (adstr_cd × date)
       processed/realtime_dong_cat_daily.parquet  (adstr_cd × category × date)

→ backtest 통합키(행정동)와 동일 → 모델 서빙/재학습에 바로 결합 가능.

사용:
    python scripts/realtime_to_dong.py
"""
from __future__ import annotations
import sys, glob, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

RT_DIR = ROOT / "raw" / "realtime"
PROC = ROOT / "processed"
MAP_CSV = ROOT / "scripts" / "spot_to_dong.csv"

# 혼잡도 텍스트 → ordinal
LVL_MAP = {"여유":1, "한산한":1, "보통":2, "약간 붐빔":3, "약간붐빔":3, "붐빔":4}
# 실시간 업종(RSB 중분류) → 푸드트럭 적합 업종(backtest category) 매핑
RSB_TO_FT = {
    "한식":"한식음식점", "한식음식점":"한식음식점",
    "커피전문점":"커피-음료", "커피":"커피-음료", "음료":"커피-음료",
    "양식":"양식음식점", "양식음식점":"양식음식점",
    "중식":"중식음식점", "중식음식점":"중식음식점",
    "일식":"일식음식점", "일식음식점":"일식음식점",
    "분식":"분식전문점", "분식전문점":"분식전문점",
    "제과":"제과점", "제과점":"제과점", "베이커리":"제과점",
    "패스트푸드":"패스트푸드점", "패스트푸드점":"패스트푸드점",
    "치킨":"치킨전문점", "치킨전문점":"치킨전문점",
    "호프":"호프-간이주점", "유흥주점":"호프-간이주점", "간이주점":"호프-간이주점",
}


def _load_map():
    m = pd.read_csv(MAP_CSV, dtype={"adstr_cd": str})
    m = m.dropna(subset=["adstr_cd"])
    return dict(zip(m["AREA_NM"], m["adstr_cd"]))


def _read_category(cat: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(RT_DIR / cat / "*.csv")))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, dtype=str))
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _date_col(df: pd.DataFrame) -> pd.Series:
    """fetched_at(ISO) 또는 *_TIME(YYYYMMDD HHMM) 에서 날짜 추출."""
    if "fetched_at" in df.columns:
        return pd.to_datetime(df["fetched_at"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul").dt.normalize().dt.tz_localize(None)
    for c in df.columns:
        if c.endswith("_TIME"):
            return pd.to_datetime(df[c].str[:8], format="%Y%m%d", errors="coerce")
    return pd.to_datetime(pd.Series([pd.NaT]*len(df)))


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def aggregate_cmrcl(spot2dong: dict) -> pd.DataFrame:
    df = _read_category("cmrcl")
    if df.empty: return pd.DataFrame()
    df["adstr_cd"] = df["AREA_NM"].map(spot2dong)
    df = df[df["adstr_cd"].notna()].copy()
    df["date"] = _date_col(df)
    df["rt_cmrcl_lvl"] = df.get("AREA_CMRCL_LVL", pd.Series(index=df.index)).map(LVL_MAP)
    df["rt_pay_cnt"] = _num(df.get("AREA_SH_PAYMENT_CNT"))
    df["rt_pay_amt"] = (_num(df.get("AREA_SH_PAYMENT_AMT_MIN")) + _num(df.get("AREA_SH_PAYMENT_AMT_MAX"))) / 2
    for c in ["CMRCL_20_RATE","CMRCL_30_RATE","CMRCL_MALE_RATE","CMRCL_FEMALE_RATE","CMRCL_PERSONAL_RATE"]:
        if c in df.columns: df["rt_"+c.lower()] = _num(df[c])
    aggcols = [c for c in df.columns if c.startswith("rt_")]
    out = df.groupby(["adstr_cd","date"])[aggcols].mean().reset_index()
    out["ymd"] = out["date"].dt.strftime("%Y%m%d").astype("Int64")
    return out


def aggregate_ppltn(spot2dong: dict) -> pd.DataFrame:
    df = _read_category("ppltn")
    if df.empty: return pd.DataFrame()
    df["adstr_cd"] = df["AREA_NM"].map(spot2dong)
    df = df[df["adstr_cd"].notna()].copy()
    df["date"] = _date_col(df)
    # 인구수: AREA_PPLTN_MIN/MAX 평균
    if "AREA_PPLTN_MIN" in df.columns:
        df["rt_ppltn"] = (_num(df["AREA_PPLTN_MIN"]) + _num(df.get("AREA_PPLTN_MAX"))) / 2
    if "AREA_CONGEST_LVL" in df.columns:
        df["rt_congest"] = df["AREA_CONGEST_LVL"].map(LVL_MAP)
    aggcols = [c for c in df.columns if c.startswith("rt_")]
    if not aggcols: return pd.DataFrame()
    return df.groupby(["adstr_cd","date"])[aggcols].mean().reset_index()


def aggregate_cmrcl_rsb(spot2dong: dict) -> pd.DataFrame:
    """업종(RSB 중분류)별 실시간 결제 → 행정동×업종×일자."""
    df = _read_category("cmrcl_rsb")
    if df.empty:   # v2 통합 파일이면 cmrcl 안에 RSB 컬럼이 있을 수 있음
        df = _read_category("cmrcl")
        if df.empty or "RSB_MID_CTGR" not in df.columns: return pd.DataFrame()
    df["adstr_cd"] = df["AREA_NM"].map(spot2dong)
    df = df[df["adstr_cd"].notna()].copy()
    df["date"] = _date_col(df)
    df["category"] = df["RSB_MID_CTGR"].map(RSB_TO_FT)
    df = df[df["category"].notna()].copy()
    if df.empty: return pd.DataFrame()
    df["rt_rsb_pay_cnt"] = _num(df.get("RSB_SH_PAYMENT_CNT"))
    df["rt_rsb_pay_amt"] = (_num(df.get("RSB_SH_PAYMENT_AMT_MIN")) + _num(df.get("RSB_SH_PAYMENT_AMT_MAX"))) / 2
    df["rt_rsb_mct_cnt"] = _num(df.get("RSB_MCT_CNT"))
    aggcols = [c for c in df.columns if c.startswith("rt_rsb_")]
    return df.groupby(["adstr_cd","category","date"])[aggcols].mean().reset_index()


def main():
    spot2dong = _load_map()
    print(f"[매핑] {len(spot2dong)}개 spot→행정동")
    cm = aggregate_cmrcl(spot2dong)
    pp = aggregate_ppltn(spot2dong)
    rsb = aggregate_cmrcl_rsb(spot2dong)

    dong = cm
    if not pp.empty:
        dong = cm.merge(pp, on=["adstr_cd","date"], how="outer") if not cm.empty else pp
    if not dong.empty:
        PROC.mkdir(exist_ok=True)
        dong.to_parquet(PROC / "realtime_dong_daily.parquet", index=False)
        print(f"[SAVED] realtime_dong_daily.parquet  {dong.shape}  "
              f"(행정동 {dong['adstr_cd'].nunique()}, 일자 {dong['date'].nunique()})")
    else:
        print("[경고] cmrcl/ppltn 누적분 없음 — raw/realtime/ 확인")
    if not rsb.empty:
        rsb.to_parquet(PROC / "realtime_dong_cat_daily.parquet", index=False)
        print(f"[SAVED] realtime_dong_cat_daily.parquet  {rsb.shape}  (업종별 실시간 결제)")
    return dong


if __name__ == "__main__":
    main()
