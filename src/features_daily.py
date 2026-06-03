"""
일자(DAILY) 단위 feature engineering — 분기 추정매출을 일자로 분해한 파일 기준.

타깃: y_log_sales = log1p(당월_매출_금액=그 날의 일매출)
패널: (adstr_cd × date) — 일자 파일이 이미 dense 하므로 grid expand 불필요.

피처 구성
  A. 캘린더      (date 에서 파생, 전 기간 가용)   year/month/dow/주말/계절 sin·cos ...
  B. 매출 lag    (과거 일매출)                    lag1/lag7/lag28/lag364/roll7/roll28/growth
  C. 기상(일)    weather  by date                 2020~2024
  D. 공휴일(일)  holiday  by date                 2020~2024 (캘린더 A 는 전 기간)
  E. 대기질(일)  air      by (gu→dong, date)       2020~2024
  F. 생활인구(일) living_pop by (dong, ymd)        2023~2025
  G. 소비/상권변화/점포  분기→broadcast (dong×yyqu)
  H. 지하철/버스         월→broadcast (dong×yyyymm)
  I. 시설/집객          정적 (dong)

결측은 NaN 그대로 (XGBoost 자동 처리). 분해 매출 컬럼(요일/시간대/성별/연령)은 leakage 라 제외.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import KEY_DONG, KEY_TIME, KEY_DATE, KEY_YMD, LABEL, LEAKAGE_PREFIXES
from .features import (
    yyqu_from_yyyymm, yyqu_from_date,
    aggregate_consumption, aggregate_trdar_change, aggregate_storecount,
    normalize_facility, dong_to_gu_map, build_dong_name_to_cd,
)


# ============================================
# 0) 마스터 패널 (일자)
# ============================================
def build_master_panel_daily(sales_daily: pd.DataFrame) -> pd.DataFrame:
    """load_sales_daily() 결과 → (adstr_cd × date) 패널 + 타깃 + leakage 제거."""
    drop = [c for c in sales_daily.columns
            if any(c.startswith(p) for p in LEAKAGE_PREFIXES)]
    keep = [c for c in sales_daily.columns if c not in drop]
    p = sales_daily[keep].copy()
    p[LABEL] = np.log1p(p["sales_ft"].clip(lower=0))
    p = p.sort_values([KEY_DONG, KEY_DATE]).reset_index(drop=True)
    return p


# ============================================
# A. 캘린더 피처 (전 기간 가용 — date 에서 직접 파생)
# ============================================
def add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    d = p[KEY_DATE].dt
    p["year"]  = d.year
    p["month"] = d.month
    p["day"]   = d.day
    p["dow"]   = d.dayofweek            # 0=월 ... 6=일
    p["is_weekend"] = (p["dow"] >= 5).astype(int)
    p["doy"]   = d.dayofyear
    p["woy"]   = d.isocalendar().week.astype(int)
    p["q"]     = d.quarter
    p["yyqu"]   = p["year"] * 10 + p["q"]                 # 분기 broadcast 키
    p["yyyymm"] = p["year"] * 100 + p["month"]            # 월 broadcast 키
    p["year_norm"] = p["year"] - 2021
    # 주기성 인코딩
    p["month_sin"] = np.sin(2*np.pi*p["month"]/12)
    p["month_cos"] = np.cos(2*np.pi*p["month"]/12)
    p["dow_sin"]   = np.sin(2*np.pi*p["dow"]/7)
    p["dow_cos"]   = np.cos(2*np.pi*p["dow"]/7)
    p["doy_sin"]   = np.sin(2*np.pi*p["doy"]/365)
    p["doy_cos"]   = np.cos(2*np.pi*p["doy"]/365)
    return p


# ============================================
# B. 매출 lag / rolling (일 단위, 과거 정보만)
# ============================================
def add_lag_features_daily(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values([KEY_DONG, KEY_DATE]).copy()
    g = p.groupby(KEY_DONG)[LABEL]
    p["sales_lag1"]   = g.shift(1)
    p["sales_lag7"]   = g.shift(7)
    p["sales_lag28"]  = g.shift(28)
    p["sales_lag364"] = g.shift(364)                      # 작년 같은 요일 근방
    p["sales_roll7"]  = g.shift(1).rolling(7,  min_periods=1).mean()
    p["sales_roll28"] = g.shift(1).rolling(28, min_periods=1).mean()
    p["sales_roll7_std"]  = g.shift(1).rolling(7, min_periods=2).std()
    p["sales_growth_wow"] = g.shift(1) - g.shift(8)        # 전주 대비 로그차분 (=log비율, inf 방지)
    return p


# ============================================
# C. 기상 (일별) — date 기준
# ============================================
def aggregate_weather_daily(weather_raw: pd.DataFrame) -> pd.DataFrame:
    df = weather_raw.copy()
    df[KEY_DATE] = pd.to_datetime(df["date"], errors="coerce")
    keep = {"temp_avg": "wx_temp_avg", "temp_min": "wx_temp_min",
            "temp_max": "wx_temp_max", "precip": "wx_precip",
            "humidity_avg": "wx_humidity", "sunshine_hr": "wx_sunshine",
            "wind_avg": "wx_wind", "snow_new": "wx_snow"}
    keep = {k: v for k, v in keep.items() if k in df.columns}
    out = df.groupby(KEY_DATE)[list(keep)].mean().rename(columns=keep).reset_index()
    out["wx_is_rain"] = (out["wx_precip"] > 1).astype(int) if "wx_precip" in out else 0
    if "wx_temp_avg" in out:
        out["wx_nice"] = ((out["wx_temp_avg"] > 15) & (out["wx_temp_avg"] < 28)).astype(int)
    return out


# ============================================
# D. 공휴일 (일별) — 파일이 이미 일 단위
# ============================================
def aggregate_holidays_daily(hol_raw: pd.DataFrame) -> pd.DataFrame:
    df = hol_raw.copy()
    df[KEY_DATE] = pd.to_datetime(df["date"], errors="coerce")
    cand = {"is_holiday": "hol_is_holiday", "is_dayoff": "hol_is_dayoff",
            "holiday_eve": "hol_eve", "is_sandwich": "hol_sandwich",
            "holiday_streak": "hol_streak", "is_long_weekend": "hol_long_weekend"}
    cand = {k: v for k, v in cand.items() if k in df.columns}
    out = df[[KEY_DATE] + list(cand)].rename(columns=cand)
    return out.drop_duplicates(subset=[KEY_DATE])


# ============================================
# E. 대기질 (일별, 구) → 행정동 매핑
# ============================================
def aggregate_air_daily(air_raw: pd.DataFrame, dong_to_gu: dict[str, str] | None = None) -> pd.DataFrame:
    df = air_raw.copy()
    df[KEY_DATE] = pd.to_datetime(df["date"], errors="coerce")
    cand = {"PM10": "air_pm10", "PM25": "air_pm25", "O3": "air_o3",
            "NO2": "air_no2", "PM10_bad": "air_pm10_bad"}
    cand = {k: v for k, v in cand.items() if k in df.columns}
    gu_daily = df.groupby(["gu", KEY_DATE])[list(cand)].mean().rename(columns=cand).reset_index()
    if not dong_to_gu:
        return gu_daily.rename(columns={"gu": "_gu"})
    mp = pd.DataFrame(list(dong_to_gu.items()), columns=[KEY_DONG, "_gu"])
    return mp.merge(gu_daily, left_on="_gu", right_on="gu", how="left").drop(columns=["_gu", "gu"])


# ============================================
# H. 지하철·버스 (월별, 행정동) — 분기 아닌 '월' 단위로 집계 (broadcast 용)
# ============================================
def _aggregate_monthly_dong_ym(df: pd.DataFrame, time_col: str, prefix: str,
                               dong_name_col: str, dong_name_to_cd: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df[KEY_DONG] = df[dong_name_col].astype(str).str.strip().map(dong_name_to_cd)
    df = df[df[KEY_DONG].notna()]
    df["yyyymm"] = pd.to_numeric(df[time_col], errors="coerce").astype("Int64")
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and c not in ("yyyymm",)]
    df[f"{prefix}_total"] = df[num_cols].sum(axis=1)
    lunch = [c for c in num_cols if any(t in c for t in ["11시", "12시", "13시"])]
    dinner = [c for c in num_cols if any(t in c for t in ["18시", "19시", "20시"])]
    if lunch:  df[f"{prefix}_lunch"]  = df[lunch].sum(axis=1)
    if dinner: df[f"{prefix}_dinner"] = df[dinner].sum(axis=1)
    agg = [c for c in df.columns if c.startswith(f"{prefix}_")]
    return df.groupby([KEY_DONG, "yyyymm"])[agg].mean().reset_index()


def aggregate_subway_monthly(subway_raw, dong_name_to_cd):
    return _aggregate_monthly_dong_ym(subway_raw, "사용월", "subway", "행정동명", dong_name_to_cd)


def aggregate_bus_monthly(bus_raw, dong_name_to_cd):
    return _aggregate_monthly_dong_ym(bus_raw, "사용년월", "bus", "행정동명", dong_name_to_cd)


# ============================================
# 통합 빌더 (일자)
# ============================================
def merge_all_features_daily(
    panel: pd.DataFrame,
    weather_d=None, holiday_d=None, air_d=None, livpop_d=None,
    consumption_q=None, trdar_q=None, storecount_q=None,
    subway_m=None, bus_m=None, facility_static=None,
) -> pd.DataFrame:
    p = panel.copy()
    # 일별 결합
    for df, on in [(weather_d, [KEY_DATE]), (holiday_d, [KEY_DATE]),
                   (air_d, [KEY_DONG, KEY_DATE]), (livpop_d, [KEY_DONG, KEY_YMD])]:
        if df is not None and len(df) > 0:
            p = p.merge(df, on=on, how="left")
    # 분기 broadcast
    for df in [consumption_q, trdar_q, storecount_q]:
        if df is not None and len(df) > 0:
            d = df.rename(columns={KEY_TIME: "yyqu"}) if KEY_TIME in df.columns else df
            p = p.merge(d, on=[KEY_DONG, "yyqu"], how="left")
    # 월 broadcast
    for df in [subway_m, bus_m]:
        if df is not None and len(df) > 0:
            p = p.merge(df, on=[KEY_DONG, "yyyymm"], how="left")
    # 정적
    if facility_static is not None and len(facility_static) > 0:
        p = p.merge(facility_static, on=[KEY_DONG], how="left")
    # inf 안전장치 (0 나눗셈/float 오버플로 → NaN, XGBoost 가 결측 처리)
    p = p.replace([np.inf, -np.inf], np.nan)
    return p


# ============================================
# 원클릭 빌더 (02 / 04 공용) — version 별 일자 피처 매트릭스
# ============================================
def build_feature_matrix_daily(version: str = "v2", save: bool = True, verbose: bool = True):
    """일자 피처 매트릭스를 한 번에 빌드. 캐시(parquet) 있으면 로드.

    반환: (feat_df, feature_cols). save=True 면 PROCESSED_DIR 에 parquet 저장.
    """
    import gc, warnings
    warnings.filterwarnings("ignore")
    from .config import PROCESSED_DIR
    from .data_loader import (load_sales_daily, load_living_pop_daily, load_one, standardize_sales)
    from .features import (aggregate_consumption, aggregate_trdar_change, aggregate_storecount,
                           normalize_facility, dong_to_gu_map, build_dong_name_to_cd)

    cache = PROCESSED_DIR / f"features_daily_{version}.parquet"
    if cache.exists():
        if verbose: print(f"  [CACHE] {cache.name} 로드")
        return pd.read_parquet(cache)

    sd = load_sales_daily(version)
    panel = add_lag_features_daily(add_calendar_features(build_master_panel_daily(sd)))
    del sd; gc.collect()
    wx  = aggregate_weather_daily(load_one("weather"))
    hol = aggregate_holidays_daily(load_one("holidays"))
    lp  = load_living_pop_daily()
    fr  = load_one("facility"); fac = normalize_facility(fr); d2g = dong_to_gu_map(fr)
    air = aggregate_air_daily(load_one("air"), d2g); del fr; gc.collect()
    cons  = aggregate_consumption(load_one("consumption"))
    trdar = aggregate_trdar_change(load_one("trdar_change"))
    store = aggregate_storecount(load_one("storecount"))
    ss = standardize_sales(load_one("sales_dong")); n2c = build_dong_name_to_cd(ss); del ss; gc.collect()
    sub = aggregate_subway_monthly(load_one("subway"), n2c)
    bus = aggregate_bus_monthly(load_one("bus"), n2c)
    feat = merge_all_features_daily(panel, weather_d=wx, holiday_d=hol, air_d=air, livpop_d=lp,
        consumption_q=cons, trdar_q=trdar, storecount_q=store, subway_m=sub, bus_m=bus,
        facility_static=fac)
    del panel, wx, hol, lp, air, cons, trdar, store, sub, bus, fac; gc.collect()
    fcast = [c for c in feat.columns if feat[c].dtype == "float64"]
    feat[fcast] = feat[fcast].astype("float32")
    if save:
        feat.to_parquet(cache, index=False)
        if verbose: print(f"  [SAVED] {cache.name}  {feat.shape}")
    return feat
