"""
Feature engineering — backtest_data 실데이터 기준.

14종 데이터를 행정동(adstr_cd) × 분기(yyqu) 패널에 결합.
시간 가용성이 다르므로 부분 결측은 NaN 그대로 둠 (XGBoost 가 자동 처리).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import KEY_DONG, KEY_TIME, LABEL, FT_CATEGORIES


# ============================================
# 시간 변환 유틸
# ============================================
def yyqu_from_yyyymm(yyyymm: int | str) -> int:
    """202401 (YYYYMM int) → 20241 (YYYYQ int)."""
    s = str(int(yyyymm)).zfill(6)
    y, m = int(s[:4]), int(s[4:6])
    q = (m - 1) // 3 + 1
    return y * 10 + q


def yyqu_from_date(date_str: str) -> int:
    """'2024-03-15' → 20241."""
    ts = pd.to_datetime(date_str)
    return ts.year * 10 + ((ts.month - 1) // 3 + 1)


# ============================================
# 0) 마스터 패널 생성 (매출 → 타깃)
# ============================================
def build_master_panel(
    sales_df: pd.DataFrame, all_qu: list[int], all_dongs: list[str] | None = None
) -> pd.DataFrame:
    """
    standardize_sales() 결과를 받아 (adstr_cd × yyqu) 마스터 패널을 만든다.
    푸드트럭 적합 업종만 필터 → 합산 → log1p → 격자로 expand.
    """
    s = sales_df[sales_df[KEY_TIME].isin(all_qu)]
    s = s[s["category"].isin(FT_CATEGORIES)]
    panel_long = (
        s.groupby([KEY_DONG, KEY_TIME])["sales_amt"]
         .sum().reset_index().rename(columns={"sales_amt": "sales_ft"})
    )
    panel_long[LABEL] = np.log1p(panel_long["sales_ft"])

    if all_dongs is None:
        all_dongs = sorted(sales_df[KEY_DONG].dropna().unique())
    full = pd.MultiIndex.from_product(
        [all_dongs, all_qu], names=[KEY_DONG, KEY_TIME]
    ).to_frame(index=False)
    return full.merge(panel_long, on=[KEY_DONG, KEY_TIME], how="left")


# ============================================
# A. 매출 lag / rolling / 시간 피처
# ============================================
def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values([KEY_DONG, KEY_TIME]).copy()
    g = p.groupby(KEY_DONG)[LABEL]
    p["sales_lag1"]  = g.shift(1)
    p["sales_lag2"]  = g.shift(2)
    p["sales_lag4"]  = g.shift(4)
    p["sales_roll4"] = g.shift(1).rolling(4, min_periods=1).mean()
    p["sales_growth_qoq"] = g.pct_change(1)
    p["sales_growth_yoy"] = g.pct_change(4)
    return p


def add_time_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["year"] = (p[KEY_TIME] // 10).astype(int)
    p["q"]    = (p[KEY_TIME] % 10).astype(int)
    p["q_sin"] = np.sin(2 * np.pi * p["q"] / 4)
    p["q_cos"] = np.cos(2 * np.pi * p["q"] / 4)
    p["year_norm"] = p["year"] - 2023
    return p


# ============================================
# B. 소비 (분기, 행정동) — 정상 결합
# ============================================
def aggregate_consumption(consumption_raw: pd.DataFrame) -> pd.DataFrame:
    """소득_구간 별로 분리된 데이터를 행정동×분기 합으로 집계."""
    df = consumption_raw.rename(columns={
        "기준_년분기_코드": KEY_TIME, "행정동_코드": KEY_DONG,
    }).copy()
    df[KEY_DONG] = df[KEY_DONG].astype(str).str.zfill(8)
    df[KEY_TIME] = pd.to_numeric(df[KEY_TIME], errors="coerce").astype("Int64")
    # 소득 구간을 합치고, 음식·여가 위주 컬럼만 유지
    keep = ["지출_총금액", "식료품_지출_총금액", "음식_지출_총금액",
            "여가_문화_지출_총금액", "유흥_지출_총금액"]
    keep = [c for c in keep if c in df.columns]
    out = df.groupby([KEY_DONG, KEY_TIME])[keep].sum().reset_index()
    # 영문화
    out = out.rename(columns={
        "지출_총금액": "cons_total",
        "식료품_지출_총금액": "cons_grocery",
        "음식_지출_총금액": "cons_food",
        "여가_문화_지출_총금액": "cons_leisure",
        "유흥_지출_총금액": "cons_entertain",
    })
    # 비율 피처
    if "cons_food" in out and "cons_total" in out:
        out["cons_food_ratio"] = out["cons_food"] / out["cons_total"].replace(0, np.nan)
    return out


# ============================================
# C. 상권변화지표 (분기, 행정동) — 정상 결합
# ============================================
def aggregate_trdar_change(trdar_raw: pd.DataFrame) -> pd.DataFrame:
    df = trdar_raw.rename(columns={
        "STDR_YYQU_CD": KEY_TIME, "ADSTRD_CD": KEY_DONG,
        "OPR_SALE_MT_AVRG": "trdar_open_mo",
        "CLS_SALE_MT_AVRG": "trdar_close_mo",
        "SU_OPR_SALE_MT_AVRG": "trdar_open_mo_food",
        "SU_CLS_SALE_MT_AVRG": "trdar_close_mo_food",
    }).copy()
    df[KEY_DONG] = df[KEY_DONG].astype(str).str.zfill(8)
    df[KEY_TIME] = pd.to_numeric(df[KEY_TIME], errors="coerce").astype("Int64")
    keep = [KEY_DONG, KEY_TIME, "trdar_open_mo", "trdar_close_mo",
            "trdar_open_mo_food", "trdar_close_mo_food", "TRDAR_CHNGE_IX"]
    keep = [c for c in keep if c in df.columns]
    # 변화 지표는 카테고리 → 간단 인코딩 (HH=정체, LL=쇠퇴 등을 ordinal 로)
    df_out = df[keep].copy()
    if "TRDAR_CHNGE_IX" in df_out.columns:
        ix_map = {"HH": 3, "HL": 2, "LH": 1, "LL": 0}
        df_out["trdar_change_score"] = df_out["TRDAR_CHNGE_IX"].map(ix_map).fillna(2)
        df_out = df_out.drop(columns=["TRDAR_CHNGE_IX"])
    # 행정동×분기 평균 (혹시 중복)
    return df_out.groupby([KEY_DONG, KEY_TIME]).mean(numeric_only=True).reset_index()


# ============================================
# D. 점포 정보 (분기, 행정동, 업종) — 푸드트럭 적합 업종만 + 합
# ============================================
def aggregate_storecount(store_raw: pd.DataFrame) -> pd.DataFrame:
    df = store_raw.rename(columns={
        "기준_년분기_코드": KEY_TIME, "행정동_코드": KEY_DONG,
        "서비스_업종_코드_명": "category",
        "점포_수": "store_cnt", "유사_업종_점포_수": "similar_store_cnt",
        "개업_율": "open_rate", "폐업_률": "close_rate",
        "프랜차이즈_점포_수": "franchise_cnt",
    }).copy()
    df[KEY_DONG] = df[KEY_DONG].astype(str).str.zfill(8)
    df[KEY_TIME] = pd.to_numeric(df[KEY_TIME], errors="coerce").astype("Int64")
    # 푸드트럭 적합 업종만
    ft = df[df["category"].isin(FT_CATEGORIES)]
    out = ft.groupby([KEY_DONG, KEY_TIME]).agg(
        ft_store_cnt=("store_cnt", "sum"),
        ft_franchise_cnt=("franchise_cnt", "sum"),
        ft_open_rate_mean=("open_rate", "mean"),
        ft_close_rate_mean=("close_rate", "mean"),
    ).reset_index()
    # 전체 업종 합산 (경쟁 상권 지표)
    all_stores = df.groupby([KEY_DONG, KEY_TIME]).agg(
        all_store_cnt=("store_cnt", "sum"),
    ).reset_index()
    return out.merge(all_stores, on=[KEY_DONG, KEY_TIME], how="outer")


# ============================================
# E. 지하철·버스 (월별, 행정동) → 분기 합/평균
# ============================================
def _aggregate_monthly_dong(df: pd.DataFrame, time_col: str, dong_col: str,
                             prefix: str,
                             dong_name_col: str | None = None,
                             dong_name_to_cd: dict[str, str] | None = None) -> pd.DataFrame:
    """월별 행정동 시간대 데이터 → 분기 평균(승하차 합).

    지하철·버스 데이터는 코드 체계가 매출(행정안전부 8자리)과 달라
    행정동명("사직동" 등)으로 매핑해야 함. dong_name_col + dong_name_to_cd
    를 주면 행정동명 기반으로 표준 코드 변환.
    """
    df = df.copy()
    if dong_name_col is not None and dong_name_to_cd:
        # 행정동명 → 매출 표준 8자리 코드 매핑 (매칭 안 되는 동은 결측)
        df[KEY_DONG] = df[dong_name_col].astype(str).str.strip().map(dong_name_to_cd)
    else:
        # 폴백: 코드를 그대로 zfill
        df[KEY_DONG] = df[dong_col].astype(int).astype(str).str.zfill(8)
    df = df[df[KEY_DONG].notna()]
    df[KEY_TIME] = df[time_col].apply(yyqu_from_yyyymm).astype("Int64")
    # 시간대별 컬럼 자동 탐지 (승차/하차 모두 합)
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
                and c not in (KEY_TIME, time_col, dong_col)]
    # 행단위 합 = 일평균 승하차 (모든 시간대)
    df[f"{prefix}_total"] = df[num_cols].sum(axis=1)
    # 점심·저녁 시간대 (컬럼명 기반 추정)
    lunch_cols = [c for c in num_cols if any(t in c for t in
                  ["11시", "12시", "13시", "14시", "11-12", "12-13", "13-14"])]
    dinner_cols = [c for c in num_cols if any(t in c for t in
                   ["18시", "19시", "20시", "21시", "18-19", "19-20", "20-21"])]
    if lunch_cols:
        df[f"{prefix}_lunch"] = df[lunch_cols].sum(axis=1)
    if dinner_cols:
        df[f"{prefix}_dinner"] = df[dinner_cols].sum(axis=1)
    agg_cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
    return df.groupby([KEY_DONG, KEY_TIME])[agg_cols].mean().reset_index()


def aggregate_subway(subway_raw: pd.DataFrame,
                     dong_name_to_cd: dict[str, str] | None = None) -> pd.DataFrame:
    return _aggregate_monthly_dong(subway_raw, "사용월", "행정동코드", "subway",
                                    dong_name_col="행정동명",
                                    dong_name_to_cd=dong_name_to_cd)


def aggregate_bus(bus_raw: pd.DataFrame,
                  dong_name_to_cd: dict[str, str] | None = None) -> pd.DataFrame:
    return _aggregate_monthly_dong(bus_raw, "사용년월", "행정동코드", "bus",
                                    dong_name_col="행정동명",
                                    dong_name_to_cd=dong_name_to_cd)


# ============================================
# F. 기상 (일별, 서울 전체) → 분기 통계
# ============================================
def aggregate_weather(weather_raw: pd.DataFrame) -> pd.DataFrame:
    df = weather_raw.copy()
    df[KEY_TIME] = df["date"].apply(yyqu_from_date).astype("Int64")
    # 관측소가 여럿이면 우선 일평균 → 분기 통계
    daily = df.groupby([KEY_TIME, "date"]).agg(
        temp_avg=("temp_avg", "mean"),
        precip=("precip", "mean"),
    ).reset_index()
    out = daily.groupby(KEY_TIME).agg(
        weather_temp_avg=("temp_avg", "mean"),
        weather_temp_std=("temp_avg", "std"),
        weather_rain_days=("precip", lambda x: (x > 1).sum()),
        weather_nice_days=("temp_avg",
                           lambda x: ((x > 18) & (x < 27)).sum()),
    ).reset_index()
    return out


# ============================================
# G. 공휴일 (일별) → 분기 통계
# ============================================
def aggregate_holidays(hol_raw: pd.DataFrame) -> pd.DataFrame:
    df = hol_raw.copy()
    df[KEY_TIME] = df["date"].apply(yyqu_from_date).astype("Int64")
    return df.groupby(KEY_TIME).agg(
        holiday_days=("is_holiday", "sum"),
        weekend_days=("is_weekend", "sum"),
        long_weekend_days=("is_long_weekend", "sum") if "is_long_weekend" in df.columns
                          else ("is_holiday", "sum"),
    ).reset_index()


# ============================================
# H. 대기질 (일별, 구별) → 분기 통계 → 행정동 매핑
# ============================================
def aggregate_air(air_raw: pd.DataFrame, dong_to_gu: dict[str, str] | None = None) -> pd.DataFrame:
    df = air_raw.copy()
    df[KEY_TIME] = df["date"].apply(yyqu_from_date).astype("Int64")
    out_gu = df.groupby(["gu", KEY_TIME]).agg(
        air_pm10=("PM10", "mean"),
        air_pm25=("PM25", "mean"),
        air_pm10_bad_days=("PM10_bad", "sum"),
    ).reset_index()
    if dong_to_gu is None:
        return out_gu.rename(columns={"gu": "_gu_name"})
    # dong_to_gu 가 주어지면 행정동 단위로 확장
    mapping_df = pd.DataFrame(list(dong_to_gu.items()), columns=[KEY_DONG, "_gu"])
    merged = mapping_df.merge(out_gu, left_on="_gu", right_on="gu", how="left").drop(columns=["_gu", "gu"])
    return merged


# ============================================
# I. 시설 (정적, 행정동) — 그대로 사용
# ============================================
def normalize_facility(facility_raw: pd.DataFrame) -> pd.DataFrame:
    df = facility_raw.copy()
    if "dong_code" in df.columns:
        df = df.rename(columns={"dong_code": KEY_DONG})
    df[KEY_DONG] = df[KEY_DONG].astype(str).str.zfill(8)
    # 숫자형만 유지
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep = [KEY_DONG] + [c for c in num_cols if c.startswith(("store_", "fac_"))]
    return df[keep].copy()


# ============================================
# 통합 빌더
# ============================================
def merge_all_features(
    panel: pd.DataFrame,
    consumption_q: pd.DataFrame | None = None,
    trdar_q: pd.DataFrame | None = None,
    storecount_q: pd.DataFrame | None = None,
    subway_q: pd.DataFrame | None = None,
    bus_q: pd.DataFrame | None = None,
    weather_q: pd.DataFrame | None = None,
    holidays_q: pd.DataFrame | None = None,
    air_q: pd.DataFrame | None = None,
    facility_static: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """panel 에 모든 집계 결과를 left join. 결측은 NaN 유지 (XGBoost 자동 처리)."""
    p = panel.copy()
    for df, on in [
        (consumption_q, [KEY_DONG, KEY_TIME]),
        (trdar_q,       [KEY_DONG, KEY_TIME]),
        (storecount_q,  [KEY_DONG, KEY_TIME]),
        (subway_q,      [KEY_DONG, KEY_TIME]),
        (bus_q,         [KEY_DONG, KEY_TIME]),
        (air_q,         [KEY_DONG, KEY_TIME]),
        (weather_q,     [KEY_TIME]),
        (holidays_q,    [KEY_TIME]),
        (facility_static, [KEY_DONG]),
    ]:
        if df is not None and len(df) > 0:
            p = p.merge(df, on=on, how="left")
    return p


# 행정동 코드 → 구 매핑 헬퍼
def dong_to_gu_map(facility_df: pd.DataFrame) -> dict[str, str]:
    """시설 데이터(raw 또는 정규화)에서 (8자리 adstr_cd → 구명) 매핑 생성.
    raw 파일은 'dong_code' 컬럼을, 정규화 후에는 KEY_DONG('adstr_cd')을 가짐 → 둘 다 지원."""
    if facility_df is None or "gu" not in facility_df.columns:
        return {}
    code_col = KEY_DONG if KEY_DONG in facility_df.columns else (
        "dong_code" if "dong_code" in facility_df.columns else None)
    if code_col is None:
        return {}
    codes = facility_df[code_col].astype(str).str.zfill(8)
    return dict(zip(codes, facility_df["gu"]))



def build_dong_name_to_cd(sales_std: pd.DataFrame) -> dict[str, str]:
    """
    standardize_sales() 결과에서 (행정동명 → 8자리 adstr_cd) 매핑 생성.
    지하철·버스 데이터 결합 시 사용.
    """
    if "dong_nm" not in sales_std.columns:
        return {}
    mapping = sales_std[[KEY_DONG, "dong_nm"]].drop_duplicates()
    return dict(zip(mapping["dong_nm"].astype(str).str.strip(), mapping[KEY_DONG]))

