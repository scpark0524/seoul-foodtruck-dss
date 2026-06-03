"""
입지 × 업종 (dong × category × quarter) 서비스 코어 모델용 feature engineering.

목적: "특정 입지를 입력하면 → 추천 메뉴(업종) + 예상 매출" 실시간 서비스.
설계 원칙:
  - self-lag 없음 → 기록 없는 신규 후보 입지도 입지속성만으로 예측/추천 가능.
  - 업종(category)을 피처로 넣어, 한 모델이 (입지, 업종) 조합의 매출을 예측.
    → 한 입지의 10개 업종 매출을 모두 예측 → 최상위 = 추천 메뉴.
  - 검증은 leave-dong-out(공간) → "안 본 동네"에 일반화되는지 확인.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import KEY_DONG, KEY_TIME, LABEL, FT_CATEGORIES
from .features import (
    aggregate_consumption, aggregate_trdar_change,
    normalize_facility, build_dong_name_to_cd,
)

CAT_LABEL = "y_log_cat_sales"   # 업종별 분기매출 log1p


# ============================================
# 0) 입지×업종×분기 패널 + 타깃
# ============================================
def build_category_panel(sales_std: pd.DataFrame, quarters: list[int],
                         categories: list[str] | None = None) -> pd.DataFrame:
    """standardize_sales() 결과 → (adstr_cd × category × yyqu) 패널 + 타깃."""
    cats = categories or FT_CATEGORIES
    s = sales_std[sales_std[KEY_TIME].isin(quarters) & sales_std["category"].isin(cats)]
    g = (s.groupby([KEY_DONG, "category", KEY_TIME])
           .agg(cat_sales=("sales_amt", "sum"), cat_cnt=("sales_cnt", "sum"))
           .reset_index())
    g[CAT_LABEL] = np.log1p(g["cat_sales"].clip(lower=0))
    return g


# ============================================
# 1) 업종별 점포(경쟁/시장존재) — dong × category × quarter
# ============================================
def aggregate_storecount_by_category(store_raw: pd.DataFrame) -> pd.DataFrame:
    df = store_raw.rename(columns={
        "기준_년분기_코드": KEY_TIME, "행정동_코드": KEY_DONG,
        "서비스_업종_코드_명": "category",
        "점포_수": "cat_store_cnt", "유사_업종_점포_수": "cat_similar_cnt",
        "개업_율": "cat_open_rate", "폐업_률": "cat_close_rate",
        "프랜차이즈_점포_수": "cat_franchise_cnt",
    }).copy()
    df[KEY_DONG] = df[KEY_DONG].astype(str).str.zfill(8)
    df[KEY_TIME] = pd.to_numeric(df[KEY_TIME], errors="coerce").astype("Int64")
    keep = [KEY_DONG, "category", KEY_TIME, "cat_store_cnt", "cat_similar_cnt",
            "cat_open_rate", "cat_close_rate", "cat_franchise_cnt"]
    keep = [c for c in keep if c in df.columns]
    out = df[df["category"].isin(FT_CATEGORIES)][keep].copy()
    # 동×분기 전체 점포 합 (전반적 상권 밀도)
    allc = df.groupby([KEY_DONG, KEY_TIME]).agg(all_store_cnt=("cat_store_cnt", "sum")).reset_index()
    return out, allc


# ============================================
# 2) 생활인구 분기 평균 (일자 parquet → 분기 집계, 입지속성으로)
# ============================================
def aggregate_livpop_quarterly(livpop_daily: pd.DataFrame) -> pd.DataFrame:
    """load_living_pop_daily() 결과(ymd) → 분기 평균 입지속성."""
    df = livpop_daily.copy()
    ymd = df["ymd"].astype(int).astype(str)
    y = ymd.str[:4].astype(int); m = ymd.str[4:6].astype(int)
    df[KEY_TIME] = (y * 10 + ((m - 1) // 3 + 1)).astype("Int64")
    val_cols = [c for c in df.columns if c.startswith("livpop")]
    return df.groupby([KEY_DONG, KEY_TIME])[val_cols].mean().reset_index()


# ============================================
# 3) 지하철·버스 분기 평균 (입지속성)  — features.py 의 분기 집계 재사용
# ============================================
# (features.aggregate_subway / aggregate_bus 가 이미 분기 단위로 집계)


# ============================================
# 4) 카테고리 원-핫
# ============================================
def add_category_onehot(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    for c in FT_CATEGORIES:
        p[f"cat_is_{c}"] = (p["category"] == c).astype(int)
    return p


# ============================================
# 5) 시간 피처 (분기)
# ============================================
def add_time_features_q(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["year"] = (p[KEY_TIME] // 10).astype(int)
    p["q"] = (p[KEY_TIME] % 10).astype(int)
    p["q_sin"] = np.sin(2*np.pi*p["q"]/4)
    p["q_cos"] = np.cos(2*np.pi*p["q"]/4)
    p["year_norm"] = p["year"] - 2021
    return p


# ============================================
# 6) 통합 빌더
# ============================================
def merge_location_features(
    panel: pd.DataFrame,
    cat_store=None, all_store=None,
    consumption_q=None, trdar_q=None,
    subway_q=None, bus_q=None, livpop_q=None,
    facility_static=None,
) -> pd.DataFrame:
    p = panel.copy()
    # 업종별 점포 (dong×category×quarter)
    if cat_store is not None:
        p = p.merge(cat_store, on=[KEY_DONG, "category", KEY_TIME], how="left")
    # dong×quarter 단위
    for df in [all_store, consumption_q, trdar_q, subway_q, bus_q, livpop_q]:
        if df is not None and len(df) > 0:
            p = p.merge(df, on=[KEY_DONG, KEY_TIME], how="left")
    # 정적
    if facility_static is not None and len(facility_static) > 0:
        p = p.merge(facility_static, on=[KEY_DONG], how="left")
    p = p.replace([np.inf, -np.inf], np.nan)
    return p


# 제외 컬럼 (키/타깃/원매출/식별)
LOC_EXCLUDE = {KEY_DONG, KEY_TIME, "category", "cat_sales", "cat_cnt", CAT_LABEL,
               "dong_nm", "_gu_name"}


# ============================================
# 7) 원클릭 빌더 (06/07/serve 공용)
# ============================================
def build_location_matrix(quarters=None, save=True, verbose=True):
    """입지×업종 피처 매트릭스 빌드 (캐시 있으면 로드)."""
    import warnings; warnings.filterwarnings("ignore")
    from .config import ALL_QU_AVAILABLE, PROCESSED_DIR
    from .data_loader import load_one, standardize_sales, load_living_pop_daily
    from .features import (aggregate_consumption, aggregate_trdar_change,
                           aggregate_subway, aggregate_bus, normalize_facility,
                           build_dong_name_to_cd)
    cache = PROCESSED_DIR / "features_location.parquet"
    if cache.exists():
        if verbose: print(f"  [CACHE] {cache.name} 로드")
        return pd.read_parquet(cache)
    QS = quarters or ALL_QU_AVAILABLE
    ss = standardize_sales(load_one("sales_dong"))
    panel = add_category_onehot(add_time_features_q(build_category_panel(ss, QS)))
    cat_store, all_store = aggregate_storecount_by_category(load_one("storecount"))
    cons = aggregate_consumption(load_one("consumption"))
    trdar = aggregate_trdar_change(load_one("trdar_change"))
    fac = normalize_facility(load_one("facility"))
    n2c = build_dong_name_to_cd(ss)
    sub = aggregate_subway(load_one("subway"), n2c)
    bus = aggregate_bus(load_one("bus"), n2c)
    lpq = aggregate_livpop_quarterly(load_living_pop_daily())
    feat = merge_location_features(panel, cat_store=cat_store, all_store=all_store,
        consumption_q=cons, trdar_q=trdar, subway_q=sub, bus_q=bus, livpop_q=lpq,
        facility_static=fac)
    # 동 이름 부착 (서비스 표시용)
    dn = ss[[KEY_DONG, "dong_nm"]].drop_duplicates(KEY_DONG)
    feat = feat.merge(dn, on=KEY_DONG, how="left")
    if save:
        feat.to_parquet(cache, index=False)
        if verbose: print(f"  [SAVED] {cache.name}  {feat.shape}")
    return feat


def location_feature_cols(feat):
    """모델 입력 피처 컬럼 (self-lag 없음, 키/타깃/식별 제외)."""
    import pandas as pd
    return [c for c in feat.columns
            if c not in LOC_EXCLUDE and pd.api.types.is_numeric_dtype(feat[c])]


# ============================================================
# ★ 일자(DAILY) 입지×업종 — v2 일별 분배비율로 분기 업종매출을 일매출로 분해
#    목적: "입지 + 날짜 입력 → 추천 메뉴 + 일 예상매출" 데일리 서빙.
# ============================================================
DAILY_CAT_LABEL = "y_log_daily_cat_sales"


def build_daily_share(version: str = "v2") -> pd.DataFrame:
    """일자 매출파일에서 (dong, date) 의 '분기내 일 점유율'(share) 계산.
       share[dong,date] = 일매출 / 그 분기 dong 총매출.  → 업종 분해에 사용."""
    from .data_loader import load_sales_daily
    sd = load_sales_daily(version)[[KEY_DONG, "date", "ymd", "sales_ft"]].copy()
    d = sd["date"].dt
    sd[KEY_TIME] = (d.year * 10 + d.quarter).astype("Int64")
    qt = sd.groupby([KEY_DONG, KEY_TIME])["sales_ft"].transform("sum")
    sd["share"] = sd["sales_ft"] / qt.replace(0, np.nan)
    return sd[[KEY_DONG, KEY_TIME, "date", "ymd", "share"]]


def build_location_daily_matrix(n_dates_per_quarter: int = 12, version: str = "v2",
                                save: bool = True, verbose: bool = True, seed: int = 42):
    """(dong × category × date) 일자 학습 매트릭스.
       타깃 = log1p(분기 업종매출 × 일 share).  날짜는 분기별로 표본추출(메모리 관리).
       피처 = 입지×업종 구조 + 일 캘린더 + 일별 외부(기상/공휴일/생활인구/대기질). self-lag 없음."""
    import gc, warnings; warnings.filterwarnings("ignore")
    from .config import ALL_QU_AVAILABLE, PROCESSED_DIR
    from .data_loader import load_one, load_living_pop_daily
    from .features_daily import (add_calendar_features, aggregate_weather_daily,
        aggregate_holidays_daily, aggregate_air_daily)
    from .features import normalize_facility, dong_to_gu_map

    cache = PROCESSED_DIR / f"features_location_daily_{version}.parquet"
    if cache.exists():
        if verbose: print(f"  [CACHE] {cache.name} 로드")
        return pd.read_parquet(cache)

    # 1) 입지×업종 구조 피처(분기) — 캐시 재사용
    qmat = build_location_matrix(ALL_QU_AVAILABLE, save=True, verbose=verbose)
    # 분기 생활인구는 일자 생활인구와 중복되므로 제거(일자 것만 사용)
    drop_q = {"cat_sales", "cat_cnt", CAT_LABEL, "dong_nm",
              "year", "q", "q_sin", "q_cos", "year_norm",
              "livpop_resident", "livpop_resident_2030", "livpop_lunch",
              "livpop_dinner", "livpop_foreign"}
    struct_cols = [c for c in qmat.columns if c not in drop_q]
    qmat = qmat[struct_cols + ["cat_sales", "dong_nm"]].copy()

    # 2) 일 share + 분기별 날짜 표본
    share = build_daily_share(version)
    rng = np.random.default_rng(seed)
    picks = []
    for q, g in share[[KEY_TIME, "date", "ymd"]].drop_duplicates("ymd").groupby(KEY_TIME):
        days = g["ymd"].unique()
        k = min(n_dates_per_quarter, len(days))
        picks.extend(rng.choice(days, size=k, replace=False).tolist())
    share_s = share[share["ymd"].isin(picks)].copy()

    # 3) (dong,cat,quarter) × 표본일 → 일 업종매출 타깃
    daily = qmat.merge(share_s, on=[KEY_DONG, KEY_TIME], how="inner")
    daily["daily_cat_sales"] = daily["cat_sales"] * daily["share"]
    daily[DAILY_CAT_LABEL] = np.log1p(daily["daily_cat_sales"].clip(lower=0))
    daily = daily.drop(columns=["cat_sales", "share"])
    del qmat; gc.collect()

    # 4) 일 캘린더
    daily = add_calendar_features(daily)   # date 에서 dow/month/계절 sin·cos 등

    # 5) 일별 외부 결합
    wx = aggregate_weather_daily(load_one("weather"))
    hol = aggregate_holidays_daily(load_one("holidays"))
    fr = load_one("facility"); d2g = dong_to_gu_map(fr); del fr
    air = aggregate_air_daily(load_one("air"), d2g)
    lp = load_living_pop_daily()
    daily = daily.merge(wx, on="date", how="left").merge(hol, on="date", how="left")
    daily = daily.merge(air, on=[KEY_DONG, "date"], how="left")
    daily = daily.merge(lp, on=[KEY_DONG, "ymd"], how="left")
    daily = daily.replace([np.inf, -np.inf], np.nan)
    del wx, hol, air, lp; gc.collect()

    fcast = [c for c in daily.columns if daily[c].dtype == "float64"]
    daily[fcast] = daily[fcast].astype("float32")
    if save:
        daily.to_parquet(cache, index=False)
        if verbose: print(f"  [SAVED] {cache.name}  {daily.shape}")
    return daily


# 일자 모델 제외 컬럼
DAILY_LOC_EXCLUDE = {KEY_DONG, KEY_TIME, "category", "daily_cat_sales", DAILY_CAT_LABEL,
                     "dong_nm", "date", "ymd", "_gu_name", "yyqu", "yyyymm"}


def location_daily_feature_cols(feat):
    import pandas as pd
    return [c for c in feat.columns
            if c not in DAILY_LOC_EXCLUDE and pd.api.types.is_numeric_dtype(feat[c])]
