"""
backtest_data/ 폴더에서 14종 데이터를 로드.

핵심 기능:
- macOS NFD vs Python NFC 정규화 호환
- 다중 파일 자동 합치기 (연도/월별로 분할 저장된 경우)
- cp949 인코딩 자동 폴백
- 생활인구 파일의 헤더 어긋남 (33열 vs 32열) 자동 보정
"""
from __future__ import annotations
from pathlib import Path
import unicodedata
import pandas as pd

from .config import DATA_DIR, DATA_SOURCES, KEY_DONG, KEY_TIME, SALES_COL_MAP


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _read_csv_smart(p: Path, **kwargs) -> pd.DataFrame:
    """utf-8 → cp949 폴백. kwargs 는 pd.read_csv 에 전달."""
    try:
        return pd.read_csv(p, low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="cp949", low_memory=False, **kwargs)


def _list_files(key: str) -> list[Path]:
    src = DATA_SOURCES[key]
    sub = DATA_DIR / src["dir"]
    if not sub.exists():
        raise FileNotFoundError(
            f"[{key}] 폴더가 없습니다: {sub}\n"
            f"  → backtest_data/ 폴더 안에 '{src['dir']}' 가 있는지 확인하세요."
        )
    pattern = src["pattern"]
    files = sorted([p for p in sub.iterdir()
                    if p.suffix.lower() == ".csv"
                    and (not pattern or pattern in _nfc(p.name))])
    return files


# ============================================
# 데이터셋별 특별 처리가 필요한 경우의 reader
# ============================================
def _read_living_pop(p: Path) -> pd.DataFrame:
    """
    생활인구 파일은 데이터 행 끝에 빈 필드가 추가로 있어 컬럼 33개,
    헤더는 32개 → pandas 가 첫 컬럼을 자동 index 로 처리해 데이터가 밀림.
    index_col=False 로 해결.
    """
    return _read_csv_smart(p, encoding="utf-8-sig", index_col=False)


SPECIAL_READERS = {
    "living_pop": _read_living_pop,
}


def load_one(key: str) -> pd.DataFrame:
    """단일 데이터셋 로드 (multi=True 이면 모든 파일 concat)."""
    files = _list_files(key)
    if not files:
        raise FileNotFoundError(f"[{key}] 매칭되는 파일이 없습니다.")
    reader = SPECIAL_READERS.get(key, _read_csv_smart)
    src = DATA_SOURCES[key]
    if src["multi"]:
        dfs = [reader(p) for p in files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"  [OK] {key:14s} concat {len(files)}개 파일 → shape={df.shape}")
    else:
        df = reader(files[0])
        print(f"  [OK] {key:14s} {files[0].name} → shape={df.shape}")
    return df


def load_all(keys: list[str] | None = None) -> dict[str, pd.DataFrame]:
    keys = keys or list(DATA_SOURCES.keys())
    out = {}
    for k in keys:
        try:
            out[k] = load_one(k)
        except FileNotFoundError as e:
            print(f"  [SKIP] {k:14s} {str(e).splitlines()[0]}")
    return out


# ============================================
# 매출 데이터 전용 표준화
# ============================================
def standardize_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=SALES_COL_MAP).copy()
    df[KEY_DONG] = df[KEY_DONG].astype(str).str.zfill(8)
    df[KEY_TIME] = pd.to_numeric(df[KEY_TIME], errors="coerce").astype("Int64")
    return df


# ============================================
# 시간 가용성 매트릭스 — 어느 데이터셋이 어느 분기를 커버하는지
# ============================================
DATA_TIME_COVERAGE = {
    # key: (시점 종류, 가용 시작, 가용 끝, 12분기 중 커버, 비고)
    "sales_dong":   ("분기", "2021Q1", "2025Q4", "전체 + 2022 누락", "타깃 — 5_추정매출(행정동)"),
    "storecount":   ("분기", "2020Q1", "2024Q4", "8/12 (2025 결측)", "점포 정보"),
    "consumption":  ("분기", "2019Q1", "2025Q4", "전체", "소득·소비"),
    "trdar_change": ("분기", "2021Q1", "2025Q4", "전체", "상권변화지표"),
    "living_pop":   ("일별",   "2025-01", "2026-04", "4/12 (2025년만)", "생활인구 — 분기 집계 가능"),
    "subway":       ("월별",   "2021-01", "2026-04", "전체", "지하철 행정동 시간대 집계"),
    "bus":          ("월별",   "2021-01", "2026-04", "전체", "버스 행정동 시간대 집계"),
    "weather":      ("일별",   "2020-01", "2024-12", "8/12 (2025 결측)", "관측소 기상"),
    "holidays":     ("일별",   "2020-01", "2024-12", "8/12 (2025 결측)", "공휴일"),
    "air":          ("일별",   "2020-01", "2024-12", "8/12 (2025 결측)", "구별 대기질"),
    "facility":     ("정적",   "—", "—",          "전체",                  "상가업소+집객시설 merged"),
    "census":       ("연별",   "2020", "2024",   "8/12 (2025 결측)",       "총 조사 인구"),
    "resident":     ("월별",   "2025-01", "2026-04", "4/12 (2025년만)",    "주민등록인구 5세별 — pivot"),
}


def print_coverage_matrix() -> None:
    print(f'{"key":14s} {"종류":6s} {"가용 시작":10s} {"가용 끝":10s} {"12분기 커버":20s} 비고')
    print("-" * 100)
    for k, v in DATA_TIME_COVERAGE.items():
        print(f"{k:14s} {v[0]:6s} {v[1]:10s} {v[2]:10s} {v[3]:20s} {v[4]}")


# ============================================================
# ★ 일자(DAILY) 단위 로더  (2026-06 추가)
# ============================================================
from .config import (
    SALES_DAILY_FILES, SALES_DAILY_DIR, SALES_DAILY_COL_MAP,
    KEY_DATE, KEY_YMD,
)


def load_sales_daily(version: str = "v2") -> pd.DataFrame:
    """일자 분해 추정매출 파일 로드 + 표준화.

    version: 'v1'(균등 1/n) 또는 'v2'(요일 트렌드).
    반환 컬럼: adstr_cd, date(Timestamp), ymd(int), dong_nm, sales_ft, sales_cnt,
              + 원본의 분해 컬럼들(요일/시간대/성별/연령) — leakage 라 모델에선 제외.
    """
    if version not in SALES_DAILY_FILES:
        raise ValueError(f"version 은 {list(SALES_DAILY_FILES)} 중 하나여야 합니다.")
    p = DATA_DIR / SALES_DAILY_DIR / SALES_DAILY_FILES[version]
    if not p.exists():
        raise FileNotFoundError(f"[sales_daily/{version}] 파일이 없습니다: {p}")
    df = _read_csv_smart(p, encoding="utf-8-sig")
    df = df.rename(columns=SALES_DAILY_COL_MAP)
    df[KEY_DONG] = df[KEY_DONG].astype(str).str.zfill(8)
    df[KEY_YMD] = pd.to_numeric(df[KEY_YMD], errors="coerce").astype("Int64")
    df[KEY_DATE] = pd.to_datetime(df[KEY_YMD].astype(str), format="%Y%m%d", errors="coerce")
    print(f"  [OK] sales_daily/{version}  {p.name} → shape={df.shape}  "
          f"({df[KEY_DATE].min().date()} ~ {df[KEY_DATE].max().date()})")
    return df


def load_living_pop_daily() -> pd.DataFrame:
    """생활인구(내국인+외국인) parquet 을 (행정동 × 일자) 일별 피처로 집계.

    내국인: 시간대밴드(6) × 성별 × 연령 → 일 총인구 + 점심/저녁밴드 + 20·30대.
    외국인: 시간대(24) × 체류유형 → 일 총외국인.
    커버: 2023-01 ~ 2025-12 (그 외 일자는 결측 → XGBoost 자동 처리).
    """
    from .config import PROCESSED_DIR
    _cache = PROCESSED_DIR / "livpop_daily.parquet"
    if _cache.exists():
        out = pd.read_parquet(_cache)
        print(f"  [CACHE] livpop_daily.parquet → shape={out.shape}")
        return out
    base = DATA_DIR / "1. 서울시 생활인구(내국인)_행정동 단위"
    res_p = base / "내국인_merged_2023_2025.parquet"
    for_p = base / "foreigner_merged_2023_2025.parquet"

    # --- 내국인 ---
    res = pd.read_parquet(res_p)
    res = res.rename(columns={"기준일ID": KEY_YMD, "행정동코드": KEY_DONG, "시간대밴드": "band"})
    res[KEY_DONG] = res[KEY_DONG].astype(str).str.zfill(8)
    male_tot, fem_tot = "남자_합계_생활인구수", "여자_합계_생활인구수"
    res["pop_band_total"] = res[male_tot] + res[fem_tot]
    age_2030 = ["남자_20_29세_생활인구수", "남자_30_39세_생활인구수",
                "여자_20_29세_생활인구수", "여자_30_39세_생활인구수"]
    res["pop_band_2030"] = res[age_2030].sum(axis=1)
    # 일 총인구 = 모든 시간대밴드 합 / 6 (밴드별 평균 체류인구의 일 대표값으로 합산)
    daily_tot = res.groupby([KEY_DONG, KEY_YMD]).agg(
        livpop_resident=("pop_band_total", "sum"),
        livpop_resident_2030=("pop_band_2030", "sum"),
    ).reset_index()
    # 점심(11~14)·저녁(17~21) 밴드
    lunch = res[res["band"] == "11~14"].groupby([KEY_DONG, KEY_YMD])["pop_band_total"].sum() \
              .rename("livpop_lunch").reset_index()
    dinner = res[res["band"] == "17~21"].groupby([KEY_DONG, KEY_YMD])["pop_band_total"].sum() \
              .rename("livpop_dinner").reset_index()
    out = daily_tot.merge(lunch, on=[KEY_DONG, KEY_YMD], how="left") \
                   .merge(dinner, on=[KEY_DONG, KEY_YMD], how="left")

    # --- 외국인 ---
    fr = pd.read_parquet(for_p, columns=["기준일ID", "행정동코드", "총생활인구수"])
    fr = fr.rename(columns={"기준일ID": KEY_YMD, "행정동코드": KEY_DONG})
    fr[KEY_DONG] = fr[KEY_DONG].astype(str).str.zfill(8)
    fr_daily = fr.groupby([KEY_DONG, KEY_YMD])["총생활인구수"].sum() \
                 .rename("livpop_foreign").reset_index()
    out = out.merge(fr_daily, on=[KEY_DONG, KEY_YMD], how="left")

    out[KEY_YMD] = pd.to_numeric(out[KEY_YMD], errors="coerce").astype("Int64")
    try:
        out.to_parquet(_cache, index=False)
    except Exception:
        pass
    print(f"  [OK] living_pop daily 집계 → shape={out.shape} "
          f"(dongs={out[KEY_DONG].nunique()}, days={out[KEY_YMD].nunique()})")
    return out
