"""
프로젝트 전역 설정 — 경로, 분기 리스트, 컬럼 매핑.

⚠️ backtest_data/ 폴더의 실제 구조에 맞춰 작성됨 (2026-05-23 갱신).
   - 행정동 코드: 8자리 (예: 11110515)
   - 시간 코드: 기준_년분기_코드 YYYYQ (예: 20241)
   - 푸드트럭 적합 업종: 실데이터 10종 확정
"""
from __future__ import annotations
from pathlib import Path

# ============================================
# 1) 경로
# ============================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "backtest_data"   # ★ 사용자가 만든 폴더
PROCESSED_DIR = PROJECT_ROOT / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
for d in (PROCESSED_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ============================================
# 2) 시간키 (yyqu = YYYYQ)
#    실데이터의 추정매출(행정동) 분기: 2021Q1~Q4, 2023Q1~Q4, 2024Q1~Q4, 2025Q1~Q4
#    → PDF 발표 (424 행정동 × 12분기 ≈ 5,088 obs) 와 일치하는
#       연속 12분기는 2023+2024+2025
# ============================================
ALL_QU_AVAILABLE = [   # 매출 파일이 존재하는 모든 분기 (2022 추가로 연속 20분기 — 2026-06 갱신)
    20211, 20212, 20213, 20214,
    20221, 20222, 20223, 20224,
    20231, 20232, 20233, 20234,
    20241, 20242, 20243, 20244,
    20251, 20252, 20253, 20254,
]
ALL_QU = [   # ★ 실제 모델링에 사용할 12분기 (연속)
    20231, 20232, 20233, 20234,
    20241, 20242, 20243, 20244,
    20251, 20252, 20253, 20254,
]
# 시간 기반 holdout split: 8 / 2 / 2
TRAIN_QU = [20231, 20232, 20233, 20234, 20241, 20242, 20243, 20244]
VAL_QU   = [20251, 20252]
TEST_QU  = [20253, 20254]

# ============================================
# 3) 푸드트럭 적합 업종 — 실데이터 63개 중 F&B/즉시소비 10종
#    (매출 큰 순으로 정렬, 매출 합 = 약 28만억원 ≈ 전체의 26%)
# ============================================
FT_CATEGORIES = [
    "한식음식점", "호프-간이주점", "커피-음료",
    "양식음식점", "분식전문점", "중식음식점", "일식음식점",
    "제과점", "패스트푸드점", "치킨전문점",
]

# ★ 실제 푸드트럭으로 운영 가능한 업종만 (추천 시 이 집합으로 한정).
#    좌식 레스토랑(한식/양식/중식/일식)·호프-간이주점(주류·좌석)은 제외.
FOODTRUCK_CATEGORIES = [
    "분식전문점", "커피-음료", "패스트푸드점", "제과점", "치킨전문점",
]

# ============================================
# 4) backtest_data/ 하위 폴더 매핑
#    (key → 폴더명 + 파일 패턴 + 다중 파일 여부)
# ============================================
DATA_SOURCES = {
    "sales_dong":       {"dir": "5. 추정매출",       "pattern": "행정동", "multi": True},
    "storecount":       {"dir": "7. 점포 정보",      "pattern": "행정동", "multi": True},
    "consumption":      {"dir": "8. 소득 · 소비 수준", "pattern": "",      "multi": False},
    "trdar_change":     {"dir": "6. 상권변화지표",   "pattern": "",       "multi": False},
    "living_pop":       {"dir": "1. 서울시 생활인구(내국인)_행정동 단위", "pattern": "LOCAL_PEOPLE", "multi": True},
    "subway":           {"dir": "3. 지하철 호선별 승하차", "pattern": "",  "multi": False},
    "bus":              {"dir": "4. 버스 노선별 승하차",   "pattern": "",  "multi": True},
    "weather":          {"dir": "11. 기상",          "pattern": "",       "multi": False},
    "holidays":         {"dir": "12. 공휴일",        "pattern": "",       "multi": False},
    "air":              {"dir": "13. 대기질",        "pattern": "",       "multi": False},
    "facility":         {"dir": "10. 상가업소_집객시설", "pattern": "", "multi": False},
    "census":           {"dir": "2. 총 조사 인구",    "pattern": "",       "multi": True},
    "resident":         {"dir": "2.1 주민등록인구",   "pattern": "",       "multi": True},
}

# ============================================
# 5) 표준 키 + 라벨
# ============================================
KEY_DONG = "adstr_cd"   # 표준화: 행정동 코드 (str, 8자리 그대로)
KEY_TIME = "yyqu"       # 기준분기 (int, 20231 등)
LABEL = "y_log_sales"   # 타깃 (log1p 변환된 푸드트럭 적합 업종 매출 합)

# 매출 데이터 컬럼 표준화 매핑
SALES_COL_MAP = {
    "기준_년분기_코드": KEY_TIME,
    "행정동_코드": KEY_DONG,
    "행정동_코드_명": "dong_nm",
    "서비스_업종_코드": "category_cd",
    "서비스_업종_코드_명": "category",
    "당월_매출_금액": "sales_amt",
    "당월_매출_건수": "sales_cnt",
}

# ============================================
# 6) 랜덤 시드
# ============================================
RANDOM_STATE = 42


# ============================================
# 7) ★ 일자(DAILY) 단위 모델링 설정  (2026-06 추가)
#    분기 추정매출을 일자로 분해한 파일로 ML 을 돌리기 위한 설정.
#    - v1: 분기매출을 단순 1/n (균등) 분배
#    - v2: 분기매출을 요일 트렌드(weekday) 기준 분배
# ============================================
SALES_DAILY_FILES = {
    "v1": "sales_v1_full_one_over_n.csv",
    "v2": "sales_v2_full_weekday_trend.csv",
}
SALES_DAILY_VERSION = "v2"          # 기본 타깃 (v1/v2 비교 시 노트북에서 전환)
SALES_DAILY_DIR = "5. 추정매출"

KEY_DATE = "date"                   # 표준 일자키 (pd.Timestamp)
KEY_YMD  = "ymd"                    # 정수 yyyymmdd (참고/머지용)

# 일자 파일 컬럼 표준화
SALES_DAILY_COL_MAP = {
    "일자": KEY_YMD,
    "행정동_코드": KEY_DONG,
    "행정동_코드_명": "dong_nm",
    "당월_매출_금액": "sales_ft",     # ← 그 날의 일매출 (푸드트럭 적합 업종 합)
    "당월_매출_건수": "sales_cnt",
}

# ★ 데이터 누수(leakage) 컬럼 — 타깃을 산식으로 쪼갠 값이므로 피처에서 제외
#    (요일별/시간대별/성별/연령별 매출 금액·건수 전부)
LEAKAGE_PREFIXES = (
    "주중_", "주말_", "월요일_", "화요일_", "수요일_", "목요일_",
    "금요일_", "토요일_", "일요일_",
    "시간대_", "남성_", "여성_", "연령대_",
)

# 일자 기반 holdout split — 시간 순서 (train: ~2024 / val: 2025 상반기 / test: 2025 하반기)
DAILY_TRAIN_END = "2024-12-31"
DAILY_VAL_END   = "2025-06-30"
# test = 2025-07-01 ~ 끝
