"""
7조 푸드트럭 DSS — 데이터 수집 스타터 스크립트
================================================
이 파일 하나로 4개 대표 데이터(생활인구, 추정매출, 기상, 공휴일)를
호출해보고 raw/ 폴더에 parquet으로 저장합니다.

사용법:
    1. 같은 폴더에 .env 파일을 만들고 아래 키를 채웁니다.
        SEOUL_API_KEY=...
        DATA_GO_KR_KEY_DEC=...
    2. pip install requests pandas python-dotenv pyarrow
    3. python collect_starter.py

본 스크립트는 학습 / 검증용 최소 호출만 수행합니다.
실서비스용 전체 수집은 phase별 스크립트(03_collect_sales.py 등)로 분리하세요.
"""

import os
import time
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ---------- 환경 설정 ----------
load_dotenv()
SEOUL_KEY = os.getenv("SEOUL_API_KEY", "REPLACE_ME")
DATA_KEY = os.getenv("DATA_GO_KR_KEY_DEC", "REPLACE_ME")

RAW_DIR = Path(__file__).parent.parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Raw data will be saved to: {RAW_DIR.resolve()}")


# ===============================================================
# 공통 헬퍼
# ===============================================================
def fetch_seoul_open(service: str, start: int = 1, end: int = 1000, *args) -> pd.DataFrame:
    """서울 열린데이터광장 표준 호출.
    URL pattern: /{KEY}/json/{SERVICE}/{START}/{END}/{ARG1}/{ARG2}...
    """
    base = "http://openapi.seoul.go.kr:8088"
    url = f"{base}/{SEOUL_KEY}/json/{service}/{start}/{end}"
    if args:
        url += "/" + "/".join(str(a) for a in args)

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()

    # 응답 구조: { service: { "list_total_count": ..., "row": [...] } }
    if service not in js:
        # 에러 응답 구조 확인
        err = js.get("RESULT") or js
        print(f"[WARN] Unexpected response: {json.dumps(err, ensure_ascii=False)[:200]}")
        return pd.DataFrame()

    rows = js[service].get("row", [])
    return pd.DataFrame(rows)


def paged_fetch(service: str, page_size: int = 1000, *args, max_pages: int = 100) -> pd.DataFrame:
    """페이징을 자동으로 처리하는 호출."""
    all_dfs = []
    for p in range(max_pages):
        start = p * page_size + 1
        end = (p + 1) * page_size
        df = fetch_seoul_open(service, start, end, *args)
        if df.empty:
            break
        all_dfs.append(df)
        print(f"   page {p+1}: {len(df)} rows (total {sum(len(d) for d in all_dfs)})")
        if len(df) < page_size:
            break
        time.sleep(0.3)  # rate limit 보호
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ===============================================================
# 1. 서울 생활인구 (행정동 단위, 일별)
#    서비스명 예: SPOP_LOCAL_RESD_DONG
#    인자: stdrDe (YYYYMMDD)
# ===============================================================
def collect_living_pop_sample(date_str: str = "20231201"):
    print(f"\n[1] 서울 생활인구 — {date_str}")
    df = paged_fetch("SPOP_LOCAL_RESD_DONG", 1000, date_str, max_pages=20)
    if df.empty:
        print("   ⚠️ 응답 없음 — 서비스명/날짜/키 확인 필요")
        return
    out = RAW_DIR / f"01_living_pop_{date_str}.parquet"
    df.to_parquet(out, index=False)
    print(f"   ✅ saved {len(df):,} rows → {out.name}")
    print(f"   columns: {list(df.columns)[:8]}...")


# ===============================================================
# 5. 추정매출 (행정동 단위, 분기)
#    서비스명 예: VwsmAdstrdSelngW
#    인자: stdrYyquCd (YYYYQ, 예: 20234)
# ===============================================================
def collect_sales_sample(yyqu: int = 20234):
    print(f"\n[5] 추정매출(행정동) — {yyqu}")
    df = paged_fetch("VwsmAdstrdSelngW", 1000, yyqu, max_pages=10)
    if df.empty:
        print("   ⚠️ 응답 없음 — 서비스명을 확인하세요. (서울 열린데이터광장 검색)")
        return
    out = RAW_DIR / f"05_sales_dong_{yyqu}.parquet"
    df.to_parquet(out, index=False)
    print(f"   ✅ saved {len(df):,} rows → {out.name}")
    print(f"   columns: {list(df.columns)[:8]}...")
    # 행정동 유니크 카운트로 검증
    if "ADSTRD_CD" in df.columns:
        print(f"   행정동 유니크: {df['ADSTRD_CD'].nunique()}개")


# ===============================================================
# 12. 기상 (ASOS 일별 종관관측) — 공공데이터포털
#     서비스: AsosDalyInfoService / getWthrDataList
#     서울 종관관측소 stnIds = 108
# ===============================================================
def collect_weather_sample(start_dt: str = "20231201", end_dt: str = "20231231"):
    print(f"\n[12] 기상(ASOS, 서울) — {start_dt}~{end_dt}")
    url = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
    params = {
        "serviceKey": DATA_KEY,
        "pageNo": 1,
        "numOfRows": 999,
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": start_dt,
        "endDt": end_dt,
        "stnIds": "108",  # 서울
    }
    r = requests.get(url, params=params, timeout=30)
    try:
        js = r.json()
        items = js["response"]["body"]["items"]["item"]
    except (KeyError, ValueError) as e:
        print(f"   ⚠️ 파싱 실패: {e} / 응답: {r.text[:200]}")
        return
    df = pd.DataFrame(items)
    out = RAW_DIR / f"12_weather_seoul_{start_dt}_{end_dt}.parquet"
    df.to_parquet(out, index=False)
    print(f"   ✅ saved {len(df):,} rows → {out.name}")
    print(f"   columns: {list(df.columns)[:10]}...")


# ===============================================================
# 14. 공휴일 (천문연구원 특일 정보)
#     서비스: SpcdeInfoService / getRestDeInfo
# ===============================================================
def collect_holidays_sample(year: int = 2023):
    print(f"\n[14] 공휴일 — {year}")
    url = "https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"
    holidays = []
    for month in range(1, 13):
        params = {
            "serviceKey": DATA_KEY,
            "solYear": year,
            "solMonth": f"{month:02d}",
            "_type": "json",
            "numOfRows": 100,
        }
        r = requests.get(url, params=params, timeout=20)
        try:
            js = r.json()
            items = js["response"]["body"]["items"]
            if not items:
                continue
            item = items.get("item", [])
            if isinstance(item, dict):
                item = [item]
            holidays.extend(item)
        except (KeyError, ValueError) as e:
            print(f"   month {month}: parse fail {e}")
        time.sleep(0.2)

    df = pd.DataFrame(holidays)
    if df.empty:
        print("   ⚠️ 응답 없음")
        return
    out = RAW_DIR / f"14_holidays_{year}.parquet"
    df.to_parquet(out, index=False)
    print(f"   ✅ saved {len(df):,} rows → {out.name}")
    print(df[["dateName", "locdate", "isHoliday"]].head(10).to_string(index=False))


# ===============================================================
# 메인 — 4개 데이터 샘플 호출
# ===============================================================
def main():
    if SEOUL_KEY == "REPLACE_ME":
        print("[ERROR] .env 파일에 SEOUL_API_KEY를 설정하세요.")
        return
    if DATA_KEY == "REPLACE_ME":
        print("[ERROR] .env 파일에 DATA_GO_KR_KEY_DEC를 설정하세요.")
        return

    print("="*60)
    print(f"7조 푸드트럭 DSS — 데이터 수집 스타터")
    print(f"실행 시각: {datetime.now()}")
    print("="*60)

    # 4개 카테고리 샘플 수집
    try: collect_living_pop_sample("20231201")
    except Exception as e: print(f"   ❌ 실패: {e}")

    try: collect_sales_sample(20234)
    except Exception as e: print(f"   ❌ 실패: {e}")

    try: collect_weather_sample("20231201", "20231231")
    except Exception as e: print(f"   ❌ 실패: {e}")

    try: collect_holidays_sample(2023)
    except Exception as e: print(f"   ❌ 실패: {e}")

    print("\n" + "="*60)
    print(f"완료. 다음 단계:")
    print(f"  1. raw/ 폴더의 parquet 파일들을 열어 응답 구조를 확인")
    print(f"  2. 컬럼명을 표준 매핑(adstr_cd, yyqu, ...)으로 정리")
    print(f"  3. phase별 스크립트(01_~06_)로 분리 후 전체 기간 수집")
    print("="*60)


if __name__ == "__main__":
    main()
