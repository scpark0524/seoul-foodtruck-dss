"""
7조 푸드트럭 DSS — 서울시 실시간 도시데이터 수집기 (통합 v3)
==============================================================
서울 실시간 도시데이터 매뉴얼 v8.5 기준
- API: citydata (통합) — 13개 카테고리 모두 수집
- 인구·상권·도로·주차장·지하철·버스·사고·전기차충전소·따릉이·날씨·문화행사·재난문자·연합뉴스
- 121개 주요장소 (인구 121곳, 상권 82곳)

저장 구조 (카테고리별 분리):
    raw/realtime/{category}/{YYYY-MM-DD}.csv

설계 원칙:
1. Generic parser — 응답 키를 동적으로 추출 (매뉴얼 v 업그레이드 자동 대응)
2. 카테고리별 CSV 분리 — 분석 시 한 카테고리 한 파일로 로드 용이
3. nested list 자동 분리 (예: CMRCL_RSB → cmrcl_rsb.csv 별도)
4. 멀티키 풀 + Fallback (v2와 동일)
5. 모든 값을 string으로 보존 (CSV 호환성)

사용법:
    python collect_realtime.py            # 전체 수집
    python collect_realtime.py --test     # 광화문 1곳만
    python collect_realtime.py --place 강남역
    python collect_realtime.py --check-keys
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict
from urllib.parse import quote

import requests
import pandas as pd
from dotenv import load_dotenv

try:
    import xmltodict
except ImportError:
    print("[FATAL] xmltodict 미설치 — pip install xmltodict")
    sys.exit(1)


# ===============================================================
# 환경 설정
# ===============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(SCRIPT_DIR / ".env")

DATA_DIR = PROJECT_ROOT / "raw" / "realtime"
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

KST = timezone(timedelta(hours=9))

BASE_URL = "http://openapi.seoul.go.kr:8088"
SERVICE = "citydata"   # 통합 API (이전 citydata_cmrcl → citydata)
RETURN_TYPE = "xml"    # citydata는 XML만 (citydata_ppltn/cmrcl만 JSON 지원)

RATE_LIMIT_SLEEP = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF = [2, 5, 10]
REQUEST_TIMEOUT = 60   # 통합 응답이 크므로 timeout 증가


# 매뉴얼 표 3-2 기준 13개 카테고리 정의
# (응답 키 → 우리 카테고리 폴더명)
CATEGORY_SECTIONS: Dict[str, str] = {
    "LIVE_PPLTN_STTS":     "ppltn",      # 실시간 인구현황 (25개 항목)
    "LIVE_CMRCL_STTS":     "cmrcl",      # 실시간 상권 현황 (24개 항목, RSB nested)
    "ROAD_TRAFFIC_STTS":   "road",       # 도로소통현황 (17개 항목)
    "PRK_STTS":            "parking",    # 주차장 현황 (17개 항목, 주차장별)
    "SUB_STTS":            "subway",     # 지하철 현황 (43개 항목, 역별)
    "BUS_STN_STTS":        "bus",        # 버스 정류소 현황 (26개 항목, 정류장별)
    "ACDNT_CNTRL_STTS":    "accident",   # 사고통제현황 (9개 항목)
    "CHARGER_STTS":        "charger",    # 전기차충전소 현황 (20개 항목)
    "SBIKE_STTS":          "bike",       # 따릉이 현황 (8개 항목, 대여소별)
    "WEATHER_STTS":        "weather",    # 날씨 현황 (39개 항목, FCST24HOURS nested)
    "EVENT_STTS":          "events",     # 문화행사 현황 (10개 항목)
    "LIVE_DST_MESSAGE":    "disaster",   # 긴급재난문자 (5개 항목)
    "LIVE_YNA_NEWS":       "news",       # 연합뉴스 (6개 항목)
}


# ===============================================================
# 멀티키 관리 (v2와 동일)
# ===============================================================
@dataclass
class APIKey:
    name: str
    value: str
    daily_quota: int = 1000
    used_today: int = 0
    blocked: bool = False
    block_reason: str = ""

    def can_call(self) -> bool:
        return (not self.blocked) and (self.used_today < self.daily_quota)

    def remaining(self) -> int:
        return max(0, self.daily_quota - self.used_today)

    def __str__(self):
        status = "🟢" if self.can_call() else "🔴"
        return f"{status} {self.name} ({self.used_today}/{self.daily_quota}) " \
               f"{'BLOCKED:'+self.block_reason if self.blocked else ''}"


class KeyPool:
    def __init__(self):
        self.keys: List[APIKey] = []
        candidates = [
            ("MAIN", os.getenv("SEOUL_API_KEY", "").strip()),
            ("KEY_1", os.getenv("SEOUL_API_KEY_1", "").strip()),
            ("KEY_2", os.getenv("SEOUL_API_KEY_2", "").strip()),
            ("KEY_3", os.getenv("SEOUL_API_KEY_3", "").strip()),
            ("KEY_4", os.getenv("SEOUL_API_KEY_4", "").strip()),
        ]
        for name, value in candidates:
            if value:
                quota_env = f"SEOUL_API_QUOTA_{name.replace('KEY_','')}" \
                            if name != "MAIN" else "SEOUL_API_QUOTA"
                quota = int(os.getenv(quota_env, "1000"))
                self.keys.append(APIKey(name=name, value=value, daily_quota=quota))

        if not self.keys:
            raise RuntimeError(".env에 SEOUL_API_KEY 또는 SEOUL_API_KEY_1~4 중 하나라도 필요")

    def __len__(self):
        return len(self.keys)

    def total_quota(self) -> int:
        return sum(k.daily_quota for k in self.keys)

    def total_used(self) -> int:
        return sum(k.used_today for k in self.keys)

    def status_summary(self) -> str:
        active = sum(1 for k in self.keys if k.can_call())
        return (f"키 풀: {active}/{len(self.keys)} 활성 | "
                f"오늘 사용 {self.total_used()}/{self.total_quota()}건")

    def split_pois(self, poi_list: List[str]) -> dict:
        if not self.keys:
            return {}
        n_keys = len(self.keys)
        assignment = {k.name: [] for k in self.keys}
        for i, poi in enumerate(poi_list):
            assignment[self.keys[i % n_keys].name].append(poi)
        return assignment

    def get_fallback_key(self, exclude: APIKey) -> Optional[APIKey]:
        candidates = [k for k in self.keys if k.can_call() and k.name != exclude.name]
        return max(candidates, key=lambda k: k.remaining()) if candidates else None


# ===============================================================
# 로깅
# ===============================================================
def setup_logging():
    today = datetime.now(KST).strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"realtime_{today}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("realtime")


log = setup_logging()


# ===============================================================
# 121개 주요장소 (매뉴얼 v8.5 표 2-2 기준)
# ===============================================================
POI_LIST_121 = [
    # 고궁·문화유산 (5)
    "경복궁", "광화문·덕수궁", "보신각", "서울 암사동 유적", "창덕궁·종묘",
    # 관광특구 (7)
    "강남 MICE 관광특구", "동대문 관광특구", "명동 관광특구",
    "이태원 관광특구", "잠실 관광특구", "종로·청계 관광특구", "홍대 관광특구",
    # 공원 (33)
    "강서한강공원", "고척돔", "광나루한강공원", "광화문광장",
    "국립중앙박물관·용산가족공원", "난지한강공원", "남산공원", "노들섬",
    "뚝섬한강공원", "망원한강공원", "반포한강공원", "보라매공원",
    "북서울꿈의숲", "서대문독립공원", "서리풀공원·몽마르뜨공원", "서울대공원",
    "서울숲공원", "송현녹지광장", "아차산", "안양천", "양화한강공원",
    "어린이대공원", "여의도한강공원", "여의서로", "올림픽공원", "월드컵공원",
    "응봉산", "이촌한강공원", "잠실종합운동장", "잠실한강공원", "잠원한강공원",
    "청계산", "홍제폭포",
    # 발달상권 (28)
    "가락시장", "가로수길", "광장(전통)시장", "김포공항", "남대문시장",
    "노량진", "덕수궁길·정동길", "북창동 먹자골목", "북촌한옥마을", "서촌",
    "성수카페거리", "송리단길·호수단길", "신촌 스타광장", "압구정로데오거리",
    "여의도", "연남동", "영등포 타임스퀘어", "용리단길", "이태원 앤틱가구거리",
    "익선동", "인사동", "잠실롯데타워·석촌호수", "창동 신경제 중심지",
    "청담동 명품거리", "청량리 제기동 일대 전통시장", "해방촌·경리단길",
    "DDP(동대문디자인플라자)", "DMC(디지털미디어시티)",
    # 인구밀집지역 (48)
    "가산디지털단지역", "강남역", "건대입구역", "고덕역", "고속터미널역",
    "교대역", "구로디지털단지역", "구로역", "군자역", "대림역", "동대문역",
    "뚝섬역", "미아사거리역", "발산역", "사당역", "삼각지역", "서울대입구역",
    "서울식물원·마곡나루역", "서울역", "선릉역", "성신여대입구역", "수유역",
    "숭례문", "시의회 앞", "신논현역·논현역", "신도림역", "신림역",
    "신정네거리역", "신촌·이대역", "쌍문역", "양재역", "역삼역", "연신내역",
    "오목교역·목동운동장", "왕십리역", "용산역", "이태원역", "잠실새내역",
    "잠실역", "장지역", "장한평역", "천호역", "총신대입구(이수)역",
    "충정로역", "합정역", "혜화역", "홍대입구역(2호선)", "회기역",
]


def load_poi_list():
    """우선순위: scripts/poi_list.csv → raw/*.xlsx → POI_LIST_121"""
    csv_path = SCRIPT_DIR / "poi_list.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        col = "place_name" if "place_name" in df.columns else df.columns[0]
        pois = df[col].dropna().astype(str).tolist()
        log.info(f"POI 로드: {len(pois)}개 (poi_list.csv)")
        return pois

    xlsx_candidates = list((PROJECT_ROOT / "raw").glob("*장소*목록*.xlsx")) + \
                      list((PROJECT_ROOT / "raw").glob("*82장소*.xlsx")) + \
                      list((PROJECT_ROOT / "raw").glob("*121*.xlsx"))
    if xlsx_candidates:
        df = pd.read_excel(xlsx_candidates[0])
        for col in ["AREA_NM", "장소명", "place_name", "POI_NM", "AREA_NAME"]:
            if col in df.columns:
                pois = df[col].dropna().astype(str).tolist()
                log.info(f"POI 로드: {len(pois)}개 ({xlsx_candidates[0].name})")
                return pois

    log.info(f"POI 로드: 매뉴얼 v8.5 기준 fallback {len(POI_LIST_121)}개 (121개 전체)")
    return POI_LIST_121


# ===============================================================
# API 호출
# ===============================================================
def fetch_one_poi(place_name: str, key_pool: KeyPool, primary_key: APIKey) -> Optional[dict]:
    encoded = quote(place_name, safe="")
    keys_to_try = [primary_key]
    while True:
        nxt = key_pool.get_fallback_key(exclude=keys_to_try[-1]) \
              if not keys_to_try[-1].can_call() else None
        if nxt and nxt not in keys_to_try:
            keys_to_try.append(nxt)
        else:
            break
    keys_to_try = [k for k in keys_to_try if k.can_call()]
    if not keys_to_try:
        log.error(f"  ❌ 사용 가능한 키 없음 ({place_name})")
        return None

    for key in keys_to_try:
        url = f"{BASE_URL}/{key.value}/{RETURN_TYPE}/{SERVICE}/1/5/{encoded}"
        for attempt in range(MAX_RETRIES):
            try:
                key.used_today += 1
                r = requests.get(url, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()

                parsed = xmltodict.parse(r.text)
                root_key = list(parsed.keys())[0] if parsed else None
                if not root_key:
                    return None

                body = parsed[root_key]
                # 에러 응답 처리
                result = body.get("RESULT", {}) if isinstance(body, dict) else {}
                if isinstance(result, dict):
                    code = result.get("RESULT.CODE", result.get("CODE"))
                    msg = result.get("RESULT.MESSAGE", result.get("MESSAGE", ""))
                    if code and code != "INFO-000":
                        if code in ("INFO-200", "INFO-300"):
                            log.warning(f"  🔒 키 한도 초과 [{key.name}]: {code} {msg}")
                            key.blocked = True
                            key.block_reason = f"{code}: {msg}"
                            break
                        if code in ("INFO-100",):
                            log.error(f"  🔒 키 인증 실패 [{key.name}]: {msg}")
                            key.blocked = True
                            key.block_reason = f"{code}: {msg}"
                            break
                        log.warning(f"  ⚠️ {place_name}: {code} {msg}")
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(RETRY_BACKOFF[attempt])
                        continue
                return body

            except requests.exceptions.RequestException as e:
                log.warning(f"  ⚠️ 호출 실패 ({place_name}, key={key.name}, "
                            f"시도 {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
            except Exception as e:
                log.error(f"  ❌ 파싱 실패 ({place_name}, key={key.name}): {e}")
                break
    return None


# ===============================================================
# 응답 → 카테고리별 DataFrame 분해
# ===============================================================
def _stringify(v):
    """모든 값을 CSV 친화적 string으로 변환."""
    if v is None:
        return ""
    if isinstance(v, (str, int, float)):
        return v
    return json.dumps(v, ensure_ascii=False)


def _split_scalars_and_lists(d: dict) -> tuple:
    """dict에서 (scalar_dict, nested_lists) 분리.
    - scalar: str/int/float/None → 그대로
    - nested dict: 한 단계 평탄화 (key__subkey)
    - list of dict: nested_lists에 보관 (별도 카테고리로 처리)
    """
    if not isinstance(d, dict):
        return {}, {}
    scalars, nested_lists = {}, {}
    for k, v in d.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            nested_lists[k] = v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if not isinstance(vv, (dict, list)):
                    scalars[f"{k}__{kk}"] = _stringify(vv)
        elif isinstance(v, list):
            # 단순 리스트는 JSON 직렬화
            scalars[k] = json.dumps(v, ensure_ascii=False)
        else:
            scalars[k] = _stringify(v)
    return scalars, nested_lists


def parse_section(section, base_meta: dict, category_name: str) -> Dict[str, pd.DataFrame]:
    """한 카테고리 응답을 {카테고리명: DataFrame} 사전으로.
    nested list가 있으면 {category_name_subkey: DataFrame}으로 분리.
    """
    result: Dict[str, pd.DataFrame] = {}
    if section is None or section == "":
        return result

    # 단일 dict
    if isinstance(section, dict):
        scalars, nested = _split_scalars_and_lists(section)
        if scalars:
            result[category_name] = pd.DataFrame([{**base_meta, **scalars}])
        for nkey, nlist in nested.items():
            sub_name = f"{category_name}_{nkey.lower()}"
            sub_rows = []
            for item in nlist:
                if isinstance(item, dict):
                    sc, _ = _split_scalars_and_lists(item)
                    sub_rows.append({**base_meta, **sc})
            if sub_rows:
                result[sub_name] = pd.DataFrame(sub_rows)

    # 리스트 (각 항목이 dict)
    elif isinstance(section, list):
        rows = []
        for item in section:
            if isinstance(item, dict):
                sc, _ = _split_scalars_and_lists(item)
                rows.append({**base_meta, **sc})
        if rows:
            result[category_name] = pd.DataFrame(rows)

    return result


def normalize_response(body: dict, place_name: str, fetched_at: datetime,
                       key_used: str) -> Dict[str, pd.DataFrame]:
    """통합 응답을 {카테고리명: DataFrame} 사전으로 분해."""
    result: Dict[str, pd.DataFrame] = {}
    if not body or not isinstance(body, dict):
        return result

    # 컨테이너 찾기 (CITYDATA 키 또는 body 자체)
    container = body.get("CITYDATA", body)
    if not isinstance(container, dict):
        container = body

    base_meta = {
        "fetched_at": fetched_at.isoformat(),
        "place_name": place_name,
        "AREA_NM": container.get("AREA_NM", place_name),
        "AREA_CD": container.get("AREA_CD", ""),
        "_key_used": key_used,
    }

    # area_meta 카테고리 (호출당 1행)
    result["area_meta"] = pd.DataFrame([base_meta])

    # 13개 카테고리 순회
    for section_key, cat_name in CATEGORY_SECTIONS.items():
        section = container.get(section_key)
        cat_results = parse_section(section, base_meta, cat_name)
        for k, v in cat_results.items():
            if k in result:
                result[k] = pd.concat([result[k], v], ignore_index=True, sort=False)
            else:
                result[k] = v

    return result


# ===============================================================
# 저장 (카테고리별 일별 CSV)
# ===============================================================
def save_categorized(category_dfs: Dict[str, pd.DataFrame], fetched_at: datetime):
    date_str = fetched_at.strftime("%Y-%m-%d")
    saved_summary = []
    for cat_name, df in category_dfs.items():
        if df.empty:
            continue
        cat_dir = DATA_DIR / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)
        out_path = cat_dir / f"{date_str}.csv"

        if out_path.exists():
            try:
                existing = pd.read_csv(out_path, encoding="utf-8-sig", dtype=str)
                df_str = df.astype(str)
                combined = pd.concat([existing, df_str], ignore_index=True, sort=False)
                # 모든 컬럼 일치 시 dedup (같은 시각 같은 장소 같은 행)
                combined = combined.drop_duplicates(keep="last")
                combined.to_csv(out_path, index=False, encoding="utf-8-sig")
                saved_summary.append((cat_name, len(combined), len(df), len(combined.columns)))
            except Exception as e:
                log.error(f"기존 CSV 읽기 실패 ({cat_name}): {e} — 새 파일로 덮어씀")
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
                saved_summary.append((cat_name, len(df), len(df), len(df.columns)))
        else:
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            saved_summary.append((cat_name, len(df), len(df), len(df.columns)))

    return saved_summary


# ===============================================================
# 메인 수집
# ===============================================================
def collect_all(poi_list: List[str], key_pool: KeyPool, dry_run: bool = False):
    fetched_at = datetime.now(KST)
    log.info(f"=== 수집 시작 {fetched_at.isoformat()} ===")
    log.info(f"대상 장소: {len(poi_list)}개 | API: {SERVICE} (통합)")
    log.info(f"{key_pool.status_summary()}")

    assignment = key_pool.split_pois(poi_list)
    for key in key_pool.keys:
        log.info(f"  [{key.name}] 담당 {len(assignment[key.name])}장소")

    poi_to_key = {poi: key for key in key_pool.keys for poi in assignment[key.name]}

    all_collected: Dict[str, list] = defaultdict(list)
    success_count = 0
    fail_count = 0

    for i, poi in enumerate(poi_list, 1):
        primary = poi_to_key.get(poi, key_pool.keys[0])
        if dry_run:
            log.info(f"[{i}/{len(poi_list)}] {poi} (primary={primary.name}) (dry-run)")
            continue

        log.info(f"[{i}/{len(poi_list)}] {poi} (primary={primary.name})")
        body = fetch_one_poi(poi, key_pool, primary)
        if body is None:
            fail_count += 1
            time.sleep(RATE_LIMIT_SLEEP)
            continue

        cat_dfs = normalize_response(body, poi, fetched_at, key_used=primary.name)
        if cat_dfs:
            for cat_name, df in cat_dfs.items():
                if not df.empty:
                    all_collected[cat_name].append(df)
            success_count += 1
        else:
            fail_count += 1

        time.sleep(RATE_LIMIT_SLEEP)

    # 카테고리별 합치고 저장
    if all_collected:
        merged: Dict[str, pd.DataFrame] = {
            cat: pd.concat(dfs, ignore_index=True, sort=False)
            for cat, dfs in all_collected.items()
        }
        saved = save_categorized(merged, fetched_at)

        log.info(f"=== 수집 완료 | 성공 {success_count}, 실패 {fail_count} ===")
        log.info(f"카테고리별 저장 결과:")
        for cat_name, total_rows, new_rows, n_cols in sorted(saved):
            log.info(f"   {cat_name:25s} | 신규 {new_rows:5d}행 | 누적 {total_rows:6d}행 | {n_cols}컬럼")
        log.info(f"키별 사용량:")
        for key in key_pool.keys:
            log.info(f"   {key}")
    else:
        log.error(f"=== 수집 완료 | 모든 호출 실패 ===")


# ===============================================================
# 키 점검
# ===============================================================
def check_keys(key_pool: KeyPool):
    log.info(f"=== 키 점검 ({len(key_pool)}개) ===")
    for key in key_pool.keys:
        log.info(f"\n[{key.name}] 광화문·덕수궁 호출 시도")
        body = fetch_one_poi("광화문·덕수궁", key_pool, key)
        if body and not key.blocked:
            cat_dfs = normalize_response(body, "광화문·덕수궁", datetime.now(KST), key.name)
            categories = sorted([k for k, v in cat_dfs.items() if not v.empty])
            log.info(f"  ✅ {key.name} 정상 | 받은 카테고리 {len(categories)}개: {categories}")
        elif key.blocked:
            log.error(f"  ❌ {key.name} 차단: {key.block_reason}")
        else:
            log.error(f"  ❌ {key.name} 응답 없음")
    log.info(f"\n{key_pool.status_summary()}")


# ===============================================================
# CLI
# ===============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="광화문·덕수궁 1곳만")
    parser.add_argument("--place", type=str, default=None,
                        help="특정 장소만 (예: --place 강남역)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-keys", action="store_true")
    args = parser.parse_args()

    try:
        key_pool = KeyPool()
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)

    log.info(f"등록된 키: {len(key_pool)}개, 합산 한도 {key_pool.total_quota():,}건/일")

    if args.check_keys:
        check_keys(key_pool)
        return

    if args.test:
        poi_list = ["광화문·덕수궁"]
    elif args.place:
        poi_list = [args.place]
    else:
        poi_list = load_poi_list()

    collect_all(poi_list, key_pool, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
