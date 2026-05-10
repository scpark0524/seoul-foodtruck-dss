# 7조 푸드트럭 DSS

## 폴더 구조
- `docs/` — 데이터 수집 가이드, 컬럼 정의서 등
- `scripts/` — 데이터 수집/전처리/모델링 스크립트
- `raw/` — 원본 응답 (gitignore)
- `processed/` — 표준화된 패널 데이터 (gitignore)

## 시작하기
1. `scripts/.env.template` → `.env`로 복사 후 API 키 4종 입력
2. `pip install requests pandas python-dotenv pyarrow`
3. `python scripts/collect_starter.py` 실행 — 4개 데이터 샘플 수집

자세한 내용은 `docs/데이터수집_가이드.md` 참조.
