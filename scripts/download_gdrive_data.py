"""
구글드라이브 폴더에서 푸드트럭 DSS용 분기 데이터를 다운로드합니다.

사용법:
    python scripts/download_gdrive_data.py

전제:
    - 폴더 공유 설정이 '링크 있는 모든 사용자' 이상이어야 함
    - gdown 패키지 설치 필요 (requirements_ml.txt에 포함)

가이드:
    구글드라이브 폴더가 비공개이거나 권한 문제로 실패하면,
    STEP_BY_STEP_가이드.md 의 '방법 A. 수동 다운로드' 를 사용하세요.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("[ERROR] gdown 패키지가 없습니다. 먼저 `pip install gdown` 또는")
    print("        `pip install -r requirements_ml.txt` 를 실행하세요.")
    sys.exit(1)


# ============================================
# 1) 구글드라이브 폴더 ID — 7조 공유 폴더
# ============================================
FOLDER_ID = "1gW4xoJRYCvLqcEiyplk5smY_bt2HBS0Q"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"

# ============================================
# 2) 로컬 저장 경로
# ============================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = PROJECT_ROOT / "data" / "quarterly"


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 다운로드 시작")
    print(f"       Source: {FOLDER_URL}")
    print(f"       Dest:   {DEST_DIR}")
    print()

    try:
        result = gdown.download_folder(
            url=FOLDER_URL,
            output=str(DEST_DIR),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )
        if result is None:
            raise RuntimeError("gdown 반환값이 None — 권한 문제 가능성")
        print()
        print(f"[OK] {len(result)}개 파일 다운로드 완료.")
    except Exception as e:
        print(f"[ERROR] 다운로드 실패: {e}")
        print()
        print("아래 방법 중 하나로 해결하세요:")
        print("  1) 구글드라이브 폴더 공유 설정 → '링크 있는 모든 사용자'로 변경 후 재시도")
        print(f"  2) 수동 다운로드: {FOLDER_URL}")
        print(f"     → 다운받은 파일을 {DEST_DIR} 에 넣기")
        sys.exit(1)

    files = sorted(p.name for p in DEST_DIR.iterdir() if p.is_file())
    print()
    print(f"[INFO] {DEST_DIR} 내 파일 목록 ({len(files)}개):")
    for f in files:
        print(f"   - {f}")

    print()
    print("[NEXT] 다음 단계: VSCode 에서 notebooks/01_eda_baseline.py 실행")


if __name__ == "__main__":
    main()
