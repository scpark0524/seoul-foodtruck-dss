#!/bin/bash
# 매시간 실행 진입점 — cron/launchd에서 호출
# 이 파일에 실행권한 필요: chmod +x run_hourly.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 가상환경 사용 시 활성화 (옵션)
# source ../venv/bin/activate

/usr/bin/env python3 collect_realtime.py 2>&1
