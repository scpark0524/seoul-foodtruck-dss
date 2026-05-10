# GitHub Actions로 24/7 자동 수집 — 셋업 가이드

> 박승찬 Mac을 켜놓지 않아도 매시간 자동으로 데이터가 모이도록 GitHub Actions를 사용합니다. 셋업 30분, 이후 무료 무한 운영.

---

## 0. 핵심 그림

```
[GitHub Actions 서버 (Ubuntu)]
   ↓ 매시간 정각 5분
[collect_realtime.py 실행]
   ↓ 4개 키 분산 호출
[서울 OpenAPI 121장소]
   ↓ 응답 13개 카테고리
[CSV 16개 파일에 누적]
   ↓
[git add → commit → push]
   ↓
[GitHub repo의 raw/realtime/ 자동 업데이트]
   ↓
[팀원 4명이 git pull로 데이터 동기화]
```

---

## 1. GitHub Repo 만들기 (5분)

### 옵션 A: 비공개 Private repo (권장)

1. https://github.com → New repository
2. 이름: `food-truck-dss-data` (자유)
3. **Private** 선택 (필수: 데이터/로그 외부 노출 방지)
4. README, .gitignore: 일단 빈 repo로 생성

> Private repo의 GitHub Actions 무료 한도: **월 2,000분**
> 우리 작업: 1회당 ~3분 × 24회/일 × 30일 = 2,160분 → 살짝 초과
> 해결책: 매시간 → 매 1.5시간 (16회/일 = 1,440분/월) 또는 Public repo로 전환

### 옵션 B: 공개 Public repo (호출 한도 무제한)

- Actions 무료 무제한
- 단, **API 키는 절대 코드에 박지 말 것** (Secrets 사용 시 안전)
- 데이터(.csv)는 푸드트럭 매출 분석용 공공데이터라 공개해도 무관

> **결론**: Private + 매 1.5시간 호출 또는 Public + 매시간 — 팀에서 결정

---

## 2. 로컬 폴더를 GitHub에 올리기 (10분)

```bash
cd ~/Desktop/3_KAIST/2026-Spring/2_인공지능/2_TeamProject

# git 초기화
cd food_truck_dss
git init
git branch -M main
git add .
git commit -m "Initial: collector + guides"

# 원격 repo 연결 (위에서 만든 repo URL)
git remote add origin https://github.com/<본인계정>/food-truck-dss-data.git
git push -u origin main
```

⚠️ **`.gitignore`에 `scripts/.env`가 들어있는지 반드시 확인** (API 키 노출 방지)

이미 .gitignore에 다음이 있어야 함:
```
scripts/.env
__pycache__/
*.pyc
.DS_Store
```

---

## 3. GitHub Secrets에 4개 키 등록 (10분)

API 키를 코드에 박지 않고 GitHub의 안전한 Secrets에 보관합니다.

1. GitHub repo 페이지 → **Settings** 탭 → 좌측 **Secrets and variables** → **Actions**
2. **New repository secret** 클릭
3. 다음 4~8개를 차례로 추가:

| Name | Value | 비고 |
|---|---|---|
| `SEOUL_API_KEY_1` | 박승찬 키 | 필수 |
| `SEOUL_API_KEY_2` | 허진우 키 | 필수 |
| `SEOUL_API_KEY_3` | 홍하경 키 | 필수 |
| `SEOUL_API_KEY_4` | 이연우 키 | 필수 |
| `SEOUL_API_QUOTA_1` | `1000` 또는 상향된 한도 | 선택 (미설정 시 1000) |
| `SEOUL_API_QUOTA_2` | `1000` 또는 상향된 한도 | 선택 |
| `SEOUL_API_QUOTA_3` | `1000` 또는 상향된 한도 | 선택 |
| `SEOUL_API_QUOTA_4` | `1000` 또는 상향된 한도 | 선택 |

> Secrets는 한 번 등록하면 다시 볼 수 없습니다 (저장만 가능). 분실 시 마이페이지에서 키 재발급 후 다시 등록.

---

## 4. Actions 워크플로우 등록 (5분)

워크플로우 파일이 이미 만들어져 있습니다. 다음 위치에 복사:

```bash
mkdir -p ~/Desktop/3_KAIST/2026-Spring/2_인공지능/2_TeamProject/.github/workflows
cp ~/Desktop/3_KAIST/2026-Spring/2_인공지능/2_TeamProject/food_truck_dss/scripts/collector.yml \
   ~/Desktop/3_KAIST/2026-Spring/2_인공지능/2_TeamProject/.github/workflows/collector.yml
```

> ⚠️ `.github/workflows/`는 **repo의 루트**에 있어야 합니다 (food_truck_dss 안 X).
> 따라서 git repo의 루트가 `food_truck_dss` 폴더라면, `.github/workflows/`도 같은 위치에 두면 됩니다.

워크플로우 내용 핵심:
- 매시간 5분(UTC 기준)에 자동 실행
- 4개 키를 환경변수로 주입
- `python collect_realtime.py` 실행
- 새 CSV가 생기면 자동 commit + push

```bash
# 푸시
cd food_truck_dss
git add .github/workflows/collector.yml
git commit -m "Add: GitHub Actions workflow"
git push
```

---

## 5. 첫 실행 (수동 트리거로 검증)

1. GitHub repo → **Actions** 탭
2. 좌측 **Realtime Collector** 워크플로우 선택
3. 우측 **Run workflow** 버튼 → mode `check-keys` → **Run**
4. 1분 후 실행 결과 확인 — 4개 키 모두 `✅ 정상 작동` 떠야 함

정상이면 `mode: test` (광화문 1곳) → `mode: full` (121장소) 순서로 검증.

---

## 6. 자동 실행 시작

위 단계까지 완료되면 **이미 자동 실행 중**입니다 (cron 등록됨). 확인:

1. Actions 탭에서 매시간 자동 실행 기록 확인
2. repo의 `food_truck_dss/raw/realtime/` 폴더가 매시간 업데이트되는지 확인
3. 팀원들은 `git pull`로 최신 데이터 동기화

---

## 7. 데이터 동기화 (팀원용)

팀원이 분석을 위해 최신 데이터를 가져오는 방법:

```bash
cd food_truck_dss
git pull
```

수집된 데이터가 자동으로 본인 PC로 복사됩니다.

---

## 8. 자주 발생하는 문제

### Q1. 매시간 5분이 정확히 5분에 안 돔
- GitHub Actions cron은 ±5~10분 지연 가능 (서버 부하)
- 푸드트럭 분석에 영향 없음 (분 단위 정확도 불필요)

### Q2. 같은 시간대에 두 번 실행됨
- 워크플로우의 `concurrency: cancel-in-progress: false` 설정으로 방지됨
- 그래도 발생하면 dedup 로직(`drop_duplicates`)이 자동 처리

### Q3. Actions 한도 (월 2,000분) 초과 알림
- Public repo로 전환 (무제한 무료) — `Settings → General → Visibility → Make public`
- 또는 cron을 `5 */2 * * *`로 변경 (매 2시간)

### Q4. git push 실패 (충돌)
- 워크플로우에 `git pull --rebase` 자동 처리됨
- 그래도 충돌 시 Actions 로그 확인 후 수동 해결

### Q5. API 키가 차단됨 (`INFO-200`)
- 마이페이지에서 한도 상향 신청
- 또는 5번째 키 발급 후 `SEOUL_API_KEY_5` Secret 추가 (코드는 자동으로 인식)

---

## 9. 모니터링 (체크리스트)

매주 한 번씩 확인:
- [ ] Actions 탭에서 최근 7일간 실행 성공률 95% 이상
- [ ] `raw/realtime/area_meta/YYYY-MM-DD.csv` 파일 크기 증가 추세
- [ ] 키별 차단 여부 (Actions 로그에서 `🔒` 검색)
- [ ] git repo 용량 (Settings → Storage)

---

## 10. 로컬 launchd 중지 (Actions로 완전 이전 후)

GitHub Actions가 1주일 안정 가동되면, 로컬 자동화는 중지:

```bash
launchctl unload ~/Library/LaunchAgents/com.team7.realtime.collector.plist
rm ~/Library/LaunchAgents/com.team7.realtime.collector.plist
```

> 단, 코드 수정·테스트 시에는 로컬에서 `python collect_realtime.py --test` 등으로 수동 실행 가능.

---

## 11. 비용 정산

| 항목 | 비용 |
|---|---|
| GitHub Actions (Public repo) | **무제한 무료** |
| GitHub Actions (Private, 월 2,000분) | 무료 (초과분은 분당 $0.008) |
| GitHub Repo 저장 (1GB까지) | 무료 |
| 서울 OpenAPI 호출 | 무료 |
| **합계** | **0원** |

---

*Last updated: 2026-05-10 / Team 7*
