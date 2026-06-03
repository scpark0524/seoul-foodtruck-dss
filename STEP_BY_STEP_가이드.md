# 푸드트럭 DSS — Step-by-Step ML 실행 가이드 (★ 일자 DAILY 버전)

> **변경(2026-06)**: 분기 단위 → **일자(daily) 단위** 파이프라인으로 전환.
> 분기 추정매출을 일자로 분해한 파일(v1/v2)을 타깃으로, 행정동×일자(2021-01-01~2025-12-31)
> 약 77만 행을 학습. 분기 버전 노트북은 `notebooks/_quarterly_backup/`, 가이드는
> `STEP_BY_STEP_가이드_분기버전.md.bak` 에 보관.

---

## 전체 흐름

```
STEP 0  환경 셋업 (venv + 패키지)            ── 한 번만
STEP 1  EDA + Naive baseline (일 단위)       ── notebooks/01_eda_baseline.py
STEP 2  Feature Engineering (~103 피처)       ── notebooks/02_feature_engineering.py
STEP 3  XGBoost 학습 (일자 split)             ── notebooks/03_xgboost_tuned.py
STEP 4  Walk-Forward + v1/v2 비교             ── notebooks/04_evaluation_backtest.py
STEP 5  SHAP 해석                            ── notebooks/05_shap_interpretation.py
```

산출물: `processed/`(parquet), `models/`(xgb_best.*), `results/`(png·json·csv).

---

## 핵심 설계

- **타깃**: `y_log_sales = log1p(당월_매출_금액)` = 그 날의 푸드트럭 적합 업종 일매출.
- **일자 매출 버전** (`src/config.py` 의 `SALES_DAILY_VERSION`):
  - `v2` (기본, 요일 트렌드) — 실제 요일 패턴 반영, 더 현실적.
  - `v1` (균등 1/n) — 분기매출을 일수로 균등 분배. STEP 4 에서 자동 비교.
- **★ 데이터 누수 차단**: 같은 행의 요일별·시간대별·성별·연령별 매출 컬럼은
  타깃을 산식으로 쪼갠 값이라 **피처에서 제외**(`LEAKAGE_PREFIXES`). 외부 피처
  (기상·생활인구·점포·시설·과거매출 lag)로만 예측.
- **피처군**: A 캘린더 / B 매출 lag(1·7·28·364, roll7·28) / C 기상 / D 공휴일 /
  E 대기질 / F 생활인구(내국인·외국인·점심·저녁·20-30대) / G 소비·상권변화·점포(분기 broadcast) /
  H 지하철·버스(월 broadcast) / I 시설·집객(정적).
- **Split**: 일자 기준 train(~2024-12-31) / val(~2025-06-30) / test(2025-07-01~).
  미래 정보 누수 방지를 위해 시간 순서로만 분할.

---

## STEP 0 — 환경 셋업

Python **3.11~3.12** 권장. VSCode 확장: Python, Jupyter.

```bash
cd food_truck_dss
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements_ml.txt
# (macOS) XGBoost OpenMP 런타임
brew install libomp
python -c "import xgboost, pandas, pyarrow, matplotlib; print('OK')"
```

> 메모리: 일자 데이터는 약 77만 행 × 103 피처. 8GB RAM 이면 충분. 노트북은
> 단계별 parquet 핸드오프 + float32 다운캐스팅으로 메모리를 절약하도록 작성됨.

---

## STEP 1 — 01_eda_baseline.py

일자 매출(v2) 로드 → 마스터 패널 → 타깃 분포 / 일별·요일별 추세 / Naive baseline.

### 예상 결과 (실측)
```
마스터 패널: (774,589, 7)  기간 2021-01-01 ~ 2025-12-31  (425동 × 1,826일)

=== Naive Baseline (일 단위) ===
                        rmse     mae      r2     mape
Naive lag1 (어제)        0.358   0.151   0.917   0.818
Naive lag7 (지난주 동요일) 0.086   0.008   0.995   0.043   ← 거의 완벽
Group mean (행정동 평균)   0.305   0.170   0.940   0.920
```
> **해석**: lag7(지난주 같은 요일)이 R²=0.995 로 거의 완벽. v2 가 요일 패턴으로
> 분해된 데이터라 "같은 요일은 매출이 거의 동일"하기 때문. → 모델의 진짜 가치는
> 단순 예측이 아니라 **입지 추천(STEP 4)** 과 **해석(STEP 5)** 에 있음.

생성: `processed/panel_master_daily.parquet`, `results/01_*.png`, `01_naive_baseline_metrics.json`

---

## STEP 2 — 02_feature_engineering.py

13종 데이터를 일자 패널에 결합 → **약 103개 피처**. → `processed/features_daily_v2.parquet`

### 예상 결과 (실측)
```
최종 피처 매트릭스: (774,589, 112)  mem≈600MB
결측 있는 컬럼: 41/112  (지하철 45%, 생활인구 41%(2023~), 기상·대기질 20%(~2024))

Top |corr(피처, y)|:
  sales_lag7  0.998 | sales_lag28 0.990 | sales_roll7 0.979
  store_food  0.695 | ft_store_cnt 0.694   ← F&B 점포·시설이 외부 피처 중 최강
```
> 결측은 NaN 그대로(XGBoost 자동 처리). 데이터별 시간 가용성 차이 때문이며 정상.

생성: `processed/features_daily_v2.parquet`, `results/02_missing_rate.png`, `02_top_correlation.png`

---

## STEP 3 — 03_xgboost_tuned.py

일자 split + XGBoost(early stopping).

### 예상 결과 (실측)
```
train≈619,464  val≈76,925  test≈78,200   |  학습 피처 103개  best_iter≈155
Val  : RMSE=0.146  MAE=0.026  R²=0.987
Test : RMSE=0.093  MAE=0.023  R²=0.994
```
생성: `models/xgb_best.json` + `.pkl`, `results/03_xgb_*.png`, `03_xgb_metrics.json`,
`processed/predictions_test.parquet`

---

## STEP 4 — 04_evaluation_backtest.py

Walk-forward(90일×4 fold) + **v1 vs v2 비교** + Top-K 입지추천 + residual.
※ v1 피처가 없으면 자동 빌드(처음 1회 ~30초).

### 예상 결과 (실측, 축소 검증 기준)
```
            r2     rmse   recall@5
v1 (균등)   0.9997 0.022  1.000     ← 분기 내 매일 동일 → 가장 쉬움
v2 (요일)   0.996  0.07   ~0.97     ← 요일 구조 → 약간 더 현실적
```
> **인사이트**: "분기매출을 어떻게 일자로 쪼개느냐"가 모델 난이도를 좌우.
> v1(1/n)은 분기 내 변동이 0이라 trivial, v2(요일)는 weekday 신호가 있어 더 의미 있음.

생성: `results/04_backtest_metrics.csv`, `04_backtest_summary.png`, `04_residual_analysis.png`,
`processed/backtest_predictions.parquet`

---

## STEP 5 — 05_shap_interpretation.py

XGBoost 내장 SHAP(`pred_contribs=True`) + 카테고리별 기여도.

### 예상 결과 — 카테고리별 일매출 예측 기여도 (v2, 실측)
```
A. 매출 lag/추세  88.4%   ← 과거(특히 지난주 동요일) 매출이 압도적
L. 시설/집객      3.8%    ← F&B 점포·약국 등 정적 시설
I. 점포          1.8%
B. 캘린더        1.7%
G. 소비          1.7%
C. 생활인구       0.8%
D. 기상          0.7%  / F. 대기질 0.5% / H. 상권변화 0.2% / E. 공휴일 0.1%
```
> **시사점(중요)**: 일별 외부 피처(생활인구·기상·공휴일)는 결합돼 있지만 기여가 작음.
> 이는 **일별 변동이 실제 날씨·인구 반응이 아니라 분해 규칙(1/n·요일)으로 생성**됐기 때문.
> → 발표 포인트: "현재 daily 타깃은 분기값의 결정적 분해라 외부 일별 신호의 설명력이
> 제한적이다. 진짜 일별 신호를 보려면 POS·카드사 실거래 일자 데이터가 필요하다."

생성: `results/05_shap_*.png`

---

## 자주 마주칠 오류 & 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `No module named 'xgboost'` | venv 미활성화 | `source .venv/bin/activate` |
| `libxgboost ... libomp` | macOS OpenMP 없음 | `brew install libomp` |
| `Input data contains inf` | (수정 완료) growth 0나눗셈 | 이미 로그차분+inf가드 적용됨 |
| MemoryError / Killed | RAM 부족 | 노트북을 **순서대로 따로** 실행(한 프로세스에 다 넣지 말 것) |
| 한글 폰트 깨짐 | OS 한글폰트 없음 | 무시(그래프는 생성됨). mac=AppleGothic 자동 |
| `FileNotFoundError: features_daily_v2.parquet` | 02 미실행 | 01→02 순서대로 |

---

*모든 수치는 backtest_data/ 실데이터로 검증됨 (2026-06-03, 샌드박스 Python 3.10 + xgboost 3.2).*

---

# ★★ 서비스 트랙 — 입지 입력 → 추천 메뉴 + 예상 매출 (STEP 6~7)

> 최종 목표("웹에서 특정 입지 체크 → 추천 메뉴 + 예상 매출")에 맞춘 **서비스 코어 모델**.
> STEP 1~5(일자 모델)는 매출 시계열 예측(운영용)이고, 아래가 **입지 추천·창업 의사결정용**.

## 왜 별도 모델인가
- "추천 메뉴"는 **업종(category)별** 신호가 필요 → 일자 파일엔 업종이 없어 **분기 업종매출**을 씀.
- "최적입지"는 기록 없는 새 후보지를 점수화해야 함 → **self-lag 제외**, 입지속성만으로 학습.
- 한 모델이 (입지 × 업종) 매출을 예측 → 입지의 10개 업종을 모두 예측 → 최상위 = 추천 메뉴.

## STEP 6 — 06_location_menu_model.py
입지×업종×분기(74,336행, 2021~2025) + 입지속성(생활인구·집객시설·경쟁점포·소비력·유동인구).
- 검증 ① 시간 holdout(2025): **R²≈0.90**
- 검증 ② **leave-dong-out 공간 CV: R²≈0.71** ← "안 본 동네" 일반화 = 서비스 신뢰도 핵심
- SHAP 핵심 변수: 유사업종 밀집도 > 업종 점포수 > 업종 정체성 > 프랜차이즈 > 구매력(소비) > 은행 > 식품소비
- 산출: `models/xgb_location.*`, `location_feature_cols.json`, `results/06_location_*`

## STEP 7 — 07_service_demo.py  (+ src/serve.py)
`serve.py` 서비스 API:
```python
from src import serve
serve.recommend_menu("11680640")          # 역삼1동 추천 메뉴 랭킹
serve.predict_sales("11680640", "커피-음료") # (입지,업종) 예상 분기매출(원)
serve.rank_locations("커피-음료", top_k=10)  # 업종별 최적입지 Top10
serve.location_report("11680640")          # 서비스 JSON 응답
```
산출: `results/07_serving_predictions.csv`(4,250행), `07_best_menu_by_dong.csv`,
`service_demo/predictions.js`

## 웹 데모
`service_demo/index.html` 을 브라우저로 열면 끝(서버 불필요). 행정동 검색→추천 메뉴·예상매출,
업종 선택→최적입지 Top10. 실제 서비스의 프로토타입.

## 한계 & 발전 방향 (보고서용)
- 현재 매출 타깃은 **공공 추정매출**(분기) 기반 → 실거래(POS/카드) 일자 데이터가 들어오면
  일자 모델(STEP1~5)과 결합해 "요일·날씨까지 반영한 동적 예상매출"로 고도화 가능.
- 경쟁피처(점포수)는 진입 전후 변화를 단순화함 → 향후 수요·공급 분리 모델로 정교화.
