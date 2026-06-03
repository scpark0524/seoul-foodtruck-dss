"""
회귀 + Top-K 추천 평가 메트릭.

회귀: RMSE, MAE, R², MAPE
추천: Recall@K, NDCG@K, HitRate@K
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================
# 1) 회귀 메트릭
# ============================================
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape": np.nan}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    # MAPE 는 y=0 처리
    nonzero = np.abs(y_true) > 1e-9
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) \
        if nonzero.any() else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ============================================
# 2) Top-K 추천 메트릭
#    "각 분기마다 매출 Top-K 행정동을 모델이 얼마나 맞췄는가"
# ============================================
def _topk_per_group(scores: pd.Series, k: int) -> pd.Index:
    return scores.nlargest(k).index


def recall_at_k(
    df: pd.DataFrame, score_col: str, label_col: str, group_col: str, k: int = 10
) -> float:
    """그룹별로 Top-K 예측 vs Top-K 실측의 일치 비율."""
    out = []
    for _, g in df.groupby(group_col):
        if len(g) < k:
            continue
        pred_top = set(_topk_per_group(g[score_col], k))
        true_top = set(_topk_per_group(g[label_col], k))
        out.append(len(pred_top & true_top) / k)
    return float(np.mean(out)) if out else np.nan


def ndcg_at_k(
    df: pd.DataFrame, score_col: str, label_col: str, group_col: str, k: int = 10
) -> float:
    """순위 가중 정답률. 정답이 상위에 있을수록 점수 높음."""
    out = []
    for _, g in df.groupby(group_col):
        if len(g) < k:
            continue
        # 예측 순위 기준 정렬 후 실제 점수
        g_sorted = g.sort_values(score_col, ascending=False).head(k)
        gains = g_sorted[label_col].to_numpy(dtype=float)
        discounts = np.log2(np.arange(2, k + 2))
        dcg = np.sum(gains / discounts)
        ideal = g[label_col].nlargest(k).to_numpy(dtype=float)
        idcg = np.sum(ideal / discounts)
        if idcg > 0:
            out.append(dcg / idcg)
    return float(np.mean(out)) if out else np.nan


def hit_rate_at_k(
    df: pd.DataFrame, score_col: str, label_col: str, group_col: str, k: int = 10
) -> float:
    """Top-K 예측에 정답(=실제 Top-1) 이 포함된 그룹 비율."""
    out = []
    for _, g in df.groupby(group_col):
        if len(g) < k:
            continue
        pred_top = set(_topk_per_group(g[score_col], k))
        true_top1 = g[label_col].idxmax()
        out.append(1.0 if true_top1 in pred_top else 0.0)
    return float(np.mean(out)) if out else np.nan


def topk_metrics(
    df: pd.DataFrame, score_col: str, label_col: str, group_col: str,
    ks: tuple[int, ...] = (3, 5, 10),
) -> dict[str, float]:
    out = {}
    for k in ks:
        out[f"recall@{k}"]  = recall_at_k(df, score_col, label_col, group_col, k)
        out[f"ndcg@{k}"]    = ndcg_at_k(df, score_col, label_col, group_col, k)
        out[f"hit@{k}"]     = hit_rate_at_k(df, score_col, label_col, group_col, k)
    return out


# ============================================
# 3) Naive baseline 예측 (모델링_가이드.md Section 4)
# ============================================
def naive_lag(df: pd.DataFrame, group_col: str, time_col: str, value_col: str, lag: int = 1) -> pd.Series:
    """그룹별 lag 예측 (직전 분기 또는 작년 동분기)."""
    return df.sort_values([group_col, time_col]).groupby(group_col)[value_col].shift(lag)


def naive_group_mean(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """그룹 평균 예측."""
    return df.groupby(group_col)[value_col].transform("mean")
