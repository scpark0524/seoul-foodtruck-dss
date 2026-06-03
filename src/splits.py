"""
시계열 기반 train/val/test 분할 + Walk-forward Cross-Validation.

핵심 원칙: 시간 순서대로만 split → 미래 정보 누수(data leakage) 방지.
"""
from __future__ import annotations
from typing import Iterator
import pandas as pd

from .config import ALL_QU, TRAIN_QU, VAL_QU, TEST_QU, KEY_TIME


def time_holdout_split(
    df: pd.DataFrame,
    train_qu: list[int] | None = None,
    val_qu: list[int] | None = None,
    test_qu: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    시간 기반 holdout split (모델링_가이드.md Section 3).
    반환: (train, val, test) DataFrame.
    """
    train_qu = train_qu or TRAIN_QU
    val_qu   = val_qu   or VAL_QU
    test_qu  = test_qu  or TEST_QU

    train = df[df[KEY_TIME].isin(train_qu)].copy()
    val   = df[df[KEY_TIME].isin(val_qu)].copy()
    test  = df[df[KEY_TIME].isin(test_qu)].copy()
    print(f"[SPLIT] train={len(train):,} val={len(val):,} test={len(test):,}")
    return train, val, test


def walk_forward_folds(
    df: pd.DataFrame,
    min_train_qu: int = 4,
    test_window: int = 2,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, str]]:
    """
    Walk-forward (expanding window) cross-validation.

    Fold k:
      train = ALL_QU[:min_train_qu + k*test_window]
      test  = ALL_QU[min_train_qu + k*test_window : ... + test_window]

    예) min_train_qu=4, test_window=2, ALL_QU 12개 →
      Fold 1: train Q1~Q4   test Q5~Q6
      Fold 2: train Q1~Q6   test Q7~Q8
      Fold 3: train Q1~Q8   test Q9~Q10
      Fold 4: train Q1~Q10  test Q11~Q12

    Yields: (train_df, test_df, fold_name)
    """
    qu = ALL_QU
    fold = 0
    train_end = min_train_qu
    while train_end + test_window <= len(qu):
        train_q = qu[:train_end]
        test_q  = qu[train_end:train_end + test_window]
        fold += 1
        name = f"fold{fold}_train{train_q[0]}-{train_q[-1]}_test{test_q[0]}-{test_q[-1]}"
        tr = df[df[KEY_TIME].isin(train_q)].copy()
        te = df[df[KEY_TIME].isin(test_q)].copy()
        yield tr, te, name
        train_end += test_window


# ============================================================
# ★ 일자(DAILY) 단위 split  (2026-06 추가)
# ============================================================
import pandas as _pd
from .config import KEY_DATE, DAILY_TRAIN_END, DAILY_VAL_END


def date_holdout_split(df, train_end=None, val_end=None):
    """일자 기반 holdout. train: ~train_end, val: ~val_end, test: 그 이후."""
    train_end = _pd.Timestamp(train_end or DAILY_TRAIN_END)
    val_end   = _pd.Timestamp(val_end or DAILY_VAL_END)
    d = df[KEY_DATE]
    train = df[d <= train_end].copy()
    val   = df[(d > train_end) & (d <= val_end)].copy()
    test  = df[d > val_end].copy()
    print(f"[SPLIT-daily] train={len(train):,} (~{train_end.date()}) "
          f"val={len(val):,} (~{val_end.date()}) test={len(test):,} (>{val_end.date()})")
    return train, val, test


def walk_forward_daily(df, n_folds=4, test_days=90):
    """일자 expanding-window walk-forward. 마지막 구간들을 test_days 씩 잘라 평가."""
    dates = df[KEY_DATE]
    dmax = dates.max()
    folds = []
    for k in range(n_folds, 0, -1):
        te_start = dmax - _pd.Timedelta(days=test_days * k) + _pd.Timedelta(days=1)
        te_end   = dmax - _pd.Timedelta(days=test_days * (k - 1))
        tr = df[dates < te_start].copy()
        te = df[(dates >= te_start) & (dates <= te_end)].copy()
        if len(tr) == 0 or len(te) == 0:
            continue
        name = f"fold{n_folds-k+1}_test_{te_start.date()}~{te_end.date()}"
        folds.append((tr, te, name))
    return folds


# ============================================================
# ★ 공간(leave-dong-out) 교차검증  (2026-06 추가) — 서비스 입지모델용
#    "한 번도 안 본 행정동"의 매출을 맞히는지 검증 → 신규 후보입지 일반화 확인.
# ============================================================
def leave_dong_out_cv(df, n_splits=5, seed=42):
    """행정동(KEY_DONG)을 그룹으로 GroupKFold. yields (train_df, test_df, fold_name)."""
    from sklearn.model_selection import GroupKFold
    from .config import KEY_DONG
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[KEY_DONG].values
    for i, (tr_idx, te_idx) in enumerate(gkf.split(df, groups=groups), 1):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        name = f"spatial_fold{i}_testdongs={te[KEY_DONG].nunique()}"
        yield tr.copy(), te.copy(), name


def time_holdout_quarters(df, train_qu, test_qu):
    """분기 기반 holdout (입지×업종 모델용)."""
    from .config import KEY_TIME
    tr = df[df[KEY_TIME].isin(train_qu)].copy()
    te = df[df[KEY_TIME].isin(test_qu)].copy()
    return tr, te
