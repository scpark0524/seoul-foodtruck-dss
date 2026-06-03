"""개선 루프 실험 헬퍼 — 공간(leave-dong-out) CV 로 상대 비교."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from .config import PROCESSED_DIR, RESULTS_DIR, KEY_DONG, RANDOM_STATE
from .features_location import location_daily_feature_cols, DAILY_CAT_LABEL

LOG = RESULTS_DIR / "08_improvement_log.json"

def load_exp(frac=0.4, seed=42):
    df = pd.read_parquet(PROCESSED_DIR / "features_location_daily_v2.parquet")
    if frac < 1.0:
        df = df.sample(frac=frac, random_state=seed).reset_index(drop=True)
    return df

def spatial_cv(df, feats, params, n_splits=3):
    gkf = GroupKFold(n_splits=n_splits); g = df[KEY_DONG].values
    r2s=[]
    for tr_i, te_i in gkf.split(df, groups=g):
        m = xgb.XGBRegressor(**params)
        m.fit(df.iloc[tr_i][feats], df.iloc[tr_i][DAILY_CAT_LABEL], verbose=False)
        r2s.append(r2_score(df.iloc[te_i][DAILY_CAT_LABEL], m.predict(df.iloc[te_i][feats])))
    return float(np.mean(r2s)), [round(x,4) for x in r2s]

def log_result(step, info):
    data = json.loads(LOG.read_text()) if LOG.exists() else {}
    data[step] = info
    LOG.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return data

BASE_PARAMS = dict(n_estimators=200, max_depth=7, learning_rate=0.08,
                   subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                   random_state=RANDOM_STATE, tree_method="hist")
