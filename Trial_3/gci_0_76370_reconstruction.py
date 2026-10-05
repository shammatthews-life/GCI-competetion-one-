"""
GCI World 2026 - Public Champion Reproduction Program
Target public leaderboard score: 0.76370

IMPORTANT:
- This file reproduces the feature/model family associated with our 0.76370
  public champion, but the exact original submission artifact is not available
  in the current runtime. Therefore, this is a reconstruction, not a guarantee
  of bit-for-bit reproduction.
- Set BASE to the folder containing train.csv, test.csv and sample_submission.csv.
"""

import os
import gc
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ============================================================
# PATHS
# ============================================================
BASE = "input"
OUT = "output"
os.makedirs(OUT, exist_ok=True)

TRAIN = os.path.join(BASE, "train.csv")
TEST = os.path.join(BASE, "test.csv")
SAMPLE = os.path.join(BASE, "sample_submission.csv")

# ============================================================
# LOAD
# ============================================================
train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
sample = pd.read_csv(SAMPLE)

y = train.pop("TARGET").astype("int8")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def feature_engineer(df):
    d = df.drop(columns=["SK_ID_CURR"], errors="ignore").copy()

    # Sentinel handling
    if "DAYS_EMPLOYED" in d.columns:
        d.loc[d["DAYS_EMPLOYED"] > 300000, "DAYS_EMPLOYED"] = np.nan

    # Missing categorical values
    cats = d.select_dtypes(include="object").columns
    for c in cats:
        d[c] = d[c].fillna("Missing")

    # Time representations
    for c in [
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE",
    ]:
        if c in d.columns:
            d[c + "_Y"] = -d[c] / 365.25

    # Log representations
    for c in [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    ]:
        if c in d.columns:
            d["LOG_" + c] = np.log1p(d[c].clip(lower=0))

    # Financial ratios
    eps = 1e-6
    ratio_specs = [
        ("AMT_CREDIT", "AMT_INCOME_TOTAL", "CREDIT_INCOME_RATIO"),
        ("AMT_ANNUITY", "AMT_INCOME_TOTAL", "ANNUITY_INCOME_RATIO"),
        ("AMT_ANNUITY", "AMT_CREDIT", "ANNUITY_CREDIT_RATIO"),
        ("AMT_GOODS_PRICE", "AMT_CREDIT", "GOODS_CREDIT_RATIO"),
        ("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "INCOME_PER_FAMILY"),
        ("AMT_CREDIT", "CNT_FAM_MEMBERS", "CREDIT_PER_FAMILY"),
    ]

    for a, b, name in ratio_specs:
        if a in d.columns and b in d.columns:
            d[name] = d[a] / (d[b].abs() + eps)

    # EXT_SOURCE aggregate / interaction features
    ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
           if c in d.columns]

    if ext:
        E = d[ext]
        d["EXT_MEAN"] = E.mean(axis=1)
        d["EXT_MIN"] = E.min(axis=1)
        d["EXT_MAX"] = E.max(axis=1)
        d["EXT_STD"] = E.std(axis=1)
        d["EXT_SUM"] = E.sum(axis=1)

        for i, a in enumerate(ext):
            d[a + "_SQ"] = d[a] ** 2
            for b in ext[i + 1:]:
                d[a + "_" + b] = d[a] * d[b]

    # Missingness / family structure
    d["MISSING_COUNT"] = d.isna().sum(axis=1)

    if "CNT_FAM_MEMBERS" in d.columns and "CNT_CHILDREN" in d.columns:
        d["ADULTS_IN_FAMILY"] = (
            d["CNT_FAM_MEMBERS"] - d["CNT_CHILDREN"]
        )

    return d


X = feature_engineer(train)
T = feature_engineer(test)

# ============================================================
# ONE-HOT ENCODING
# ============================================================
cats = X.select_dtypes(include="object").columns.tolist()

all_data = pd.concat([X, T], axis=0, ignore_index=True)

all_data = pd.get_dummies(
    all_data,
    columns=cats,
    dummy_na=True
)

all_data = (
    all_data
    .replace([np.inf, -np.inf], np.nan)
    .astype("float32")
)

X = all_data.iloc[:len(train)].copy()
T = all_data.iloc[len(train):].copy()

# Training-only median imputation
medians = X.median()
X = X.fillna(medians).fillna(0)
T = T.fillna(medians).fillna(0)

del all_data
gc.collect()

# ============================================================
# MODEL
# ============================================================
params = dict(
    n_estimators=1099,
    max_depth=3,
    learning_rate=0.04,
    min_child_weight=4,
    subsample=0.90,
    colsample_bytree=0.90,
    gamma=0,
    reg_alpha=0.05,
    reg_lambda=4,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    n_jobs=8,
    random_state=42,
)

model = XGBClassifier(**params)
model.fit(X, y)

# ============================================================
# SUBMISSION
# ============================================================
pred = model.predict_proba(T)[:, 1]

submission = sample.copy()
target_col = "TARGET"

# Preserve sample submission structure
if target_col in submission.columns:
    submission[target_col] = pred
else:
    submission[target_col] = pred

path = os.path.join(OUT, "submission_0_76370_reconstruction.csv")
submission.to_csv(path, index=False)

print("Saved:", path)
print("Rows:", len(submission))
print("Prediction range:", float(pred.min()), float(pred.max()))
