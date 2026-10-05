"""
GCI World 2026 - Credit-Term Experimental Program
Controlled experiment local OOF/holdout AUC: approximately 0.762337

This is the experimental model associated with the CREDIT_TERM feature:
    CREDIT_TERM = AMT_CREDIT / AMT_ANNUITY

It is NOT the public 0.76370 champion.
"""

import os
import gc
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

BASE = "input"
OUT = "output"
os.makedirs(OUT, exist_ok=True)

TRAIN = os.path.join(BASE, "train.csv")
TEST = os.path.join(BASE, "test.csv")
SAMPLE = os.path.join(BASE, "sample_submission.csv")

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
sample = pd.read_csv(SAMPLE)

y = train.pop("TARGET").astype("int8")

def feature_engineer(df):
    d = df.drop(columns=["SK_ID_CURR"], errors="ignore").copy()

    # Preserve rows; do not drop missing records.
    # Convert the DAYS_EMPLOYED sentinel to missing.
    if "DAYS_EMPLOYED" in d.columns:
        d.loc[d["DAYS_EMPLOYED"] > 300000, "DAYS_EMPLOYED"] = np.nan

    # Categorical missing values
    for c in d.select_dtypes(include="object").columns:
        d[c] = d[c].fillna("Missing")

    # Time features
    for c in [
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE",
    ]:
        if c in d.columns:
            d[c + "_Y"] = -d[c] / 365.25

    # Log features
    for c in [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    ]:
        if c in d.columns:
            d["LOG_" + c] = np.log1p(d[c].clip(lower=0))

    eps = 1e-6

    # The key experimental feature
    if "AMT_CREDIT" in d.columns and "AMT_ANNUITY" in d.columns:
        d["CREDIT_TERM"] = (
            d["AMT_CREDIT"] / (d["AMT_ANNUITY"].abs() + eps)
        )

    # Core financial ratios
    ratio_specs = [
        ("AMT_CREDIT", "AMT_INCOME_TOTAL", "CREDIT_INCOME_RATIO"),
        ("AMT_ANNUITY", "AMT_INCOME_TOTAL", "ANNUITY_INCOME_RATIO"),
        ("AMT_GOODS_PRICE", "AMT_CREDIT", "GOODS_CREDIT_RATIO"),
    ]

    for a, b, name in ratio_specs:
        if a in d.columns and b in d.columns:
            d[name] = d[a] / (d[b].abs() + eps)

    # EXT_SOURCE aggregates
    ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
           if c in d.columns]

    if ext:
        E = d[ext]
        d["EXT_MEAN"] = E.mean(axis=1)
        d["EXT_MIN"] = E.min(axis=1)
        d["EXT_MAX"] = E.max(axis=1)
        d["EXT_STD"] = E.std(axis=1)

    d["MISSING_COUNT"] = d.isna().sum(axis=1)

    return d


X = feature_engineer(train)
T = feature_engineer(test)

cats = X.select_dtypes(include="object").columns.tolist()
all_data = pd.concat([X, T], ignore_index=True)

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

medians = X.median()
X = X.fillna(medians).fillna(0)
T = T.fillna(medians).fillna(0)

del all_data
gc.collect()

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

pred = model.predict_proba(T)[:, 1]

submission = sample.copy()
submission["TARGET"] = pred

path = os.path.join(OUT, "submission_0_762337_credit_term.csv")
submission.to_csv(path, index=False)

print("Saved:", path)
print("Rows:", len(submission))
print("Prediction range:", float(pred.min()), float(pred.max()))
