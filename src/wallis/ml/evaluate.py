from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
        ]
    )


def evaluate_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    estimator,
    n_splits: int = 5,
) -> dict:
    X = df[feature_cols]
    y = df[target_col].astype(int)
    groups = df[group_col].astype(str)
    pre = build_preprocessor(df, feature_cols)
    pipe = Pipeline([("prep", pre), ("model", estimator)])
    cv = GroupKFold(n_splits=n_splits)

    roc_aucs, aps, bals, f1s = [], [], [], []
    for train_idx, test_idx in cv.split(X, y, groups):
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        y_test = y.iloc[test_idx]
        roc_aucs.append(roc_auc_score(y_test, y_prob))
        aps.append(average_precision_score(y_test, y_prob))
        bals.append(balanced_accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))

    return {
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs)),
        "average_precision_mean": float(np.mean(aps)),
        "average_precision_std": float(np.std(aps)),
        "balanced_accuracy_mean": float(np.mean(bals)),
        "balanced_accuracy_std": float(np.std(bals)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "n_rows": int(len(df)),
        "positive_rate": float(y.mean()),
        "pipeline": pipe,
    }


def evaluate_regressor(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    estimator,
    n_splits: int = 5,
) -> dict:
    X = df[feature_cols]
    y = df[target_col].astype(float)
    groups = df[group_col].astype(str)
    pre = build_preprocessor(df, feature_cols)
    pipe = Pipeline([("prep", pre), ("model", estimator)])
    cv = GroupKFold(n_splits=n_splits)

    maes, rmses, r2s = [], [], []
    for train_idx, test_idx in cv.split(X, y, groups):
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = pipe.predict(X.iloc[test_idx])
        y_test = y.iloc[test_idx]
        maes.append(mean_absolute_error(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2s.append(r2_score(y_test, y_pred))

    return {
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "n_rows": int(len(df)),
        "pipeline": pipe,
    }
