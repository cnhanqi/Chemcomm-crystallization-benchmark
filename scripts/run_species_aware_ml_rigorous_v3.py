import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wallis.ml.datasets import get_species_screening_features, load_ml_table
from wallis.ml.evaluate import build_preprocessor

DATA = ROOT / "data" / "ml" / "species_aware_screening_ml_v2.csv"
OUT_DIR = ROOT / "results" / "ml" / "species_aware_v2"

def _build_pipeline(df: pd.DataFrame, feature_cols: list[str], estimator) -> Pipeline:
    pre = build_preprocessor(df, feature_cols)
    return Pipeline([("prep", pre), ("model", estimator)])

def main():
    df = load_ml_table(DATA)
    feature_cols = get_species_screening_features(df)
    group_col = "group_id"
    cls_target = "observed_in_structure_label"
    reg_target = "solvent_content_percent"

    # Define feature sets
    feature_sets = {
        "combined": feature_cols,
        "species_chemistry_only": [col for col in feature_cols if col != "protein"],
        "protein_only": ["protein"]
    }

    # 1. Classification Rigorous
    print("Running classification tuning...")
    cls_df = df.dropna(subset=[cls_target]).copy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(cls_df, groups=cls_df[group_col]))
    
    cls_train = cls_df.iloc[train_idx].copy()
    cls_test = cls_df.iloc[test_idx].copy()

    cls_results = []
    
    param_grid_cls = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20, None],
        'model__min_samples_leaf': [1, 2]
    }
    
    for fs_name, f_cols in feature_sets.items():
        print(f"  Feature set: {fs_name}")
        estimator = RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        pipe = _build_pipeline(cls_train, f_cols, estimator)
        
        cv = GroupKFold(n_splits=3)
        grid = GridSearchCV(pipe, param_grid=param_grid_cls, cv=cv, scoring='roc_auc', n_jobs=1)
        
        grid.fit(cls_train[f_cols], cls_train[cls_target].astype(int), groups=cls_train[group_col])
        best_pipe = grid.best_estimator_
        
        y_test_pred_prob = best_pipe.predict_proba(cls_test[f_cols])[:, 1]
        y_test_pred_class = best_pipe.predict(cls_test[f_cols])
        
        y_test = cls_test[cls_target].astype(int)
        
        roc_auc = roc_auc_score(y_test, y_test_pred_prob)
        avg_prec = average_precision_score(y_test, y_test_pred_prob)
        bal_acc = balanced_accuracy_score(y_test, y_test_pred_class)
        
        print(f"    Test ROC-AUC: {roc_auc:.4f}")
        
        cls_results.append({
            'feature_set': fs_name,
            'model': 'RandomForestClassifier_GridSearchCV',
            'test_roc_auc': roc_auc,
            'test_avg_precision': avg_prec,
            'test_balanced_accuracy': bal_acc,
            'best_params': str(grid.best_params_)
        })

    # 2. Regression Rigorous
    print("Running regression tuning...")
    reg_df = df.dropna(subset=[reg_target]).copy()
    reg_df[reg_target] = reg_df.groupby("protein")[reg_target].transform(lambda x: x - x.median())
    
    gss_reg = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx_reg, test_idx_reg = next(gss_reg.split(reg_df, groups=reg_df[group_col]))
    
    reg_train = reg_df.iloc[train_idx_reg].copy()
    reg_test = reg_df.iloc[test_idx_reg].copy()

    reg_results = []
    param_grid_reg = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20, None],
        'model__min_samples_leaf': [1, 2]
    }

    for fs_name, f_cols in feature_sets.items():
        print(f"  Feature set: {fs_name}")
        estimator = RandomForestRegressor(random_state=42, n_jobs=-1)
        pipe = _build_pipeline(reg_train, f_cols, estimator)
        
        cv = GroupKFold(n_splits=3)
        grid = GridSearchCV(pipe, param_grid=param_grid_reg, cv=cv, scoring='r2', n_jobs=1)
        
        grid.fit(reg_train[f_cols], reg_train[reg_target], groups=reg_train[group_col])
        best_pipe = grid.best_estimator_
        
        y_test_pred = best_pipe.predict(reg_test[f_cols])
        y_test = reg_test[reg_target]
        
        r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        print(f"    Test R2: {r2:.4f}")
        
        reg_results.append({
            'feature_set': fs_name,
            'model': 'RandomForestRegressor_GridSearchCV',
            'test_r2': r2,
            'test_mae': mae,
            'test_rmse': rmse,
            'best_params': str(grid.best_params_)
        })

    pd.DataFrame(cls_results).to_csv(OUT_DIR / "rigorous_classification_metrics_v3.csv", index=False)
    pd.DataFrame(reg_results).to_csv(OUT_DIR / "rigorous_regression_metrics_v3.csv", index=False)
    print("Done. Saved to rigorous metrics CSVs.")

if __name__ == '__main__':
    main()

