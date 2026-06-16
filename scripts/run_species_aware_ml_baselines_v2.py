from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wallis.ml.datasets import get_species_screening_features, load_ml_table
from wallis.ml.evaluate import evaluate_classifier, evaluate_regressor
from wallis.ml.explain import permutation_importance_table
DATA = ROOT / "data" / "ml" / "species_aware_screening_ml_v2.csv"
OUT_DIR = ROOT / "results" / "ml" / "species_aware_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fast_classification_models(random_state: int = 42) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=8000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def fast_regression_models(random_state: int = 42) -> dict:
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }


def main() -> None:
    df = load_ml_table(DATA)
    feature_cols = get_species_screening_features(df)
    group_col = "group_id"
    random_state = 42
    n_splits = 3

    feature_sets = {
        "protein_only": ["protein"],
        "protein_and_species_only": ["protein", "species_name"],
        "species_chemistry_only": [col for col in feature_cols if col != "protein"],
        "combined": feature_cols,
    }

    cls_target = "observed_in_structure_label"
    reg_target = "solvent_content_percent"

    cls_df = df.dropna(subset=[cls_target]).copy()
    reg_df = df.dropna(subset=[reg_target]).copy()
    # Address shortcut learning / regression alignment by standardizing per protein
    reg_df[reg_target] = reg_df.groupby("protein")[reg_target].transform(lambda x: x - x.median())

    cls_records: list[dict] = []
    best_cls_artifact: dict | None = None
    best_cls_score = float("-inf")
    for feature_set_name, cols in feature_sets.items():
        for model_name, estimator in fast_classification_models(random_state).items():
            result = evaluate_classifier(
                cls_df,
                cols,
                cls_target,
                group_col,
                estimator,
                n_splits=n_splits,
            )
            cls_records.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "target": cls_target,
                    "n_features": len(cols),
                    "roc_auc_mean": result["roc_auc_mean"],
                    "roc_auc_std": result["roc_auc_std"],
                    "average_precision_mean": result["average_precision_mean"],
                    "average_precision_std": result["average_precision_std"],
                    "balanced_accuracy_mean": result["balanced_accuracy_mean"],
                    "balanced_accuracy_std": result["balanced_accuracy_std"],
                    "f1_mean": result["f1_mean"],
                    "f1_std": result["f1_std"],
                    "n_rows": result["n_rows"],
                    "positive_rate": result["positive_rate"],
                }
            )
            if feature_set_name == "combined" and result["roc_auc_mean"] > best_cls_score:
                best_cls_score = result["roc_auc_mean"]
                best_cls_artifact = {
                    "model_name": model_name,
                    "feature_cols": cols,
                    "pipeline": result["pipeline"],
                }

    reg_records: list[dict] = []
    best_reg_artifact: dict | None = None
    best_reg_score = float("-inf")
    for feature_set_name, cols in feature_sets.items():
        for model_name, estimator in fast_regression_models(random_state).items():
            result = evaluate_regressor(
                reg_df,
                cols,
                reg_target,
                group_col,
                estimator,
                n_splits=n_splits,
            )
            reg_records.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "target": reg_target,
                    "n_features": len(cols),
                    "mae_mean": result["mae_mean"],
                    "mae_std": result["mae_std"],
                    "rmse_mean": result["rmse_mean"],
                    "rmse_std": result["rmse_std"],
                    "r2_mean": result["r2_mean"],
                    "r2_std": result["r2_std"],
                    "n_rows": result["n_rows"],
                }
            )
            if feature_set_name == "combined" and result["r2_mean"] > best_reg_score:
                best_reg_score = result["r2_mean"]
                best_reg_artifact = {
                    "model_name": model_name,
                    "feature_cols": cols,
                    "pipeline": result["pipeline"],
                }

    cls_results = pd.DataFrame(cls_records).sort_values(
        ["feature_set", "roc_auc_mean"], ascending=[True, False]
    )
    reg_results = pd.DataFrame(reg_records).sort_values(
        ["feature_set", "r2_mean"], ascending=[True, False]
    )
    cls_results.to_csv(OUT_DIR / "classification_model_comparison.csv", index=False)
    reg_results.to_csv(OUT_DIR / "regression_model_comparison.csv", index=False)

    if best_cls_artifact is not None:
        best_cls_pipeline = best_cls_artifact["pipeline"]
        best_cls_pipeline.fit(cls_df[best_cls_artifact["feature_cols"]], cls_df[cls_target])
        cls_sample = cls_df.sample(n=min(2500, len(cls_df)), random_state=random_state)
        cls_importance = permutation_importance_table(
            best_cls_pipeline,
            cls_sample[best_cls_artifact["feature_cols"]],
            cls_sample[cls_target],
            best_cls_artifact["feature_cols"],
            random_state=random_state,
            n_repeats=4,
        )
        cls_importance.to_csv(OUT_DIR / "classification_feature_importance.csv", index=False)
    else:
        cls_importance = pd.DataFrame()

    if best_reg_artifact is not None:
        best_reg_pipeline = best_reg_artifact["pipeline"]
        best_reg_pipeline.fit(reg_df[best_reg_artifact["feature_cols"]], reg_df[reg_target])
        reg_sample = reg_df.sample(n=min(2500, len(reg_df)), random_state=random_state)
        reg_importance = permutation_importance_table(
            best_reg_pipeline,
            reg_sample[best_reg_artifact["feature_cols"]],
            reg_sample[reg_target],
            best_reg_artifact["feature_cols"],
            random_state=random_state,
            n_repeats=4,
        )
        reg_importance.to_csv(OUT_DIR / "regression_feature_importance.csv", index=False)
    else:
        reg_importance = pd.DataFrame()

    summary = {
        "dataset": str(DATA),
        "n_rows_classification": int(len(cls_df)),
        "n_rows_regression": int(len(reg_df)),
        "group_count": int(df[group_col].nunique()),
        "classification_target": cls_target,
        "regression_target": reg_target,
        "best_classification_model_combined": None if best_cls_artifact is None else best_cls_artifact["model_name"],
        "best_regression_model_combined": None if best_reg_artifact is None else best_reg_artifact["model_name"],
        "best_classification_roc_auc": None if cls_results.empty else float(
            cls_results[cls_results["feature_set"] == "combined"]["roc_auc_mean"].max()
        ),
        "best_regression_r2": None if reg_results.empty else float(
            reg_results[reg_results["feature_set"] == "combined"]["r2_mean"].max()
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Species-Aware ML Baselines v2",
        "",
        f"- classification target: `{cls_target}`",
        f"- regression target: `{reg_target}`",
        f"- grouped entries: `{df[group_col].nunique()}`",
        f"- species-aware rows: `{len(df)}`",
        "",
        "## Headline classification result",
        "",
    ]
    if not cls_results.empty:
        top_cls = cls_results[cls_results["feature_set"] == "combined"].sort_values(
            "roc_auc_mean", ascending=False
        ).iloc[0]
        md_lines.extend(
            [
                f"- best combined model: `{top_cls['model']}`",
                f"- grouped CV ROC-AUC: `{top_cls['roc_auc_mean']:.3f} +/- {top_cls['roc_auc_std']:.3f}`",
                f"- grouped CV AP: `{top_cls['average_precision_mean']:.3f} +/- {top_cls['average_precision_std']:.3f}`",
                f"- grouped CV balanced accuracy: `{top_cls['balanced_accuracy_mean']:.3f} +/- {top_cls['balanced_accuracy_std']:.3f}`",
            ]
        )

    md_lines.extend(["", "## Headline regression result", ""])
    if not reg_results.empty:
        top_reg = reg_results[reg_results["feature_set"] == "combined"].sort_values(
            "r2_mean", ascending=False
        ).iloc[0]
        md_lines.extend(
            [
                f"- best combined model: `{top_reg['model']}`",
                f"- grouped CV R2: `{top_reg['r2_mean']:.3f} +/- {top_reg['r2_std']:.3f}`",
                f"- grouped CV MAE: `{top_reg['mae_mean']:.3f} +/- {top_reg['mae_std']:.3f}`",
                f"- grouped CV RMSE: `{top_reg['rmse_mean']:.3f} +/- {top_reg['rmse_std']:.3f}`",
            ]
        )

    md_lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `classification_model_comparison.csv`",
            "- `regression_model_comparison.csv`",
            "- `classification_feature_importance.csv`",
            "- `regression_feature_importance.csv`",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

