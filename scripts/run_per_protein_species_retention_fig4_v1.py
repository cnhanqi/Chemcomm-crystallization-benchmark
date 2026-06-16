from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ml" / "species_aware_screening_ml_v2.csv"
PER_PROTEIN_DIR = ROOT / "data" / "ml" / "per_protein_species_retention_v1"
OUT_DIR = ROOT / "results" / "ml" / "per_protein_species_retention_v1"
FIGURE_ASSET_DIR = ROOT / "results" / "figure_assets"

TARGET = "observed_in_structure_label"
ALPHA = 10.0
BETA = 5.0
RANDOM_STATE = 42

PROTEIN_FILE_NAMES = {
    "lysozyme": "df_lysozyme.csv",
    "ribonuclease": "df_ribonuclease.csv",
    "trypsin": "df_trypsin.csv",
    "insulin": "df_insulin.csv",
    "proteinase k": "df_proteinaseK.csv",
}

FEATURE_SETS = {
    "species_class_only": ["species_class"],
    "species_identity_only": ["species_name"],
    "condition_context_only": ["concentration_bin", "pH_bin", "temperature_bin", "method_family"],
    "species_identity_plus_class": ["species_name", "species_class"],
    "full_model": [
        "species_name",
        "species_class",
        "concentration_bin",
        "pH_bin",
        "temperature_bin",
        "method_family",
    ],
}

CLASS_ORDER = [
    "Anions",
    "Metal/cations",
    "Neutral solvents/polyols",
    "PEG/polymer",
    "Organic ligands/additives",
    "IL/DES",
]

CASE_SYSTEMS = [
    ("lysozyme", "nitrate"),
    ("ribonuclease", "magnesium"),
    ("trypsin", "calcium"),
    ("insulin", "zinc"),
    ("proteinase k", "calcium"),
]

STRUCTURAL_CASE_SYSTEMS = CASE_SYSTEMS + [
    ("proteinase k", "nitrate"),
    ("ribonuclease", "sulfate"),
]

ANIONS = {
    "nitrate",
    "sulfate",
    "chloride",
    "acetate",
    "citrate",
    "phosphate",
    "formate",
    "iodide",
    "bromide",
    "tartrate",
    "malate",
}
METAL_CATIONS = {
    "calcium",
    "magnesium",
    "zinc",
    "sodium",
    "potassium",
    "manganese",
    "cobalt",
    "copper",
    "ammonium",
    "lithium",
    "cesium",
    "cadmium",
    "nickel",
}
NEUTRAL_SOLVENTS = {
    "glycerol",
    "ethylene glycol",
    "mpd",
    "2-propanol",
    "acetone",
    "dioxane",
    "ethanol",
    "methanol",
    "dmf",
}
PEG_POLYMERS = {"peg"}
ORGANIC_ADDITIVES = {
    "phenol",
    "cresol",
    "imidazole",
    "tris",
    "hepes",
    "mes",
    "bis-tris",
    "cacodylate",
    "dtt",
    "chaps",
}
IL_DES = {"ethylammonium nitrate", "ionic liquid"}

CURATED_LOW_SUPPORT_MAIN_FIGURE = {
    ("insulin", "cresol"),
    ("ribonuclease", "iodide"),
    ("proteinase k", "glycerol"),
}

CONCENTRATION_BIN_LABELS = ["trace-low", "low-mid", "mid", "high", "very high", "missing"]
PH_BIN_LABELS = ["acidic", "mildly acidic", "neutral", "mildly basic", "basic", "missing"]

CONCENTRATION_BIN_ORDER = {
    "trace-low": 0,
    "low-mid": 1,
    "mid": 2,
    "high": 3,
    "very high": 4,
}


def classify_species(row: pd.Series) -> str:
    name = str(row["species_name"]).strip().lower()
    major = str(row.get("species_major_class", "")).strip().lower()
    role = str(row.get("species_role_class", "")).strip().lower()

    if name in IL_DES or major == "ionic_liquid_or_des" or bool(row.get("is_ionic_liquid_or_des", 0)):
        return "IL/DES"
    if name in ANIONS or bool(row.get("is_anion", 0)):
        return "Anions"
    if name in METAL_CATIONS or bool(row.get("is_cation", 0)):
        return "Metal/cations"
    if name in PEG_POLYMERS or bool(row.get("is_polymeric_precipitant", 0)):
        return "PEG/polymer"
    if name in NEUTRAL_SOLVENTS or bool(row.get("is_solvent_like", 0)) or bool(row.get("is_polyol_like", 0)):
        return "Neutral solvents/polyols"
    if name in ORGANIC_ADDITIVES or "buffer" in major or "additive" in major or "buffer" in role:
        return "Organic ligands/additives"
    return "Organic ligands/additives"


def normalize_concentration_bin(value: str | float) -> str:
    if pd.isna(value):
        return "missing"
    text = str(value).strip()
    mapping = {
        "trace_to_low": "trace-low",
        "low_to_mid": "low-mid",
        "mid": "mid",
        "high": "high",
        "very_high": "very high",
        "missing": "missing",
        "nonmolar_numeric": "missing",
    }
    return mapping.get(text, "missing")


def pH_bin(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value < 5.5:
        return "acidic"
    if value < 6.5:
        return "mildly acidic"
    if value < 7.5:
        return "neutral"
    if value < 8.5:
        return "mildly basic"
    return "basic"


def temperature_bin(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value < 285:
        return "cold"
    if value <= 298:
        return "ambient"
    return "warm"


def support_flag(n_condition: int) -> str:
    if n_condition >= 20:
        return "robust"
    if n_condition >= 10:
        return "moderate"
    if n_condition >= 3:
        return "low_support"
    return "exploratory"


def median_concentration_bin(values: pd.Series) -> str:
    cleaned = values.dropna().astype(str)
    molar = cleaned[cleaned.isin(CONCENTRATION_BIN_ORDER)]
    if len(molar):
        median_rank = int(round(float(np.median(molar.map(CONCENTRATION_BIN_ORDER)))))
        inverse = {v: k for k, v in CONCENTRATION_BIN_ORDER.items()}
        return inverse[median_rank]
    if (cleaned == "nonmolar_numeric").any():
        return "nonmolar_numeric"
    return "missing"


def prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["protein"] = work["protein"].astype(str).str.lower()
    work["species_name"] = work["species_name"].astype(str).str.lower()
    work = work[work["protein"].isin(PROTEIN_FILE_NAMES)].copy()
    work["species_class"] = work.apply(classify_species, axis=1)
    work["pH_bin"] = work["p_h"].apply(pH_bin)
    work["temperature_bin"] = work["temp_k"].apply(temperature_bin)
    work["concentration_bin"] = work["concentration_bin"].apply(normalize_concentration_bin)
    work["method_family"] = work["method_family"].fillna("missing").astype(str)
    work[TARGET] = work[TARGET].astype(int)
    return work


def write_per_protein_tables(df: pd.DataFrame) -> None:
    PER_PROTEIN_DIR.mkdir(parents=True, exist_ok=True)
    combined_cols = [
        "protein",
        "pdb_id",
        "crystal_id",
        "species_name",
        "species_class",
        "concentration_bin",
        "pH_bin",
        "temperature_bin",
        "method_family",
        TARGET,
    ]
    df[combined_cols].to_csv(PER_PROTEIN_DIR / "model_input_table.csv", index=False)

    per_protein_cols = [c for c in combined_cols if c != "protein"]
    for protein, filename in PROTEIN_FILE_NAMES.items():
        subset = df[df["protein"] == protein].copy()
        subset[per_protein_cols].to_csv(PER_PROTEIN_DIR / filename, index=False)


def make_preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer([("cat", categorical, [])], remainder="drop")


def build_pipeline(model_name: str, feature_cols: list[str]) -> Pipeline:
    preprocessor = make_preprocessor()
    preprocessor.transformers[0] = ("cat", preprocessor.transformers[0][1], feature_cols)
    if model_name == "regularized_logistic_regression":
        estimator = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=10000,
            random_state=RANDOM_STATE,
        )
    elif model_name == "random_forest_sensitivity":
        estimator = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def evaluate_model(
    protein_df: pd.DataFrame,
    protein: str,
    feature_set: str,
    feature_cols: list[str],
    model_name: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if protein == "proteinase k":
        record = {
            "protein": protein,
            "feature_set": feature_set,
            "model": model_name,
            "n_rows": len(protein_df),
            "n_groups": protein_df["group_id"].nunique(),
            "positive_rate": protein_df[TARGET].mean(),
            "roc_auc_mean": np.nan,
            "roc_auc_std": np.nan,
            "average_precision_mean": np.nan,
            "average_precision_std": np.nan,
            "balanced_accuracy_mean": np.nan,
            "balanced_accuracy_std": np.nan,
            "brier_score_mean": np.nan,
            "brier_score_std": np.nan,
            "report_in_si": False,
            "evaluation_note": "small_sample_hybrid_use_shrunken_probability_no_standalone_auc",
        }
        return record, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X = protein_df[feature_cols]
    y = protein_df[TARGET].astype(int)
    groups = protein_df["group_id"].astype(str)
    n_splits = min(5, groups.nunique())
    if n_splits < 2 or y.nunique() < 2:
        record = {
            "protein": protein,
            "feature_set": feature_set,
            "model": model_name,
            "n_rows": len(protein_df),
            "n_groups": protein_df["group_id"].nunique(),
            "positive_rate": y.mean(),
            "roc_auc_mean": np.nan,
            "roc_auc_std": np.nan,
            "average_precision_mean": np.nan,
            "average_precision_std": np.nan,
            "balanced_accuracy_mean": np.nan,
            "balanced_accuracy_std": np.nan,
            "brier_score_mean": np.nan,
            "brier_score_std": np.nan,
            "report_in_si": False,
            "evaluation_note": "insufficient_rows_or_single_class",
        }
        return record, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    pred_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        pipe = build_pipeline(model_name, feature_cols)
        pipe.fit(X.iloc[train_idx], y_train)
        y_prob = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        fold_rows.append(
            {
                "protein": protein,
                "feature_set": feature_set,
                "model": model_name,
                "fold": fold,
                "n_test": len(test_idx),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "average_precision": average_precision_score(y_test, y_prob),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "brier_score": brier_score_loss(y_test, y_prob),
            }
        )
        fold_pred = protein_df.iloc[test_idx][
            ["protein", "pdb_id", "crystal_id", "species_name", "species_class", "group_id", TARGET]
        ].copy()
        fold_pred["feature_set"] = feature_set
        fold_pred["model"] = model_name
        fold_pred["fold"] = fold
        fold_pred["predicted_retained_probability"] = y_prob
        pred_rows.extend(fold_pred.to_dict("records"))

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)
    if fold_df.empty:
        record = {
            "protein": protein,
            "feature_set": feature_set,
            "model": model_name,
            "n_rows": len(protein_df),
            "n_groups": protein_df["group_id"].nunique(),
            "positive_rate": y.mean(),
            "roc_auc_mean": np.nan,
            "roc_auc_std": np.nan,
            "average_precision_mean": np.nan,
            "average_precision_std": np.nan,
            "balanced_accuracy_mean": np.nan,
            "balanced_accuracy_std": np.nan,
            "brier_score_mean": np.nan,
            "brier_score_std": np.nan,
            "report_in_si": False,
            "evaluation_note": "cross_validation_folds_not_estimable",
        }
        return record, pred_df, pd.DataFrame(), pd.DataFrame()

    record = {
        "protein": protein,
        "feature_set": feature_set,
        "model": model_name,
        "n_rows": len(protein_df),
        "n_groups": protein_df["group_id"].nunique(),
        "positive_rate": y.mean(),
        "roc_auc_mean": fold_df["roc_auc"].mean(),
        "roc_auc_std": fold_df["roc_auc"].std(ddof=0),
        "average_precision_mean": fold_df["average_precision"].mean(),
        "average_precision_std": fold_df["average_precision"].std(ddof=0),
        "balanced_accuracy_mean": fold_df["balanced_accuracy"].mean(),
        "balanced_accuracy_std": fold_df["balanced_accuracy"].std(ddof=0),
        "brier_score_mean": fold_df["brier_score"].mean(),
        "brier_score_std": fold_df["brier_score"].std(ddof=0),
        "report_in_si": True,
        "evaluation_note": "grouped_stratified_cv",
    }

    calibration_rows = []
    if not pred_df.empty and pred_df[TARGET].nunique() == 2:
        frac_pos, mean_pred = calibration_curve(
            pred_df[TARGET].astype(int),
            pred_df["predicted_retained_probability"],
            n_bins=5,
            strategy="quantile",
        )
        counts = pd.qcut(
            pred_df["predicted_retained_probability"],
            q=min(5, pred_df["predicted_retained_probability"].nunique()),
            duplicates="drop",
        ).value_counts(sort=False)
        for idx, (observed, predicted) in enumerate(zip(frac_pos, mean_pred), start=1):
            calibration_rows.append(
                {
                    "protein": protein,
                    "feature_set": feature_set,
                    "model": model_name,
                    "calibration_bin": idx,
                    "mean_predicted_probability": predicted,
                    "observed_fraction": observed,
                    "n": int(counts.iloc[idx - 1]) if idx - 1 < len(counts) else np.nan,
                }
            )

    stability_rows = []
    if not pred_df.empty:
        overall = (
            pred_df.groupby("species_name", as_index=False)
            .agg(overall_predicted_probability=("predicted_retained_probability", "mean"), n=("species_name", "size"))
        )
        overall = overall[overall["n"] >= 10]
        for fold, fold_part in pred_df.groupby("fold"):
            fold_species = (
                fold_part.groupby("species_name", as_index=False)
                .agg(fold_predicted_probability=("predicted_retained_probability", "mean"), n=("species_name", "size"))
            )
            merged = overall.merge(fold_species, on="species_name", how="inner")
            merged = merged[merged["n_y"] >= 3]
            if len(merged) >= 4:
                rho = spearmanr(
                    merged["overall_predicted_probability"],
                    merged["fold_predicted_probability"],
                    nan_policy="omit",
                ).statistic
                if not math.isnan(rho):
                    stability_rows.append(
                        {
                            "protein": protein,
                            "feature_set": feature_set,
                            "model": model_name,
                            "fold": fold,
                            "species_compared": len(merged),
                            "spearman_rho": rho,
                        }
                    )

    return record, pred_df, pd.DataFrame(calibration_rows), pd.DataFrame(stability_rows)


def logistic_odds_ratios(protein_df: pd.DataFrame, protein: str, feature_set: str, feature_cols: list[str]) -> pd.DataFrame:
    if protein_df[TARGET].nunique() < 2 or protein == "proteinase k":
        return pd.DataFrame()
    pipe = build_pipeline("regularized_logistic_regression", feature_cols)
    pipe.fit(protein_df[feature_cols], protein_df[TARGET].astype(int))
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_.ravel()
    out = pd.DataFrame(
        {
            "protein": protein,
            "feature_set": feature_set,
            "term": names,
            "coefficient_log_odds": coefs,
            "odds_ratio": np.exp(coefs),
        }
    )
    return out.sort_values("odds_ratio", ascending=False)


def run_feature_set_ablation(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_records = []
    prediction_frames = []
    calibration_frames = []
    stability_frames = []
    odds_ratio_frames = []

    for protein in PROTEIN_FILE_NAMES:
        protein_df = df[df["protein"] == protein].copy()
        for feature_set, feature_cols in FEATURE_SETS.items():
            for model_name in ["regularized_logistic_regression", "random_forest_sensitivity"]:
                metrics, preds, calibration, stability = evaluate_model(
                    protein_df, protein, feature_set, feature_cols, model_name
                )
                metric_records.append(metrics)
                if not preds.empty:
                    prediction_frames.append(preds)
                if not calibration.empty:
                    calibration_frames.append(calibration)
                if not stability.empty:
                    stability_frames.append(stability)
            odds = logistic_odds_ratios(protein_df, protein, feature_set, feature_cols)
            if not odds.empty:
                odds_ratio_frames.append(odds)

    pd.DataFrame(metric_records).to_csv(OUT_DIR / "per_protein_feature_set_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        OUT_DIR / "per_protein_oof_predictions.csv", index=False
    )
    pd.concat(calibration_frames, ignore_index=True).to_csv(
        OUT_DIR / "per_protein_calibration_curve_points.csv", index=False
    )
    pd.concat(stability_frames, ignore_index=True).to_csv(
        OUT_DIR / "per_protein_ranking_stability.csv", index=False
    )
    pd.concat(odds_ratio_frames, ignore_index=True).to_csv(
        OUT_DIR / "per_protein_logistic_odds_ratios.csv", index=False
    )


def species_priority_table(df: pd.DataFrame) -> pd.DataFrame:
    baselines = (
        df.groupby("protein")[TARGET]
        .agg(total_rows="size", total_retained="sum")
        .reset_index()
    )
    baselines["p0_protein"] = baselines["total_retained"] / baselines["total_rows"]

    grouped = (
        df.groupby(["protein", "species_name", "species_class"], as_index=False)
        .agg(
            n_condition=(TARGET, "size"),
            n_retained=(TARGET, "sum"),
            median_pH_retained=("p_h", lambda x: x[df.loc[x.index, TARGET] == 1].median()),
            median_concentration_bin=("concentration_bin", median_concentration_bin),
        )
    )
    grouped = grouped.merge(baselines[["protein", "p0_protein"]], on="protein", how="left")
    grouped["raw_retention_efficiency"] = grouped["n_retained"] / grouped["n_condition"]
    grouped["P_shrink"] = (grouped["n_retained"] + ALPHA * grouped["p0_protein"]) / (
        grouped["n_condition"] + ALPHA
    )
    grouped["support_flag"] = grouped["n_condition"].apply(support_flag)
    grouped = grouped[
        [
            "protein",
            "species_name",
            "species_class",
            "n_condition",
            "n_retained",
            "raw_retention_efficiency",
            "p0_protein",
            "P_shrink",
            "support_flag",
            "median_pH_retained",
            "median_concentration_bin",
        ]
    ]
    return grouped.sort_values(["protein", "P_shrink", "n_condition"], ascending=[True, False, False])


def class_level_retention(df: pd.DataFrame) -> pd.DataFrame:
    baselines = (
        df.groupby("protein")[TARGET]
        .agg(total_rows="size", total_retained="sum")
        .reset_index()
    )
    baselines["p0_protein"] = baselines["total_retained"] / baselines["total_rows"]
    grouped = (
        df.groupby(["protein", "species_class"], as_index=False)
        .agg(n_condition=(TARGET, "size"), n_retained=(TARGET, "sum"))
    )
    grouped = grouped.merge(baselines[["protein", "p0_protein"]], on="protein", how="left")
    grouped["raw_retention_efficiency"] = grouped["n_retained"] / grouped["n_condition"]
    grouped["P_shrink"] = (grouped["n_retained"] + ALPHA * grouped["p0_protein"]) / (
        grouped["n_condition"] + ALPHA
    )
    grouped["support_flag"] = grouped["n_condition"].apply(support_flag)
    grouped["species_class"] = pd.Categorical(grouped["species_class"], CLASS_ORDER, ordered=True)
    return grouped[
        [
            "protein",
            "species_class",
            "n_condition",
            "n_retained",
            "raw_retention_efficiency",
            "P_shrink",
            "support_flag",
        ]
    ].sort_values(["protein", "species_class"])


def top_species_by_protein(priority: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protein, subset in priority.groupby("protein", sort=True):
        candidate = subset[subset["support_flag"] != "exploratory"].copy()
        candidate = candidate.sort_values(["P_shrink", "n_condition"], ascending=[False, False])
        candidate["rank"] = range(1, len(candidate) + 1)
        eligible = candidate[
            (candidate["n_condition"] >= 10)
            | candidate.apply(
                lambda row: (row["protein"], row["species_name"]) in CURATED_LOW_SUPPORT_MAIN_FIGURE,
                axis=1,
            )
        ].copy()
        recommended_species = set(eligible.head(4)["species_name"])
        top = candidate.head(12).copy()
        top["recommended_for_main_figure"] = top["species_name"].isin(recommended_species)
        rows.append(top)
    out = pd.concat(rows, ignore_index=True)
    return out[
        [
            "protein",
            "species_name",
            "species_class",
            "rank",
            "n_condition",
            "n_retained",
            "raw_retention_efficiency",
            "P_shrink",
            "support_flag",
            "recommended_for_main_figure",
        ]
    ]


def case_bin_support_flag(n_bin: int) -> str:
    if n_bin == 0:
        return "NA"
    if n_bin < 3:
        return "low_support"
    return "supported"


def case_level_bin_ranking(
    df: pd.DataFrame,
    bin_col: str,
    bin_labels: list[str],
    shrink_col: str,
    rank_col: str,
) -> pd.DataFrame:
    records = []
    for protein, species_name in CASE_SYSTEMS:
        case_df = df[(df["protein"] == protein) & (df["species_name"] == species_name)].copy()
        if case_df.empty:
            for label in bin_labels:
                records.append(
                    {
                        "protein": protein,
                        "species_name": species_name,
                        bin_col: label,
                        "n_bin": 0,
                        "r_bin": 0,
                        "raw_retention_efficiency": np.nan,
                        shrink_col: np.nan,
                        "support_flag": "NA",
                    }
                )
            continue

        p0_case = case_df[TARGET].mean()
        counts = (
            case_df.groupby(bin_col)[TARGET]
            .agg(n_bin="size", r_bin="sum")
            .reindex(bin_labels, fill_value=0)
            .reset_index()
        )
        counts["protein"] = protein
        counts["species_name"] = species_name
        counts["raw_retention_efficiency"] = np.where(
            counts["n_bin"] > 0,
            counts["r_bin"] / counts["n_bin"],
            np.nan,
        )
        counts[shrink_col] = np.where(
            counts["n_bin"] > 0,
            (counts["r_bin"] + BETA * p0_case) / (counts["n_bin"] + BETA),
            np.nan,
        )
        counts["support_flag"] = counts["n_bin"].apply(case_bin_support_flag)
        records.extend(
            counts[
                [
                    "protein",
                    "species_name",
                    bin_col,
                    "n_bin",
                    "r_bin",
                    "raw_retention_efficiency",
                    shrink_col,
                    "support_flag",
                ]
            ].to_dict("records")
        )

    out = pd.DataFrame(records)
    out[rank_col] = np.nan
    for _, case_idx in out.groupby(["protein", "species_name"]).groups.items():
        case_part = out.loc[case_idx].copy()
        ranked = case_part[
            case_part[shrink_col].notna() & case_part[bin_col].ne("missing")
        ].sort_values(
            [shrink_col, "n_bin"], ascending=[False, False]
        )
        out.loc[ranked.index, rank_col] = range(1, len(ranked) + 1)
    out[rank_col] = out[rank_col].astype("Int64")
    return out


def structural_interpretation_label(
    species_name: str,
    species_class: str,
    n_retained: int,
    n_condition_only: int,
    delta_solvent: float,
    delta_resolution: float,
) -> str:
    if n_retained < 3 or n_condition_only < 3:
        return "low-support"
    if species_class == "Metal/cations":
        return "metal-stabilized"
    if species_class == "Anions":
        return "anion-sensitive"
    if species_class == "Neutral solvents/polyols":
        return "solvent/polyol-associated"
    if pd.notna(delta_solvent) and abs(delta_solvent) >= 5:
        return "packing-associated"
    if pd.notna(delta_resolution) and abs(delta_resolution) >= 0.3:
        return "resolution-associated"
    return "packing-associated"


def structural_consequence_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for protein, species_name in STRUCTURAL_CASE_SYSTEMS:
        case_df = df[(df["protein"] == protein) & (df["species_name"] == species_name)].copy()
        if case_df.empty:
            continue
        retained = case_df[case_df[TARGET] == 1]
        condition_only = case_df[case_df[TARGET] == 0]
        species_class = str(case_df["species_class"].mode().iloc[0])

        median_solvent_retained = retained["solvent_content_percent"].median()
        median_solvent_condition_only = condition_only["solvent_content_percent"].median()
        delta_solvent = median_solvent_retained - median_solvent_condition_only

        median_resolution_retained = retained["resolution_high_a"].median()
        median_resolution_condition_only = condition_only["resolution_high_a"].median()
        delta_resolution = median_resolution_retained - median_resolution_condition_only

        records.append(
            {
                "protein": protein,
                "species_name": species_name,
                "n_retained": int(len(retained)),
                "n_condition_only": int(len(condition_only)),
                "median_solvent_retained": median_solvent_retained,
                "median_solvent_condition_only": median_solvent_condition_only,
                "delta_solvent": delta_solvent,
                "median_resolution_retained": median_resolution_retained,
                "median_resolution_condition_only": median_resolution_condition_only,
                "delta_resolution": delta_resolution,
                "interpretation_label": structural_interpretation_label(
                    species_name,
                    species_class,
                    int(len(retained)),
                    int(len(condition_only)),
                    delta_solvent,
                    delta_resolution,
                ),
            }
        )
    return pd.DataFrame(records)


def write_summary(
    df: pd.DataFrame,
    priority: pd.DataFrame,
    top_species: pd.DataFrame,
    concentration_ranking: pd.DataFrame,
    ph_ranking: pd.DataFrame,
    structural_summary: pd.DataFrame,
) -> None:
    summary = {
        "source_dataset": str(DATA),
        "alpha": ALPHA,
        "n_rows": int(len(df)),
        "proteins": df["protein"].value_counts().sort_index().astype(int).to_dict(),
        "outputs": {
            "per_protein_tables": str(PER_PROTEIN_DIR),
            "ml_metrics": str(OUT_DIR),
            "fig4_species_priority_table": str(FIGURE_ASSET_DIR / "fig4_species_priority_table.csv"),
            "fig4_class_level_retention": str(FIGURE_ASSET_DIR / "fig4_class_level_retention.csv"),
            "fig4_top_species_by_protein": str(FIGURE_ASSET_DIR / "fig4_top_species_by_protein.csv"),
            "fig4_concentration_bin_ranking": str(FIGURE_ASSET_DIR / "fig4_concentration_bin_ranking.csv"),
            "fig4_pH_window_ranking": str(FIGURE_ASSET_DIR / "fig4_pH_window_ranking.csv"),
            "fig4_structural_consequence_summary": str(FIGURE_ASSET_DIR / "fig4_structural_consequence_summary.csv"),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Per-Protein Species Retention Fig. 4 v1",
        "",
        f"- source: `{DATA}`",
        f"- rows: `{len(df)}`",
        f"- shrinkage alpha: `{ALPHA:g}`",
        "- main model: regularized logistic regression",
        "- sensitivity model: random forest",
        "- structural outcomes excluded from ML inputs: `solvent_content_percent`, `matthews_density`, `resolution_high_a`",
        "- Proteinase K is flagged as the small-sample hybrid case; standalone AUC metrics are not reported for it.",
        f"- concentration/pH bin shrinkage beta: `{BETA:g}`",
        "- structural consequence overlay is post-hoc only; solvent content and resolution are not ML targets.",
        "",
        "## Recommended main-figure species",
        "",
    ]
    main = top_species[top_species["recommended_for_main_figure"]].copy()
    for protein, subset in main.groupby("protein", sort=True):
        labels = []
        for _, row in subset.sort_values("rank").iterrows():
            suffix = " [low-support]" if row["support_flag"] == "low_support" else ""
            labels.append(f"{row['species_name']}{suffix} ({row['P_shrink']:.3f}, n={int(row['n_condition'])})")
        lines.append(f"- {protein}: " + "; ".join(labels))

    lines.extend(["", "## Preferred concentration windows", ""])
    preferred_conc = concentration_ranking[concentration_ranking["preferred_bin_rank"].eq(1)]
    for _, row in preferred_conc.sort_values(["protein", "species_name"]).iterrows():
        suffix = " [low-support]" if row["support_flag"] == "low_support" else ""
        lines.append(
            f"- {row['protein']}--{row['species_name']}: {row['concentration_bin']}{suffix} "
            f"({row['P_bin_shrink']:.3f}, n={int(row['n_bin'])})"
        )

    lines.extend(["", "## Preferred pH windows", ""])
    preferred_ph = ph_ranking[ph_ranking["preferred_pH_rank"].eq(1)]
    for _, row in preferred_ph.sort_values(["protein", "species_name"]).iterrows():
        suffix = " [low-support]" if row["support_flag"] == "low_support" else ""
        lines.append(
            f"- {row['protein']}--{row['species_name']}: {row['pH_bin']}{suffix} "
            f"({row['P_pH_shrink']:.3f}, n={int(row['n_bin'])})"
        )

    lines.extend(["", "## Structural overlay note", ""])
    for _, row in structural_summary.sort_values(["protein", "species_name"]).iterrows():
        lines.append(
            f"- {row['protein']}--{row['species_name']}: {row['interpretation_label']}; "
            f"delta solvent={row['delta_solvent']:.2f}, delta resolution={row['delta_resolution']:.2f} A "
            "(resolution lower is better)."
        )

    lines.extend(
        [
            "",
            "## Output files",
            "",
            f"- `{PER_PROTEIN_DIR / 'model_input_table.csv'}`",
            f"- `{OUT_DIR / 'per_protein_feature_set_metrics.csv'}`",
            f"- `{OUT_DIR / 'per_protein_logistic_odds_ratios.csv'}`",
            f"- `{OUT_DIR / 'per_protein_calibration_curve_points.csv'}`",
            f"- `{OUT_DIR / 'per_protein_ranking_stability.csv'}`",
            f"- `{FIGURE_ASSET_DIR / 'fig4_species_priority_table.csv'}`",
            f"- `{FIGURE_ASSET_DIR / 'fig4_class_level_retention.csv'}`",
            f"- `{FIGURE_ASSET_DIR / 'fig4_top_species_by_protein.csv'}`",
            f"- `{FIGURE_ASSET_DIR / 'fig4_concentration_bin_ranking.csv'}`",
            f"- `{FIGURE_ASSET_DIR / 'fig4_pH_window_ranking.csv'}`",
            f"- `{FIGURE_ASSET_DIR / 'fig4_structural_consequence_summary.csv'}`",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PER_PROTEIN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_ASSET_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA)
    df = prepare_table(raw)
    write_per_protein_tables(df)
    run_feature_set_ablation(df)

    priority = species_priority_table(df)
    class_retention = class_level_retention(df)
    top_species = top_species_by_protein(priority)
    concentration_ranking = case_level_bin_ranking(
        df,
        "concentration_bin",
        CONCENTRATION_BIN_LABELS,
        "P_bin_shrink",
        "preferred_bin_rank",
    )
    ph_ranking = case_level_bin_ranking(
        df,
        "pH_bin",
        PH_BIN_LABELS,
        "P_pH_shrink",
        "preferred_pH_rank",
    )
    structural_summary = structural_consequence_summary(df)

    priority.to_csv(FIGURE_ASSET_DIR / "fig4_species_priority_table.csv", index=False)
    class_retention.to_csv(FIGURE_ASSET_DIR / "fig4_class_level_retention.csv", index=False)
    top_species.to_csv(FIGURE_ASSET_DIR / "fig4_top_species_by_protein.csv", index=False)
    concentration_ranking.to_csv(FIGURE_ASSET_DIR / "fig4_concentration_bin_ranking.csv", index=False)
    ph_ranking.to_csv(FIGURE_ASSET_DIR / "fig4_pH_window_ranking.csv", index=False)
    structural_summary.to_csv(FIGURE_ASSET_DIR / "fig4_structural_consequence_summary.csv", index=False)
    write_summary(df, priority, top_species, concentration_ranking, ph_ranking, structural_summary)


if __name__ == "__main__":
    main()

