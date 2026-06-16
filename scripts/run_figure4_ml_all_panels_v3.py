from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wallis.figures.style import PALETTE, PROTEIN_LABELS, PROTEIN_ORDER  # noqa: E402


DATA = ROOT / "data" / "ml" / "species_aware_screening_ml_v2.csv"
OUT = ROOT / "results" / "ml" / "figure4_ml_v3"
FIG = ROOT / "results" / "figures" / "figure4_ml_v3"
PANEL_A_IMAGE = FIG / "a.png"
TARGET = "observed_in_structure_label"
RANDOM_STATE = 42
C_GRID = [0.01, 0.1, 1.0, 10.0]
MAIN_FIGURE_MIN_SUPPORT = 10
FONT_SIZE_DELTA = 2.0
AXIS_TEXT_SIZE_PT = 7.0
D_HEATMAP_PROB_TEXT_SIZE_PT = 7.2 + FONT_SIZE_DELTA
D_HEATMAP_N_TEXT_SIZE_PT = 4.9
PANEL_E_WIDTH_MULTIPLIER = 1.3
TOP_PANEL_WSPACE = 0.7246
BOTTOM_PANEL_WSPACE = 0.7246
ROW_HSPACE = 0.2456
PANEL_C_PROTEIN_GAP_MULTIPLIER = {
    "ribonuclease": 2.0,
}
PANEL_C_PROTEIN_X_REFERENCE_SPECIES = {
    "ribonuclease": "magnesium",
}

CLASS_ORDER = [
    "Anions",
    "Metal/cations",
    "Small neutral additives",
    "Organic ligands",
    "PEG/polymer",
    "IL/DES",
]
PANEL_D_CLASS_ORDER = [value for value in CLASS_ORDER if value != "IL/DES"]
CLASS_SHORT = {
    "Anions": "Anions",
    "Metal/cations": "Metals",
    "Small neutral additives": "Neutral\nadditives",
    "Organic ligands": "Ligands",
    "PEG/polymer": "PEG",
    "IL/DES": "IL/DES",
}
CLASS_COLORS = {
    "Anions": PALETTE["blue"],
    "Metal/cations": PALETTE["gold"],
    "Small neutral additives": PALETTE["teal"],
    "Organic ligands": PALETTE["purple"],
    "PEG/polymer": "#8D8D8D",
    "IL/DES": "#B8627A",
}
ML_HEATMAP_CMAP = sns.light_palette(PALETTE["purple"], as_cmap=True)

FEATURE_SETS = {
    "Species class only": ["species_class"],
    "Species identity only": ["species_name"],
    "Condition context only": [
        "pH_bin",
        "temperature_bin",
        "method_family",
        "concentration_context",
        "concentration_unit_family_clean",
        "has_numeric_concentration_label",
    ],
    "Species + class": ["species_name", "species_class"],
    "Full model": [
        "species_name",
        "species_class",
        "pH_bin",
        "temperature_bin",
        "method_family",
        "concentration_context",
        "concentration_unit_family_clean",
        "has_numeric_concentration_label",
    ],
}
FEATURE_GROUPS = {
    "species_name": "Species identity",
    "species_class": "Species class",
    "pH_bin": "pH",
    "temperature_bin": "Temperature",
    "method_family": "Method",
    "concentration_context": "Concentration context",
    "concentration_unit_family_clean": "Concentration context",
    "has_numeric_concentration_label": "Concentration context",
}
FEATURE_GROUP_ORDER = [
    "Species identity",
    "Species class",
    "pH",
    "Temperature",
    "Method",
    "Concentration context",
]

ANIONS = {"nitrate", "sulfate", "chloride", "acetate", "citrate", "phosphate", "formate", "iodide", "bromide", "tartrate", "malate"}
METALS = {"calcium", "magnesium", "zinc", "sodium", "potassium", "manganese", "cobalt", "copper", "ammonium", "lithium", "cadmium", "nickel", "cesium"}
NEUTRALS = {"glycerol", "ethylene glycol", "mpd", "2-propanol", "ethanol", "methanol", "acetone", "dioxane", "dmf"}
LIGANDS = {"phenol", "cresol", "imidazole", "tris", "hepes", "mes", "bis-tris", "cacodylate", "dtt", "chaps"}

PDB_EXAMPLE_ROWS = {
    "lysozyme": {
        "primary_pdb": "2VB1",
        "visible_species": "NO3- x9; acetate x1; ethylene glycol x3",
        "backup_pdb": "7P6M",
        "backup_visible_species": "NO3- x8; acetate x2",
        "why": "Ultra-high-resolution HEWL structure (0.65 A) showing modeled anions and a neutral additive in one structure.",
    },
    "insulin": {
        "primary_pdb": "5HRQ",
        "visible_species": "Zn2+ x2; phenol x6; Cl- x2",
        "backup_pdb": "4P65",
        "backup_visible_species": "Zn2+ x2; phenol x6; Cl- x2",
        "why": "Clear insulin example linking structural Zn2+ with phenol-stabilized R6 hexamer chemistry (1.28 A).",
    },
    "trypsin": {
        "primary_pdb": "4I8H",
        "visible_species": "Ca2+ x1; sulfate x2; glycerol x1",
        "backup_pdb": "4ABE",
        "backup_visible_species": "Ca2+ x1; Cl- x1; sulfate x2; ethylene glycol x5",
        "why": "High-resolution bovine trypsin structure (0.75 A) with Ca2+ plus crystallization additives visible.",
    },
    "proteinase k": {
        "primary_pdb": "7LTD",
        "visible_species": "Ca2+ x1; nitrate x5",
        "backup_pdb": "7LN7",
        "backup_visible_species": "Ca2+ x1; sulfate x3",
        "why": "Strong proteinase K example for structural Ca2+ together with an ion-focused nitrate signal (0.90 A).",
    },
    "ribonuclease": {
        "primary_pdb": "2ETJ",
        "visible_species": "Mg2+ x1; Cl- x1; ethylene glycol x5",
        "backup_pdb": "3ULD",
        "backup_visible_species": "Mg2+ x4",
        "why": "Ribonuclease HII example that shows Mg2+ and ethylene glycol, complementing the RNase ML Mg signal (1.74 A).",
    },
}


def pretty_protein(protein: str) -> str:
    return PROTEIN_LABELS.get(protein, protein.title())


def pretty_species(name: str) -> str:
    mapping = {"mpd": "MPD", "peg": "PEG"}
    return mapping.get(str(name), str(name).title())


def panel_c_protein_label(protein: str) -> str:
    mapping = {
        "ribonuclease": "RNase",
        "proteinase k": "Prot. K",
    }
    return mapping.get(protein, pretty_protein(protein))


def wrap_items(value: str, max_items: int = 3) -> str:
    items = [pretty_species(x.strip()) for x in str(value).split(",") if x.strip()]
    return "\n".join(items[:max_items])


def compact_class_pair(value: str) -> str:
    parts = [x.strip() for x in str(value).split("+") if x.strip()]
    short = [CLASS_SHORT.get(x, x.replace(" additives", "").replace("/cations", "")) for x in parts]
    return "\n+ ".join(short[:2])


def classify_species(row: pd.Series) -> str:
    name = str(row["species_name"]).strip().lower()
    major = str(row.get("species_major_class", "")).strip().lower()
    role = str(row.get("species_role_class", "")).strip().lower()
    if bool(row.get("is_ionic_liquid_or_des", 0)) or "ionic" in major:
        return "IL/DES"
    if name in ANIONS or bool(row.get("is_anion", 0)):
        return "Anions"
    if name in METALS or bool(row.get("is_cation", 0)):
        return "Metal/cations"
    if name == "peg" or bool(row.get("is_polymeric_precipitant", 0)):
        return "PEG/polymer"
    if name in NEUTRALS or bool(row.get("is_solvent_like", 0)) or bool(row.get("is_polyol_like", 0)):
        return "Small neutral additives"
    if name in LIGANDS or "buffer" in major or "buffer" in role:
        return "Organic ligands"
    return "Organic ligands"


def pH_bin(value: object) -> str:
    if pd.isna(value):
        return "missing"
    value = float(value)
    if value < 5.5:
        return "acidic"
    if value < 6.5:
        return "mildly acidic"
    if value < 7.5:
        return "neutral"
    if value < 8.5:
        return "mildly basic"
    return "basic"


def temperature_bin(value: object) -> str:
    if pd.isna(value):
        return "missing"
    value = float(value)
    if value < 285:
        return "cold"
    if value <= 298:
        return "ambient"
    return "warm"


def concentration_context(value: object) -> str:
    if pd.isna(value):
        return "missing"
    mapping = {
        "trace_to_low": "trace-low",
        "low_to_mid": "low-mid",
        "mid_to_high": "mid-high",
        "very_high": "very high",
        "nonmolar_numeric": "non-molar numeric",
    }
    return mapping.get(str(value), str(value))


def prepare_table() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df = df[df["protein"].isin(PROTEIN_ORDER)].copy()
    df["protein"] = df["protein"].astype(str).str.lower()
    df["species_name"] = df["species_name"].astype(str).str.lower()
    df["species_class"] = df.apply(classify_species, axis=1)
    df["group_id"] = df["protein"].astype(str) + "|" + df["pdb_id"].astype(str) + "|" + df["crystal_id"].astype(str)
    df["per_protein_group_id"] = df["pdb_id"].astype(str) + "|" + df["crystal_id"].astype(str)
    df["pH_bin"] = df["p_h"].apply(pH_bin)
    df["temperature_bin"] = df["temp_k"].apply(temperature_bin)
    df["concentration_context"] = df["concentration_bin"].apply(concentration_context)
    df["concentration_unit_family_clean"] = df["concentration_unit_family"].fillna("missing").astype(str)
    df["has_numeric_concentration_label"] = np.where(df["has_numeric_concentration"].astype(int).eq(1), "yes", "no")
    df["method_family"] = df["method_family"].fillna("missing").astype(str)
    df[TARGET] = df[TARGET].astype(int)
    return df


def make_pipeline(feature_cols: list[str], c_value: float) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", pre), ("model", clf)])


def grouped_oof(df: pd.DataFrame, feature_cols: list[str], c_value: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[TARGET].astype(int)
    groups = df["per_protein_group_id"].astype(str)
    n_splits = min(5, groups.nunique(), int(y.value_counts().min()))
    if n_splits < 2 or y.nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    pred_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(df[feature_cols], y, groups), start=1):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        pipe = make_pipeline(feature_cols, c_value)
        pipe.fit(df.iloc[train_idx][feature_cols], y_train)
        prob = pipe.predict_proba(df.iloc[test_idx][feature_cols])[:, 1]
        fold_rows.append(
            {
                "fold_id": fold,
                "roc_auc": roc_auc_score(y_test, prob),
                "pr_auc": average_precision_score(y_test, prob),
                "brier_score": brier_score_loss(y_test, prob),
                "n_test": int(len(test_idx)),
            }
        )
        pred = df.iloc[test_idx][
            [
                "protein",
                "pdb_id",
                "crystal_id",
                "species_name",
                "species_class",
                "pH_bin",
                "temperature_bin",
                "method_family",
                "concentration_context",
                "group_id",
                TARGET,
            ]
        ].copy()
        pred["P_oof_logistic"] = prob
        pred["P_oof_random_forest_optional"] = np.nan
        pred["fold_id"] = fold
        pred_rows.extend(pred.to_dict("records"))
    return pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def validate_feature_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    validation_rows = []
    full_oof_frames = []
    selected_c_by_protein = {}
    for protein in PROTEIN_ORDER:
        protein_df = df[df["protein"] == protein].copy()
        best_full_score = -np.inf
        for feature_set, feature_cols in FEATURE_SETS.items():
            c_records = []
            pred_by_c = {}
            for c_value in C_GRID:
                folds, preds = grouped_oof(protein_df, feature_cols, c_value)
                if folds.empty:
                    continue
                pred_by_c[c_value] = preds
                c_records.append(
                    {
                        "C": c_value,
                        "roc_auc_mean": folds["roc_auc"].mean(),
                        "roc_auc_sd": folds["roc_auc"].std(ddof=0),
                        "pr_auc_mean": folds["pr_auc"].mean(),
                        "pr_auc_sd": folds["pr_auc"].std(ddof=0),
                        "brier_mean": folds["brier_score"].mean(),
                        "brier_sd": folds["brier_score"].std(ddof=0),
                    }
                )
            if not c_records:
                for metric in ["ROC-AUC", "PR-AUC", "Brier score"]:
                    validation_rows.append(
                        {
                            "protein": protein,
                            "feature_set": feature_set,
                            "metric": metric,
                            "mean_score": np.nan,
                            "std_score": np.nan,
                            "n_groups": protein_df["per_protein_group_id"].nunique(),
                            "n_positive": int(protein_df[TARGET].sum()),
                            "n_negative": int((1 - protein_df[TARGET]).sum()),
                            "model": "regularized_logistic",
                            "selected_C": np.nan,
                            "status": "not_estimable",
                        }
                    )
                continue
            c_df = pd.DataFrame(c_records).sort_values(["roc_auc_mean", "pr_auc_mean"], ascending=False)
            best = c_df.iloc[0]
            selected_c = float(best["C"])
            for metric, mean_col, sd_col in [
                ("ROC-AUC", "roc_auc_mean", "roc_auc_sd"),
                ("PR-AUC", "pr_auc_mean", "pr_auc_sd"),
                ("Brier score", "brier_mean", "brier_sd"),
            ]:
                validation_rows.append(
                    {
                        "protein": protein,
                        "feature_set": feature_set,
                        "metric": metric,
                        "mean_score": float(best[mean_col]),
                        "std_score": float(best[sd_col]),
                        "n_groups": protein_df["per_protein_group_id"].nunique(),
                        "n_positive": int(protein_df[TARGET].sum()),
                        "n_negative": int((1 - protein_df[TARGET]).sum()),
                        "model": "regularized_logistic",
                        "selected_C": selected_c,
                        "status": "grouped_oof",
                    }
                )
            if feature_set == "Full model":
                selected_c_by_protein[protein] = selected_c
                if float(best["roc_auc_mean"]) > best_full_score:
                    best_full_score = float(best["roc_auc_mean"])
                    full_preds = pred_by_c[selected_c].copy()
                    full_preds["selected_C"] = selected_c
                    full_oof_frames.append(full_preds)
    oof = pd.concat(full_oof_frames, ignore_index=True) if full_oof_frames else pd.DataFrame()
    return pd.DataFrame(validation_rows), oof, selected_c_by_protein


def coefficient_contributions(df: pd.DataFrame, selected_c: dict[str, float]) -> pd.DataFrame:
    rows = []
    feature_cols = FEATURE_SETS["Full model"]
    for protein in PROTEIN_ORDER:
        protein_df = df[df["protein"] == protein].copy()
        if protein not in selected_c or protein_df[TARGET].nunique() < 2:
            continue
        pipe = make_pipeline(feature_cols, selected_c[protein])
        pipe.fit(protein_df[feature_cols], protein_df[TARGET])
        names = pipe.named_steps["preprocess"].get_feature_names_out()
        coefs = np.abs(pipe.named_steps["model"].coef_.ravel())
        contrib = {group: 0.0 for group in FEATURE_GROUP_ORDER}
        for name, coef in zip(names, coefs):
            clean = name.replace("cat__", "", 1)
            source = next((col for col in feature_cols if clean.startswith(col + "_")), None)
            if source is None:
                continue
            contrib[FEATURE_GROUPS[source]] += float(coef)
        total = sum(contrib.values())
        for group in FEATURE_GROUP_ORDER:
            value = contrib[group]
            rows.append(
                {
                    "protein": protein,
                    "feature_group": group,
                    "importance_value": value,
                    "importance_type": "sum_abs_logistic_coefficients",
                    "normalized_importance": value / total if total else np.nan,
                    "model": "regularized_logistic",
                }
            )
    return pd.DataFrame(rows)


def species_priority(oof: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        oof.groupby(["protein", "species_name", "species_class"], as_index=False)
        .agg(
            mean_P_oof=("P_oof_logistic", "mean"),
            median_P_oof=("P_oof_logistic", "median"),
            n_condition=(TARGET, "size"),
            n_modeled=(TARGET, "sum"),
        )
    )
    grouped["raw_structure_observed_frequency"] = grouped["n_modeled"] / grouped["n_condition"]
    grouped["low_support_flag"] = grouped["n_condition"] < 10
    grouped = grouped.sort_values(["protein", "mean_P_oof", "n_condition"], ascending=[True, False, False])
    grouped["rank_within_protein"] = grouped.groupby("protein").cumcount() + 1
    return grouped


def class_signal(oof: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        oof.groupby(["protein", "species_class"], as_index=False)
        .agg(
            mean_P_oof=("P_oof_logistic", "mean"),
            n_condition=(TARGET, "size"),
            n_modeled=(TARGET, "sum"),
            number_of_species=("species_name", "nunique"),
        )
    )
    grouped["low_support_flag"] = grouped["n_condition"] < 10
    grouped["species_class"] = pd.Categorical(grouped["species_class"], CLASS_ORDER, ordered=True)
    return grouped.sort_values(["protein", "species_class"])


def screening_guide(species: pd.DataFrame, classes: pd.DataFrame, contrib: pd.DataFrame) -> pd.DataFrame:
    context_notes = {
        "lysozyme": "pH/method secondary",
        "ribonuclease": "iodide low-support",
        "trypsin": "Ca signal",
        "insulin": "Zn/phenol signal",
        "proteinase k": "support caution",
    }
    rows = []
    for protein in PROTEIN_ORDER:
        top = species[(species["protein"] == protein) & (species["n_condition"] >= MAIN_FIGURE_MIN_SUPPORT)].head(3)
        dominant_classes = []
        for value in top["species_class"].astype(str):
            if value not in dominant_classes:
                dominant_classes.append(value)
        confidence = "high"
        if protein in {"ribonuclease", "insulin"}:
            confidence = "moderate"
        if protein == "proteinase k":
            confidence = "cautious"
        rows.append(
            {
                "protein": protein,
                "top_ml_prioritized_species": ", ".join(top["species_name"].tolist()),
                "dominant_ml_chemical_class": " + ".join(dominant_classes[:2]),
                "condition_context_note": context_notes.get(protein, "context secondary"),
                "confidence_level": confidence,
                "caution_note": "first-pass ML-guided prioritization only; not crystallization-success prediction",
            }
        )
    return pd.DataFrame(rows)


def write_tables(df: pd.DataFrame, validation: pd.DataFrame, oof: pd.DataFrame, species: pd.DataFrame, classes: pd.DataFrame, contrib: pd.DataFrame, guide: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    oof_out = oof.rename(columns={TARGET: "Y_true"})[
        [
            "protein",
            "pdb_id",
            "crystal_id",
            "species_name",
            "species_class",
            "pH_bin",
            "temperature_bin",
            "method_family",
            "concentration_context",
            "Y_true",
            "P_oof_logistic",
            "P_oof_random_forest_optional",
            "fold_id",
            "group_id",
        ]
    ]
    oof_out.to_csv(OUT / "fig4_oof_predictions.csv", index=False)
    validation.to_csv(OUT / "fig4_feature_set_validation.csv", index=False)
    species.to_csv(OUT / "fig4_species_priority_ml.csv", index=False)
    classes.to_csv(OUT / "fig4_class_signal_ml.csv", index=False)
    contrib.to_csv(OUT / "fig4_feature_contribution_ml.csv", index=False)
    guide.to_csv(OUT / "fig4_ml_screening_guide.csv", index=False)
    df.to_csv(OUT / "fig4_model_input_table_internal.csv", index=False)
    summary = {
        "source": str(DATA),
        "rows": int(len(df)),
        "target": "structure-modeled species label",
        "main_values": "out-of-fold regularized logistic P(Y=1)",
        "grouping": "per protein: pdb_id|crystal_id",
        "note": "Raw frequency and support are transparency overlays only, never the main bar/color value.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def apply_style(variant: str) -> None:
    sns.set_theme(style="white", context="paper")
    base = (6.9 if variant == "nature" else 7.2) + FONT_SIZE_DELTA
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": base,
            "axes.labelsize": base,
            "axes.titlesize": base + 0.7,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.65,
            "xtick.labelsize": base - 0.5,
            "ytick.labelsize": base - 0.5,
            "legend.fontsize": base - 0.8,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.0 + FONT_SIZE_DELTA, fontweight="bold", va="top", ha="left")


def crop_near_white_border(img: np.ndarray, threshold: float = 0.985, pad: int = 18) -> np.ndarray:
    rgb = img[..., :3]
    if rgb.dtype.kind in {"u", "i"}:
        rgb = rgb.astype(float) / 255.0
    mask = np.any(rgb < threshold, axis=2)
    if not np.any(mask):
        return img
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad + 1)
    return img[y0:y1, x0:x1]


def draw_a_fallback(ax: plt.Axes) -> None:
    ax.axis("off")
    panel_label(ax, "a")
    steps = [
        "Species-aware rows\nprotein + crystal + species",
        "Input X\nidentity | class | pH | method | concentration",
        "Grouped-CV classifier\nY = structure-modeled species label",
        "Out-of-fold P(Y=1)",
        "Protein-specific ML priority",
    ]
    y0 = 0.86
    for i, text in enumerate(steps):
        y = y0 - i * 0.18
        box = patches.FancyBboxPatch((0.07, y), 0.82, 0.105, boxstyle="round,pad=0.010,rounding_size=0.018", facecolor="#F5F7F8" if i in {0, 4} else "white", edgecolor="#202326", linewidth=0.7)
        ax.add_patch(box)
        ax.text(0.48, y + 0.052, text, ha="center", va="center", fontsize=5.8 + FONT_SIZE_DELTA)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.48, y - 0.018), xytext=(0.48, y - 0.058), arrowprops={"arrowstyle": "-|>", "lw": 0.65})
    ax.text(0.07, 0.02, "Grouped by PDB/crystal ID to avoid leakage.", fontsize=5.6 + FONT_SIZE_DELTA, color="#6B7280")


def draw_a(ax: plt.Axes) -> None:
    ax.axis("off")
    panel_label(ax, "a")
    if not PANEL_A_IMAGE.exists():
        draw_a_fallback(ax)
        return
    img = crop_near_white_border(mpimg.imread(PANEL_A_IMAGE))
    ax.imshow(img, aspect="equal")
    ax.set_anchor("N")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_b(ax: plt.Axes, validation: pd.DataFrame) -> None:
    panel_label(ax, "b")
    roc = validation[validation["metric"] == "ROC-AUC"]
    mat = roc.pivot(index="protein", columns="feature_set", values="mean_score").reindex(PROTEIN_ORDER)
    mat = mat.reindex(columns=list(FEATURE_SETS))
    labels = mat.map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    sns.heatmap(
        mat,
        ax=ax,
        cmap=ML_HEATMAP_CMAP,
        vmin=0.5,
        vmax=1.0,
        annot=labels,
        fmt="",
        linewidths=0.45,
        linecolor="white",
        cbar_kws={"label": "ROC-AUC", "fraction": 0.045, "pad": 0.020},
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.8 + FONT_SIZE_DELTA, length=2.0)
    cbar.set_label("ROC-AUC", fontsize=5.9 + FONT_SIZE_DELTA)
    ax.set_title("Feature-set validation", fontsize=6.2 + FONT_SIZE_DELTA, pad=4)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels([pretty_protein(x) for x in mat.index], rotation=0)
    ax.set_xticklabels(
        [
            "Chemical class",
            "Species identity",
            "Condition context",
            "Identity + class",
            "Full model",
        ],
        rotation=31,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", labelsize=AXIS_TEXT_SIZE_PT, pad=-1)
    ax.tick_params(axis="y", labelsize=AXIS_TEXT_SIZE_PT, pad=1)


def draw_c(ax: plt.Axes, species: pd.DataFrame) -> None:
    panel_label(ax, "c")
    rows = []
    y = 0
    yticks, ylabels, separators = [], [], []
    protein_center_rows = {}
    protein_species_rows = {}
    for protein_i, protein in enumerate(reversed(PROTEIN_ORDER)):
        top = (
            species[(species["protein"] == protein) & (species["n_condition"] >= MAIN_FIGURE_MIN_SUPPORT)]
            .sort_values("mean_P_oof", ascending=False)
            .head(3)
            .sort_values("mean_P_oof", ascending=True)
        )
        protein_rows = []
        for _, row in top.iterrows():
            rows.append((y, row))
            yticks.append(y)
            ylabels.append(pretty_species(row["species_name"]))
            protein_rows.append(y)
            protein_species_rows[(protein, str(row["species_name"]))] = float(y)
            y += 1
        if protein_rows:
            protein_center_rows[protein] = float(protein_rows[len(protein_rows) // 2])
        if protein_i < len(PROTEIN_ORDER) - 1:
            separators.append(y - 0.28)
        y += 0.38
    for yy, row in rows:
        color = CLASS_COLORS.get(row["species_class"], "#8D8D8D")
        ax.barh(yy, row["mean_P_oof"], height=0.56, color=color, alpha=0.84)
        ax.scatter(row["raw_structure_observed_frequency"], yy, s=np.clip(row["n_condition"] * 1.05, 18, 145), color="#0F766E", edgecolor="white", linewidth=0.45, zorder=3)
    for sep in separators:
        ax.axhline(sep, color="#E8EDF2", lw=0.6, zorder=0)
    ax.set_yticks(yticks, ylabels)
    ax.tick_params(axis="x", labelsize=AXIS_TEXT_SIZE_PT)
    ax.tick_params(axis="y", labelsize=AXIS_TEXT_SIZE_PT, pad=0)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Mean out-of-fold P(structure-modeled)", fontsize=AXIS_TEXT_SIZE_PT)
    ax.grid(axis="x", color="#E8EDF2", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    axis_x_px = ax.bbox.x0
    px_to_pt = 72.0 / ax.figure.dpi
    ticklabel_by_row = {tick: label for tick, label in zip(yticks, ax.get_yticklabels())}
    protein_texts = []
    for protein in reversed(PROTEIN_ORDER):
        if protein not in protein_center_rows:
            continue
        row = protein_center_rows[protein]
        reference_species = PANEL_C_PROTEIN_X_REFERENCE_SPECIES.get(protein)
        reference_row = protein_species_rows.get((protein, reference_species), row)
        ticklabel = ticklabel_by_row.get(reference_row)
        center_ticklabel = ticklabel_by_row.get(row)
        if ticklabel is None or center_ticklabel is None:
            continue
        tick_bbox = ticklabel.get_window_extent(renderer=renderer)
        center_tick_bbox = center_ticklabel.get_window_extent(renderer=renderer)
        target_y_px = (center_tick_bbox.y0 + center_tick_bbox.y1) * 0.5
        gap_px = axis_x_px - tick_bbox.x1
        gap_px *= PANEL_C_PROTEIN_GAP_MULTIPLIER.get(protein, 1.0)
        desired_right_px = tick_bbox.x0 - gap_px
        text = ax.annotate(
            panel_c_protein_label(protein),
            xy=(0.0, row),
            xycoords=ax.get_yaxis_transform(),
            xytext=((desired_right_px - axis_x_px) * px_to_pt, 0.0),
            textcoords="offset points",
            fontsize=AXIS_TEXT_SIZE_PT,
            fontstyle="normal",
            fontweight="normal",
            color="#262626",
            rotation=90,
            rotation_mode="anchor",
            ha="right",
            va="center",
            zorder=4,
            annotation_clip=False,
        )
        protein_texts.append((text, desired_right_px, target_y_px))
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    for text, desired_right_px, target_y_px in protein_texts:
        text_bbox = text.get_window_extent(renderer=renderer)
        delta_x_pt = (desired_right_px - text_bbox.x1) * px_to_pt
        delta_y_pt = (target_y_px - ((text_bbox.y0 + text_bbox.y1) * 0.5)) * px_to_pt
        current_x_pt, current_y_pt = text.get_position()
        text.set_position((current_x_pt + delta_x_pt, current_y_pt + delta_y_pt))
    ax.legend(
        handles=[
            Line2D([0], [0], color=PALETTE["blue"], lw=5, label="Bar = ML probability"),
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#0F766E", markeredgecolor="white", markersize=5, label="Dot = raw frequency"),
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#0F766E", markeredgecolor="white", markersize=np.sqrt(np.clip(10 * 1.05, 18, 145)), label="n=10"),
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#0F766E", markeredgecolor="white", markersize=np.sqrt(np.clip(50 * 1.05, 18, 145)), label="n=50"),
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#0F766E", markeredgecolor="white", markersize=np.sqrt(np.clip(200 * 1.05, 18, 145)), label="n=200"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.50, -0.36),
        ncol=3,
        fontsize=4.8 + FONT_SIZE_DELTA,
        columnspacing=0.8,
        handlelength=1.7,
    )


def draw_d(ax: plt.Axes, classes: pd.DataFrame) -> None:
    panel_label(ax, "d")
    mat = classes.pivot(index="protein", columns="species_class", values="mean_P_oof").reindex(PROTEIN_ORDER)
    mat = mat.reindex(columns=PANEL_D_CLASS_ORDER)
    n_records = classes.pivot(index="protein", columns="species_class", values="n_condition").reindex(PROTEIN_ORDER).reindex(columns=PANEL_D_CLASS_ORDER)
    sns.heatmap(
        mat,
        ax=ax,
        cmap=ML_HEATMAP_CMAP,
        vmin=0,
        vmax=1,
        annot=False,
        linewidths=0.45,
        linecolor="white",
        cbar_kws={"label": "Mean OOF probability", "fraction": 0.050, "pad": 0.025},
    )
    for row_i, protein in enumerate(PROTEIN_ORDER):
        for col_i, cls in enumerate(PANEL_D_CLASS_ORDER):
            value = mat.loc[protein, cls]
            if pd.isna(value):
                continue
            ax.text(
                col_i + 0.5,
                row_i + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=D_HEATMAP_PROB_TEXT_SIZE_PT,
                color="#202326",
            )
            ax.text(
                col_i + 0.95,
                row_i + 0.92,
                f"n={int(n_records.loc[protein, cls])}",
                ha="right",
                va="bottom",
                fontsize=D_HEATMAP_N_TEXT_SIZE_PT,
                color="#202326",
            )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.8 + FONT_SIZE_DELTA, length=2.0)
    cbar.set_label("Mean OOF probability", fontsize=5.9 + FONT_SIZE_DELTA)
    ax.set_title("Class-level ML signal", fontsize=6.2 + FONT_SIZE_DELTA, pad=4)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels([pretty_protein(p) for p in PROTEIN_ORDER], rotation=0)
    ax.set_xticklabels([CLASS_SHORT[c] for c in PANEL_D_CLASS_ORDER], rotation=34, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=AXIS_TEXT_SIZE_PT, pad=1)
    ax.tick_params(axis="y", labelsize=AXIS_TEXT_SIZE_PT, pad=1)
    ax.text(0.0, -0.20, "Color = mean OOF P; text = n species/n records", ha="left", va="top", transform=ax.transAxes, fontsize=5.0 + FONT_SIZE_DELTA, color="#6B7280")


def draw_e(ax: plt.Axes, contrib: pd.DataFrame) -> None:
    panel_label(ax, "e")
    mat = contrib.pivot(index="protein", columns="feature_group", values="normalized_importance").reindex(PROTEIN_ORDER).fillna(0)
    stacked = pd.DataFrame(
        {
            "Species identity": mat.get("Species identity", 0),
            "Chemical class": mat.get("Species class", 0),
            "Condition context": mat.get("pH", 0) + mat.get("Temperature", 0) + mat.get("Method", 0),
            "Concentration context": mat.get("Concentration context", 0),
        },
        index=PROTEIN_ORDER,
    )
    colors = {
        "Species identity": PALETTE["purple"],
        "Chemical class": PALETTE["blue"],
        "Condition context": PALETTE["teal"],
        "Concentration context": PALETTE["gold"],
    }
    y = np.arange(len(PROTEIN_ORDER))
    left = np.zeros(len(PROTEIN_ORDER))
    for label in stacked.columns:
        values = stacked[label].to_numpy(dtype=float)
        ax.barh(y, values, left=left, height=0.62, color=colors[label], alpha=0.86, edgecolor="white", linewidth=0.45, label=label)
        for i, value in enumerate(values):
            if label == "Species identity":
                ax.text(left[i] + value / 2, y[i], f"{value:.2f}", ha="center", va="center", fontsize=D_HEATMAP_PROB_TEXT_SIZE_PT, color="white")
        left += values
    ax.set_yticks(y, [pretty_protein(p) for p in PROTEIN_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized coefficient contribution", fontsize=AXIS_TEXT_SIZE_PT)
    ax.tick_params(axis="x", labelsize=AXIS_TEXT_SIZE_PT)
    ax.tick_params(axis="y", labelsize=AXIS_TEXT_SIZE_PT, pad=1)
    ax.grid(axis="x", color="#E8EDF2", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.50, -0.38), ncol=2, fontsize=4.4 + FONT_SIZE_DELTA, columnspacing=0.7, handlelength=1.1)


def draw_f(ax: plt.Axes, guide: pd.DataFrame) -> None:
    ax.axis("off")
    panel_label(ax, "f")
    x = [0.02, 0.24, 0.53, 0.73, 0.90]
    headers = ["Protein", "Top ML species", "ML class", "Note", "Confidence"]
    for xi, h in zip(x, headers):
        ax.text(xi, 0.94, h, fontsize=5.2 + FONT_SIZE_DELTA, fontweight="bold", transform=ax.transAxes)
    y = 0.82
    for _, row in guide.set_index("protein").reindex(PROTEIN_ORDER).reset_index().iterrows():
        species_text = wrap_items(row["top_ml_prioritized_species"])
        class_text = compact_class_pair(row["dominant_ml_chemical_class"])
        context_text = "\n".join(textwrap.wrap(str(row["condition_context_note"]), width=14))
        conf = str(row["confidence_level"]).capitalize()
        protein_label = "\n".join(textwrap.wrap(pretty_protein(row["protein"]), width=13))
        ax.text(x[0], y, protein_label, fontsize=4.9 + FONT_SIZE_DELTA, transform=ax.transAxes, va="top")
        ax.text(x[1], y, species_text, fontsize=4.85 + FONT_SIZE_DELTA, va="top", transform=ax.transAxes)
        ax.text(x[2], y, class_text, fontsize=4.85 + FONT_SIZE_DELTA, va="top", transform=ax.transAxes)
        ax.text(x[3], y, context_text, fontsize=4.75 + FONT_SIZE_DELTA, va="top", transform=ax.transAxes)
        ax.text(x[4], y, conf, fontsize=4.85 + FONT_SIZE_DELTA, transform=ax.transAxes, va="top")
        ax.plot([0.02, 0.985], [y - 0.095, y - 0.095], color="#E8EDF2", lw=0.55, transform=ax.transAxes)
        y -= 0.165


def representative_pdb_table(guide: pd.DataFrame) -> pd.DataFrame:
    guide_lookup = guide.set_index("protein")
    rows = []
    for protein in PROTEIN_ORDER:
        example = PDB_EXAMPLE_ROWS[protein]
        guide_row = guide_lookup.loc[protein]
        rows.append(
            {
                "protein": pretty_protein(protein),
                "ml_prioritized_species": ", ".join(
                    pretty_species(x.strip())
                    for x in str(guide_row["top_ml_prioritized_species"]).split(",")
                    if x.strip()
                ),
                "dominant_ml_class": str(guide_row["dominant_ml_chemical_class"]),
                "primary_pdb": example["primary_pdb"],
                "primary_pdb_url": f"https://www.rcsb.org/structure/{example['primary_pdb']}",
                "structure_visible_species": example["visible_species"],
                "why_main_choice": example["why"],
                "backup_pdb": example["backup_pdb"],
                "backup_pdb_url": f"https://www.rcsb.org/structure/{example['backup_pdb']}",
                "backup_visible_species": example["backup_visible_species"],
            }
        )
    return pd.DataFrame(rows)


def write_representative_pdb_table(table: pd.DataFrame, variant: str) -> None:
    stem = "figure3_representative_pdb_table_" + variant
    table.to_csv(OUT / f"{stem}.csv", index=False)
    md = [
        "# Representative PDB examples for ML-prioritized ion/additive signals",
        "",
        "| Protein | ML-prioritized species | ML class | Primary PDB | Structure-visible species | Why this is the main choice | Backup PDB | Backup visible species |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in table.iterrows():
        md.append(
            "| {protein} | {ml} | {ml_class} | [{pdb}]({url}) | {visible} | {why} | [{backup}]({backup_url}) | {backup_visible} |".format(
                protein=row["protein"],
                ml=row["ml_prioritized_species"],
                ml_class=row["dominant_ml_class"],
                pdb=row["primary_pdb"],
                url=row["primary_pdb_url"],
                visible=row["structure_visible_species"],
                why=row["why_main_choice"],
                backup=row["backup_pdb"],
                backup_url=row["backup_pdb_url"],
                backup_visible=row["backup_visible_species"],
            )
        )
    (FIG / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def draw_representative_pdb_table(table: pd.DataFrame, variant: str) -> None:
    apply_style(variant)
    fig, ax = plt.subplots(figsize=(9.35, 4.25) if variant == "hank" else (9.45, 4.25))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.02,
        0.965,
        "Representative PDB examples for ML-prioritized ion/additive signals",
        fontsize=9.0 + FONT_SIZE_DELTA,
        fontweight="bold",
        va="top",
    )
    columns = [
        ("Protein", 0.02, 0.12),
        ("ML species", 0.14, 0.19),
        ("Primary PDB", 0.35, 0.11),
        ("Structure-visible species", 0.48, 0.20),
        ("Why this example", 0.70, 0.25),
        ("Backup", 0.965, 0.06),
    ]
    header_y = 0.875
    row_top = 0.815
    row_h = 0.142
    ax.add_patch(patches.Rectangle((0.015, header_y - 0.035), 0.97, 0.055, facecolor="#F5F7F8", edgecolor="none"))
    for header, x, _ in columns:
        ax.text(x, header_y, header, fontsize=7.0 + FONT_SIZE_DELTA, fontweight="bold", va="center")
    for row_i, (_, row) in enumerate(table.iterrows()):
        y = row_top - row_i * row_h
        if row_i % 2 == 1:
            ax.add_patch(patches.Rectangle((0.015, y - row_h + 0.014), 0.97, row_h - 0.010, facecolor="#FAFBFC", edgecolor="none"))
        ax.plot([0.015, 0.985], [y - row_h + 0.014, y - row_h + 0.014], color="#E8EDF2", lw=0.55)
        ax.text(0.02, y, row["protein"], fontsize=6.7 + FONT_SIZE_DELTA, fontweight="bold", va="top")
        ax.text(0.14, y, "\n".join(textwrap.wrap(row["ml_prioritized_species"], width=19)), fontsize=6.1 + FONT_SIZE_DELTA, va="top")
        ax.text(0.35, y, row["primary_pdb"], fontsize=6.7 + FONT_SIZE_DELTA, fontweight="bold", va="top", color=PALETTE["blue"])
        ax.text(0.48, y, "\n".join(textwrap.wrap(row["structure_visible_species"], width=28)), fontsize=6.1 + FONT_SIZE_DELTA, va="top")
        ax.text(0.70, y, "\n".join(textwrap.wrap(row["why_main_choice"], width=42)), fontsize=5.9 + FONT_SIZE_DELTA, va="top")
        ax.text(0.965, y, row["backup_pdb"], fontsize=6.3 + FONT_SIZE_DELTA, va="top", ha="right", color=PALETTE["blue"])
    stem = "figure3_representative_pdb_table_" + variant
    fig.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def make_figure(validation: pd.DataFrame, species: pd.DataFrame, classes: pd.DataFrame, contrib: pd.DataFrame, guide: pd.DataFrame, variant: str) -> None:
    apply_style(variant)
    fig = plt.figure(figsize=(9.45, 6.85) if variant == "nature" else (9.3491, 6.90))
    top_widths = [0.98, 1.48, 1.52]
    bottom_widths = [1.9272, 1.02, 2.18]
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=ROW_HSPACE)
    top = outer[0].subgridspec(1, 3, width_ratios=top_widths, wspace=TOP_PANEL_WSPACE)
    bottom = outer[1].subgridspec(1, 3, width_ratios=bottom_widths, wspace=BOTTOM_PANEL_WSPACE)
    axes = [
        fig.add_subplot(top[0, 0]),
        fig.add_subplot(top[0, 1]),
        fig.add_subplot(top[0, 2]),
        fig.add_subplot(bottom[0, 0]),
        fig.add_subplot(bottom[0, 1]),
    ]
    draw_a(axes[0])
    draw_b(axes[1], validation)
    draw_c(axes[2], species)
    draw_d(axes[3], classes)
    draw_e(axes[4], contrib)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.93, bottom=0.095)
    e_pos = axes[4].get_position()
    axes[4].set_position([e_pos.x0, e_pos.y0, e_pos.width * PANEL_E_WIDTH_MULTIPLIER, e_pos.height])
    stem = "figure3_ml_all_panels_v3_" + variant
    fig.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG / f'{stem}.png'}")


def write_caption() -> None:
    caption = (
        "**Figure 3. Protein-specific ML-guided ion/additive prioritization.** "
        "(a) Structure-modeled species classification workflow using species identity, species class and "
        "condition-context descriptors under grouped cross-validation. (b) Per-protein feature-set validation "
        "showing the predictive contribution of species identity, species class and condition context. "
        "(c) Classifier-predicted protein-specific species priorities. Bars show the mean out-of-fold "
        "classifier-predicted probability that a condition-side species is modeled in the deposited structure; "
        "dots show the raw structure-observed frequency, calculated as n_modeled/n_condition, and dot size "
        "indicates n_condition. The main panel displays species with n >= 10, while low-support ML hits are "
        "retained and flagged in the source table. (d) ML-predicted class-level chemical "
        "signal across the five proteins; cell color and central text indicate mean out-of-fold probability, "
        "with record support shown as explicit n labels. (e) Stacked feature-group contributions from the classifier, aggregating "
        "pH, temperature and method as condition context while keeping concentration context separate. "
        "Representative structure examples for these ML-prioritized signals are provided as a separate table."
    )
    (FIG / "figure3_ml_all_panels_v3_caption.md").write_text(caption, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = prepare_table()
    validation, oof, selected_c = validate_feature_sets(df)
    contrib = coefficient_contributions(df, selected_c)
    species = species_priority(oof)
    classes = class_signal(oof)
    guide = screening_guide(species, classes, contrib)
    write_tables(df, validation, oof, species, classes, contrib, guide)
    make_figure(validation, species, classes, contrib, guide, "hank")
    pdb_table = representative_pdb_table(guide)
    write_representative_pdb_table(pdb_table, "hank")
    draw_representative_pdb_table(pdb_table, "hank")
    write_caption()


if __name__ == "__main__":
    main()

