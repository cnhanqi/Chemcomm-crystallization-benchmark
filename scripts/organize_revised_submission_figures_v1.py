from __future__ import annotations

import shutil
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hank.figures.style import PALETTE, PROTEIN_LABELS, PROTEIN_ORDER  # noqa: E402


DATA = ROOT / "data" / "ml" / "species_aware_screening_ml_v2.csv"
FIG_ROOT = ROOT / "results" / "figures"
ML_ROOT = ROOT / "results" / "ml" / "figure4_ml_v3"
SOURCE_MAIN = FIG_ROOT / "main_text_redesign"
SOURCE_ML = FIG_ROOT / "figure4_ml_v3"
OUT = FIG_ROOT / "revised_submission_figures_v1"
MAIN_OUT = OUT / "main_figures"
SI_OUT = OUT / "supporting_information"
SI_FIG = SI_OUT / "figures"
SI_TABLE = SI_OUT / "tables"
RECORDS = OUT / "records"


CLASS_ORDER = [
    "Anions",
    "Metal/cations",
    "Small neutral additives",
    "Organic ligands",
    "PEG/polymer",
    "IL/DES",
]
PANEL_CLASS_ORDER = [value for value in CLASS_ORDER if value != "IL/DES"]
ANIONS = {"nitrate", "sulfate", "chloride", "acetate", "citrate", "phosphate", "formate", "iodide", "bromide", "tartrate", "malate"}
METALS = {"calcium", "magnesium", "zinc", "sodium", "potassium", "manganese", "cobalt", "copper", "ammonium", "lithium", "cadmium", "nickel", "cesium"}
NEUTRALS = {"glycerol", "ethylene glycol", "mpd", "2-propanol", "ethanol", "methanol", "acetone", "dioxane", "dmf"}
LIGANDS = {"phenol", "cresol", "imidazole", "tris", "hepes", "mes", "bis-tris", "cacodylate", "dtt", "chaps"}


def ensure_dirs() -> None:
    for path in [OUT, MAIN_OUT, SI_OUT, SI_FIG, SI_TABLE, RECORDS]:
        path.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 8.2,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.65,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.5,
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


def pretty_protein(value: str) -> str:
    return PROTEIN_LABELS.get(value, value.title())


def pretty_species(value: str) -> str:
    mapping = {"mpd": "MPD", "peg": "PEG"}
    return mapping.get(str(value), str(value).title())


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


def load_species_table() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df = df[df["protein"].isin(PROTEIN_ORDER)].copy()
    df["protein"] = df["protein"].astype(str).str.lower()
    df["species_name"] = df["species_name"].astype(str).str.lower()
    df["species_class"] = df.apply(classify_species, axis=1)
    df["resolution_high_a"] = pd.to_numeric(df["resolution_high_a"], errors="coerce")
    df["observed_in_structure_label"] = df["observed_in_structure_label"].astype(int)
    return df


def save_bundle(fig: plt.Figure, stem: str) -> None:
    fig.savefig(SI_FIG / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(SI_FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def copy_main_figures() -> None:
    mapping = [
        (
            SOURCE_MAIN / "figure1_benchmark_architecture_v3",
            MAIN_OUT / "Figure1_benchmark_architecture",
            "Main Figure 1: dataset architecture.",
        ),
        (
            SOURCE_ML / "figure2_condition_landscape_v5_hank",
            MAIN_OUT / "Figure2_condition_chemistry_landscape",
            "Main Figure 2: protein-specific condition-chemistry landscape.",
        ),
        (
            SOURCE_ML / "figure3_ml_all_panels_v3_hank",
            MAIN_OUT / "Figure3_ml_guided_ion_additive_prioritization",
            "Main Figure 3: ML-guided ion/additive prioritization, renamed from the earlier Figure 4.",
        ),
    ]
    lines = [
        "# Revised Main-Figure Inventory",
        "",
        "This folder is the clean three-figure main-text set. Older intermediate files are left in their original directories.",
        "",
        "| New figure | Source | Decision |",
        "|---|---|---|",
    ]
    for source_stem, target_stem, decision in mapping:
        for ext in [".png", ".svg"]:
            src = source_stem.with_suffix(ext)
            if src.exists():
                shutil.copy2(src, target_stem.with_suffix(ext))
        lines.append(f"| {target_stem.name} | {source_stem.name} | {decision} |")
    caption_src = SOURCE_ML / "figure3_ml_all_panels_v3_caption.md"
    if caption_src.exists():
        shutil.copy2(caption_src, MAIN_OUT / "Figure3_ml_guided_ion_additive_prioritization_caption.md")
    captions = [
        "# Revised Main-Figure Captions",
        "",
        "**Figure 1. Species-resolved benchmark architecture for protein crystallization.** PDB/mmCIF records, crystallization-condition text, modeled non-polymer components and crystal-output metadata were linked across five model protein systems to generate condition records, additive-level records and species-resolved records. The row definition used downstream is protein + crystal entry + condition-side species, with a binary label indicating whether that species is explicitly modeled as a non-polymer component in the deposited structure.",
        "",
        "**Figure 2. Protein-specific condition-chemistry landscape.** Numbers indicate condition records containing each species. Color indicates the structure-observed rate, defined as the fraction of those records in which the same species is explicitly modeled as a non-polymer component in the deposited structure. Selected examples, method composition and metadata coverage show that condition frequency, modeled structural presence and experimental context are protein-dependent.",
        "",
        "**Figure 3. Protein-specific ML-guided ion/additive prioritization.** Species-resolved records were used to train grouped cross-validation classifiers for the structure-modeled species label. Feature-set validation, out-of-fold ML probabilities, class-level signals and feature-group contributions show that ion/additive identity and chemical class provide protein-specific prioritization information. The output is a first-pass prioritization of species for follow-up structural inspection, not a prediction of crystallization success.",
    ]
    (MAIN_OUT / "main_figure_captions.md").write_text("\n".join(captions) + "\n", encoding="utf-8")
    (MAIN_OUT / "main_figure_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_representative_pdb_table() -> None:
    src_csv = ML_ROOT / "figure3_representative_pdb_table_hank.csv"
    src_md = SOURCE_ML / "figure3_representative_pdb_table_hank.md"
    src_png = SOURCE_ML / "figure3_representative_pdb_table_hank.png"
    src_svg = SOURCE_ML / "figure3_representative_pdb_table_hank.svg"
    for src, dst in [
        (src_csv, SI_TABLE / "TableS1_representative_PDB_examples.csv"),
        (src_md, SI_TABLE / "TableS1_representative_PDB_examples.md"),
        (src_png, SI_TABLE / "TableS1_representative_PDB_examples.png"),
        (src_svg, SI_TABLE / "TableS1_representative_PDB_examples.svg"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)


def figure_s1_dataset_coverage(df: pd.DataFrame) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.35, 3.2), gridspec_kw={"width_ratios": [0.92, 1.38]})
    summary = (
        df.groupby("protein", as_index=False)
        .agg(
            species_records=("species_name", "size"),
            structure_modeled_records=("observed_in_structure_label", "sum"),
            unique_species=("species_name", "nunique"),
        )
        .set_index("protein")
        .reindex(PROTEIN_ORDER)
    )
    y = np.arange(len(PROTEIN_ORDER))
    axes[0].barh(y, summary["species_records"], color="#DDE6EE", edgecolor="white", height=0.62, label="Condition-side species records")
    axes[0].barh(y, summary["structure_modeled_records"], color=PALETTE["blue"], edgecolor="white", height=0.62, label="Structure-modeled records")
    axes[0].set_yticks(y, [pretty_protein(p) for p in PROTEIN_ORDER])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Records")
    axes[0].set_title("Species-resolved dataset scale", loc="left")
    axes[0].legend(loc="lower right", fontsize=5.9)
    axes[0].spines[["top", "right"]].set_visible(False)
    coverage = pd.DataFrame(index=PROTEIN_ORDER)
    coverage["pH"] = df.groupby("protein")["p_h"].apply(lambda x: 100 * x.notna().mean()).reindex(PROTEIN_ORDER)
    coverage["Temperature"] = df.groupby("protein")["temp_k"].apply(lambda x: 100 * x.notna().mean()).reindex(PROTEIN_ORDER)
    coverage["Method"] = df.groupby("protein")["method_family"].apply(lambda x: 100 * x.notna().mean()).reindex(PROTEIN_ORDER)
    coverage["Numeric concentration"] = df.groupby("protein")["has_numeric_concentration"].apply(lambda x: 100 * pd.to_numeric(x, errors="coerce").fillna(0).astype(bool).mean()).reindex(PROTEIN_ORDER)
    sns.heatmap(
        coverage,
        ax=axes[1],
        cmap=sns.light_palette(PALETTE["teal"], as_cmap=True),
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        linewidths=0.45,
        linecolor="white",
        cbar_kws={"label": "Coverage (%)", "fraction": 0.045, "pad": 0.025},
        annot_kws={"fontsize": 6.6},
    )
    axes[1].set_title("Parsed condition metadata coverage", loc="left")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([pretty_protein(p) for p in PROTEIN_ORDER], rotation=0)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=28, ha="right", rotation_mode="anchor")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.20, wspace=0.36)
    save_bundle(fig, "FigureS1_dataset_coverage_and_metadata_audit")


def figure_s2_full_species_matrix(df: pd.DataFrame) -> None:
    set_style()
    grouped = (
        df.groupby(["protein", "species_name", "species_class"], as_index=False)
        .agg(condition_records=("species_name", "size"), modeled_records=("observed_in_structure_label", "sum"))
    )
    grouped["structure_observed_rate"] = grouped["modeled_records"] / grouped["condition_records"]
    totals = grouped.groupby("species_name")["condition_records"].sum().sort_values(ascending=False)
    keep = totals[totals >= 40].index.tolist()[:28]
    plot = grouped[grouped["species_name"].isin(keep)].copy()
    class_order = {value: i for i, value in enumerate(CLASS_ORDER)}
    species_order = (
        plot[["species_name", "species_class"]]
        .drop_duplicates()
        .assign(class_rank=lambda d: d["species_class"].map(class_order).fillna(99), total=lambda d: d["species_name"].map(totals))
        .sort_values(["class_rank", "total"], ascending=[True, False])["species_name"]
        .tolist()
    )
    rate = plot.pivot(index="protein", columns="species_name", values="structure_observed_rate").reindex(index=PROTEIN_ORDER, columns=species_order)
    counts = plot.pivot(index="protein", columns="species_name", values="condition_records").reindex(index=PROTEIN_ORDER, columns=species_order)
    annot = counts.map(lambda x: "" if pd.isna(x) else f"{int(x)}")
    fig, ax = plt.subplots(figsize=(9.35, 3.25))
    sns.heatmap(
        rate,
        ax=ax,
        cmap=sns.light_palette(PALETTE["blue"], as_cmap=True),
        vmin=0,
        vmax=1,
        annot=annot,
        fmt="",
        linewidths=0.38,
        linecolor="white",
        cbar_kws={"label": "Structure-observed rate", "fraction": 0.035, "pad": 0.018},
        annot_kws={"fontsize": 5.5},
    )
    ax.set_title("High-support species matrix behind the condition-chemistry landscape", loc="left")
    ax.set_xlabel("Condition species")
    ax.set_ylabel("")
    ax.set_yticklabels([pretty_protein(p) for p in PROTEIN_ORDER], rotation=0)
    ax.set_xticklabels([pretty_species(x.get_text()) for x in ax.get_xticklabels()], rotation=45, ha="right", rotation_mode="anchor")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.30)
    save_bundle(fig, "FigureS2_high_support_species_condition_matrix")


def figure_s3_resolution_audit(df: pd.DataFrame) -> None:
    set_style()
    pairs = [
        ("lysozyme", "nitrate"),
        ("lysozyme", "peg"),
        ("ribonuclease", "magnesium"),
        ("ribonuclease", "sulfate"),
        ("trypsin", "calcium"),
        ("insulin", "zinc"),
        ("insulin", "phenol"),
        ("proteinase k", "calcium"),
        ("proteinase k", "nitrate"),
    ]
    rows = []
    for protein, species in pairs:
        sub = df[(df["protein"] == protein) & (df["species_name"] == species) & df["resolution_high_a"].notna()]
        if sub.empty:
            continue
        for state, label in [(0, "Not modeled"), (1, "Modeled")]:
            state_sub = sub[sub["observed_in_structure_label"] == state]
            rows.append(
                {
                    "case": f"{pretty_protein(protein)} - {pretty_species(species)}",
                    "state": label,
                    "median_resolution": state_sub["resolution_high_a"].median() if len(state_sub) else np.nan,
                    "n": len(state_sub),
                }
            )
    plot = pd.DataFrame(rows)
    order = [f"{pretty_protein(p)} - {pretty_species(s)}" for p, s in pairs if f"{pretty_protein(p)} - {pretty_species(s)}" in set(plot["case"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    y_lookup = {case: i for i, case in enumerate(order)}
    for case in order:
        sub = plot[plot["case"] == case]
        modeled = sub[sub["state"] == "Modeled"].iloc[0]
        not_modeled = sub[sub["state"] == "Not modeled"].iloc[0]
        y = y_lookup[case]
        xs = [not_modeled["median_resolution"], modeled["median_resolution"]]
        if not any(pd.isna(xs)):
            ax.plot(xs, [y, y], color="#B8C3CF", lw=1.1, zorder=1)
        ax.scatter(not_modeled["median_resolution"], y, s=34, color="#D1D5DB", edgecolor="white", zorder=2, label="Not modeled" if y == 0 else None)
        ax.scatter(modeled["median_resolution"], y, s=34, color=PALETTE["teal"], edgecolor="white", zorder=3, label="Modeled" if y == 0 else None)
        ax.text(3.02, y, f"n={int(modeled['n'])}/{int(not_modeled['n'])}", fontsize=6.2, va="center", color="#4B5563")
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlim(0.45, 3.25)
    ax.set_xlabel("Median resolution (A)")
    ax.set_title("Selected structure-modeled species and crystallographic resolution", loc="left")
    ax.text(3.02, -0.78, "n modeled/not", fontsize=6.2, color="#4B5563", ha="left")
    ax.legend(loc="lower right")
    ax.grid(axis="x", color="#E8EDF2", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.30, right=0.96, top=0.88, bottom=0.14)
    save_bundle(fig, "FigureS3_resolution_association_audit")


def figure_s4_class_support(classes: pd.DataFrame) -> None:
    set_style()
    records = classes.pivot(index="protein", columns="species_class", values="n_condition").reindex(index=PROTEIN_ORDER, columns=PANEL_CLASS_ORDER)
    species = classes.pivot(index="protein", columns="species_class", values="number_of_species").reindex(index=PROTEIN_ORDER, columns=PANEL_CLASS_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(9.35, 3.0), gridspec_kw={"width_ratios": [1, 1]})
    for ax, data, title, label in [
        (axes[0], records, "Condition-record support behind main Fig. 3d", "Records"),
        (axes[1], species, "Distinct species per class", "Species count"),
    ]:
        sns.heatmap(
            data,
            ax=ax,
            cmap=sns.light_palette(PALETTE["purple"], as_cmap=True),
            annot=True,
            fmt=".0f",
            linewidths=0.45,
            linecolor="white",
            cbar_kws={"label": label, "fraction": 0.045, "pad": 0.020},
            annot_kws={"fontsize": 6.3},
        )
        ax.set_title(title, loc="left")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels([pretty_protein(p) for p in PROTEIN_ORDER], rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=34, ha="right", rotation_mode="anchor")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.30, wspace=0.38)
    save_bundle(fig, "FigureS4_ml_class_support_counts")


def figure_s5_ml_validation_support(validation: pd.DataFrame) -> None:
    set_style()
    roc = validation[(validation["metric"] == "ROC-AUC") & (validation["feature_set"] == "Full model")].set_index("protein").reindex(PROTEIN_ORDER)
    rows = pd.DataFrame(
        {
            "ROC-AUC": roc["mean_score"],
            "CV groups": roc["n_groups"],
            "Positive rows": roc["n_positive"],
            "Negative rows": roc["n_negative"],
        },
        index=PROTEIN_ORDER,
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), gridspec_kw={"width_ratios": [0.86, 1.14]})
    axes[0].barh(np.arange(len(rows)), rows["ROC-AUC"], color=PALETTE["purple"], edgecolor="white", height=0.62)
    axes[0].set_yticks(np.arange(len(rows)), [pretty_protein(p) for p in rows.index])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.5, 1.0)
    axes[0].set_xlabel("Full-model ROC-AUC")
    axes[0].set_title("Grouped-CV validation", loc="left")
    axes[0].grid(axis="x", color="#E8EDF2", lw=0.55)
    axes[0].spines[["top", "right"]].set_visible(False)
    support = rows[["CV groups", "Positive rows", "Negative rows"]]
    sns.heatmap(
        support,
        ax=axes[1],
        cmap=sns.light_palette(PALETTE["gold"], as_cmap=True),
        annot=True,
        fmt=".0f",
        linewidths=0.45,
        linecolor="white",
        cbar=False,
        annot_kws={"fontsize": 6.8},
    )
    axes[1].set_title("Validation support", loc="left")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([pretty_protein(p) for p in support.index], rotation=0)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.22, wspace=0.42)
    save_bundle(fig, "FigureS5_ml_validation_support_audit")


def write_si_manifest() -> None:
    lines = [
        "# Revised Supporting Information Figure Set",
        "",
        "This folder is a clean SI figure set built for the revised three-main-figure manuscript.",
        "The old `supporting_information` directory is intentionally not reused wholesale because several old figures use outdated terminology or support claims that are no longer central.",
        "",
        "## Main-text figure numbering",
        "",
        "- Figure 1: benchmark architecture.",
        "- Figure 2: protein-specific condition-chemistry landscape.",
        "- Figure 3: ML-guided protein-specific ion/additive prioritization. This was previously developed in the `figure4_ml_v3` folder and has been renamed to Figure 3 in the clean package.",
        "",
        "## Curated SI figures",
        "",
        "| SI figure | File stem | Purpose | Main figure supported |",
        "|---|---|---|---|",
        "| Figure S1 | FigureS1_dataset_coverage_and_metadata_audit | Dataset scale and metadata coverage audit. | Fig. 1/Fig. 2 |",
        "| Figure S2 | FigureS2_high_support_species_condition_matrix | High-support species matrix behind the condition-chemistry map. | Fig. 2 |",
        "| Figure S3 | FigureS3_resolution_association_audit | Compact structural-output audit for selected species. | Fig. 2/Fig. 3 |",
        "| Figure S4 | FigureS4_ml_class_support_counts | Explicit record and species support behind class-level ML signals. | Fig. 3d |",
        "| Figure S5 | FigureS5_ml_validation_support_audit | Grouped-CV support audit for the full ML model. | Fig. 3b |",
        "",
        "## Supplementary tables",
        "",
        "- Table S1: representative PDB examples for ML-prioritized ion/additive signals.",
        "",
        "## Old figures not carried forward by default",
        "",
        "- Old frequency-versus-structure-observed scatter plots: redundant with Fig. 2 and harder to interpret.",
        "- Old retained-state ranking figures: terminology conflicts with the revised structure-modeled species language.",
        "- Old pH/concentration-window ranking figures: exploratory only; they overemphasize heterogeneous metadata.",
        "- Old broad algorithm leaderboards and SHAP plots: not central to the chemical story.",
        "- Old solvent-content regression figures: not part of the main binary structure-modeled species task.",
        "",
        "Terminology to use throughout: structure-modeled species, modeled in structure, structure-observed rate, out-of-fold P(structure-modeled). Avoid retained-state and crystallization-success prediction.",
    ]
    (SI_OUT / "SI_figure_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RECORDS / "revised_submission_package_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    captions = [
        "# Revised Supporting Information Figure Captions",
        "",
        "**Figure S1. Dataset scale and condition-metadata coverage.** Species-resolved record counts and structure-modeled record counts are shown for each model protein. Metadata coverage indicates whether parsed pH, temperature, method and numeric concentration fields are available; coverage does not imply mechanistic importance.",
        "",
        "**Figure S2. High-support species matrix behind the condition-chemistry landscape.** Rows are proteins and columns are high-support condition-side species. Cell numbers indicate condition records and cell color indicates structure-observed rate. This figure provides the broader audit trail behind the compact main-text chemistry landscape.",
        "",
        "**Figure S3. Selected structure-modeled species and crystallographic resolution.** Median resolution is compared between records where a condition-side species is modeled in the deposited structure and records where it is not modeled. The comparison is descriptive and protein-specific; it should not be interpreted as a universal improvement trend.",
        "",
        "**Figure S4. Support counts behind class-level ML signals.** Record counts and distinct species counts are shown for each protein and chemical class. These values support the class-level ML probability heatmap in main Figure 3 and avoid ambiguous slash-style support labels in the main panel.",
        "",
        "**Figure S5. Grouped cross-validation support for the full ML model.** Full-model ROC-AUC is shown with the number of grouped-CV groups and positive/negative rows for each protein. The panel is a support audit for the main ML validation rather than an algorithm leaderboard.",
        "",
        "**Table S1. Representative PDB examples for ML-prioritized ion/additive signals.** For each protein, a primary and backup PDB entry are listed with visible modeled ions/additives and the reason for selecting the primary structure.",
    ]
    (SI_OUT / "SI_figure_captions.md").write_text("\n".join(captions) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    copy_main_figures()
    copy_representative_pdb_table()
    df = load_species_table()
    classes = pd.read_csv(ML_ROOT / "fig4_class_signal_ml.csv")
    validation = pd.read_csv(ML_ROOT / "fig4_feature_set_validation.csv")
    figure_s1_dataset_coverage(df)
    figure_s2_full_species_matrix(df)
    figure_s3_resolution_audit(df)
    figure_s4_class_support(classes)
    figure_s5_ml_validation_support(validation)
    write_si_manifest()
    print(f"Wrote revised submission package to {OUT}")


if __name__ == "__main__":
    main()


