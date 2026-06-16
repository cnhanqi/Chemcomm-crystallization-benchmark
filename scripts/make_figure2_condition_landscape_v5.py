from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hank.figures.style import PROTEIN_LABELS

DATA = ROOT / "data" / "benchmark_tables"
ML_DATA = ROOT / "data" / "ml" / "species_aware_screening_ml_v2.csv"
CONDITION_TABLE = DATA / "pdb_core_model_proteins_condition_entries_v2.csv"
OUT = ROOT / "results" / "figures" / "main_text_redesign"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#202326"
MUTED = "#6B7280"
LIGHT = "#E8EDF2"
HEATMAP_CMAP = sns.light_palette("#4C78A8", as_cmap=True)
Y_LABELPAD_PT = 0.41 / 25.4 * 72.0

CLASS_COLORS = {
    "Anions": "#4C78A8",
    "Metal/cations": "#8D7AAE",
    "Small neutral additives": "#4E9A8B",
    "PEG / IL-DES": "#C9A66B",
}

METHOD_ORDER = [
    "Hanging drop",
    "Sitting drop",
    "Batch",
    "Vapor diffusion / unspecified",
    "Other",
]

METHOD_COLORS = {
    "Hanging drop": "#4C78A8",
    "Sitting drop": "#4E9A8B",
    "Batch": "#C9A66B",
    "Vapor diffusion / unspecified": "#B8BEC7",
    "Other": "#8D7AAE",
}

METHOD_LEGEND_LABELS = {
    "Hanging drop": "Hanging",
    "Sitting drop": "Sitting",
    "Batch": "Batch",
    "Vapor diffusion / unspecified": "VD/unspec.",
    "Other": "Other",
}

SPECIES_ORDER = [
    "sodium",
    "chloride",
    "sulfate",
    "nitrate",
    "acetate",
    "citrate",
    "zinc",
    "calcium",
    "magnesium",
    "ethylene glycol",
    "glycerol",
    "phenol",
    "peg",
    "IL/DES",
]

SPECIES_LABELS = {
    "sodium": "Sodium",
    "chloride": "Chloride",
    "sulfate": "Sulfate",
    "nitrate": "Nitrate",
    "acetate": "Acetate",
    "citrate": "Citrate",
    "zinc": "Zinc",
    "calcium": "Calcium",
    "magnesium": "Magnesium",
    "ethylene glycol": "Ethylene glycol",
    "glycerol": "Glycerol",
    "phenol": "Phenol",
    "peg": "PEG",
    "IL/DES": "IL/DES",
}

PROTEINS = ["lysozyme", "ribonuclease", "trypsin", "insulin", "proteinase k"]
SHORT_PROTEIN_LABELS = {
    "lysozyme": "Lys.",
    "ribonuclease": "RNase",
    "trypsin": "Tryp.",
    "insulin": "Ins.",
    "proteinase k": "Prot. K",
}


def apply_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.9,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "axes.labelweight": "normal",
            "axes.linewidth": 0.65,
            "axes.grid": False,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.9,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 5.8,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.8, fontweight="bold", va="top", ha="left", color=INK)


def open_axes(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.65)
    if grid == "x":
        ax.grid(axis="x", color=LIGHT, linewidth=0.55)
    elif grid == "y":
        ax.grid(axis="y", color=LIGHT, linewidth=0.55)
    else:
        ax.grid(False)


def text_color_for_fill(color: str) -> str:
    r, g, b = matplotlib.colors.to_rgb(color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.50 else INK


def species_display_name(name: str) -> str:
    return SPECIES_LABELS.get(name, name.title())


def get_chemical_class(row: pd.Series) -> str:
    major = str(row["species_major_class"]).lower()
    minor = str(row["species_minor_class"]).lower()
    name = str(row["species_name"]).lower()

    if name in ["peg", "ionic liquid", "ethylammonium nitrate", "il/des"]:
        return "PEG / IL-DES"
    if major == "ion" and minor == "anion":
        return "Anions"
    if major == "ion" and minor == "cation":
        return "Metal/cations"
    if name in ["acetate", "citrate", "formate", "tartrate"]:
        return "Anions"
    if major in ["solvent_or_cosolvent"] or name in ["ethylene glycol", "glycerol", "phenol"]:
        return "Small neutral additives"
    if "solvent" in major or "polyol" in minor:
        return "Small neutral additives"
    return "Other"


def rate_or_zero(retained: float, total: float) -> float:
    return float(retained) / float(total) if total else 0.0


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(DATA / "pdb_core_model_proteins_species_retention_summary_v2.csv")
    df.loc[df["species_name"].isin(["ionic liquid", "ethylammonium nitrate"]), "species_name"] = "IL/DES"

    df_agg = (
        df.groupby(["protein", "species_name"], as_index=False)
        .agg({"condition_entries": "sum", "retained_entries": "sum"})
        .copy()
    )
    class_map = df.drop_duplicates(subset=["species_name"])[
        ["species_name", "species_major_class", "species_minor_class"]
    ]
    df_agg = pd.merge(df_agg, class_map, on="species_name", how="left")
    df_agg["chem_class"] = df_agg.apply(get_chemical_class, axis=1)
    df_agg["structure_observed_rate"] = df_agg.apply(
        lambda r: rate_or_zero(r["retained_entries"], r["condition_entries"]),
        axis=1,
    )
    return df_agg


def harmonize_method(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    if "hanging" in text:
        return "Hanging drop"
    if "sitting" in text:
        return "Sitting drop"
    if "batch" in text:
        return "Batch"
    if "unspecified" in text or text in {"", "nan", "unspecified/other"}:
        return "Vapor diffusion / unspecified"
    return "Other"


def mentions_yes(series: pd.Series) -> int:
    return int(series.astype(str).str.lower().isin(["yes", "true", "1"]).sum())


def build_condition_level() -> pd.DataFrame:
    species = pd.read_csv(ML_DATA)
    conditions = pd.read_csv(CONDITION_TABLE)
    base = species[["group_id", "protein", "pdb_id", "crystal_id", "method_family", "p_h", "temp_k"]].drop_duplicates("group_id")
    annotations = conditions[["protein", "pdb_id", "crystal_id", "mentions_seeding", "mentions_cryoprotectant"]].drop_duplicates(
        ["protein", "pdb_id", "crystal_id"]
    )
    return base.merge(annotations, on=["protein", "pdb_id", "crystal_id"], how="left")


def build_method_table(condition_level: pd.DataFrame) -> pd.DataFrame:
    df = condition_level.copy()
    df["method_group"] = df["method_family"].map(harmonize_method)
    counts = df.groupby(["protein", "method_group"], as_index=False).agg(condition_rows=("group_id", "size"))
    full_index = pd.MultiIndex.from_product([PROTEINS, METHOD_ORDER], names=["protein", "method_group"])
    counts = counts.set_index(["protein", "method_group"]).reindex(full_index, fill_value=0).reset_index()
    totals = counts.groupby("protein")["condition_rows"].transform("sum")
    counts["fraction"] = counts["condition_rows"] / totals
    return counts


def build_coverage(condition_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protein in PROTEINS:
        sub = condition_level[condition_level["protein"] == protein]
        n = len(sub)
        rows.append(
            {
                "protein": protein,
                "pH": 100 * sub["p_h"].notna().sum() / n,
                "Temperature": 100 * sub["temp_k"].notna().sum() / n,
                "Seeding": 100 * mentions_yes(sub["mentions_seeding"]) / n,
                "Cryoprotectant": 100 * mentions_yes(sub["mentions_cryoprotectant"]) / n,
            }
        )
    return pd.DataFrame(rows)


def draw_panel_a(ax: plt.Axes, fig: plt.Figure, df_agg: pd.DataFrame) -> None:
    df_a = df_agg[
        (df_agg["protein"].isin(PROTEINS)) & (df_agg["species_name"].isin(SPECIES_ORDER))
    ].copy()

    rate_matrix = (
        df_a.pivot(index="protein", columns="species_name", values="structure_observed_rate")
        .reindex(index=PROTEINS, columns=SPECIES_ORDER)
        .fillna(0)
    )
    count_matrix = (
        df_a.pivot(index="protein", columns="species_name", values="condition_entries")
        .reindex(index=PROTEINS, columns=SPECIES_ORDER)
        .fillna(0)
    )

    im = ax.imshow(rate_matrix.to_numpy(), cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0, aspect="auto")

    for i, protein in enumerate(PROTEINS):
        for j, species in enumerate(SPECIES_ORDER):
            count = count_matrix.loc[protein, species]
            rate = rate_matrix.loc[protein, species]
            color = "white" if rate > 0.62 else INK
            ax.text(j, i, f"{int(count)}", ha="center", va="center", fontsize=4.8, color=color, zorder=4)

    ax.set_xticks(range(len(SPECIES_ORDER)))
    ax.set_xticklabels([species_display_name(s) for s in SPECIES_ORDER], rotation=32, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(PROTEINS)))
    ax.set_yticklabels([PROTEIN_LABELS[p] for p in PROTEINS], rotation=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=2)
    ax.set_xticks(np.arange(-0.5, len(SPECIES_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PROTEINS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.010)
    cbar.set_label("Structure-observed rate", fontsize=7.4)
    cbar.ax.tick_params(labelsize=6.8, length=2.2)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.outline.set_linewidth(0.55)


def draw_examples_panel(ax: plt.Axes) -> None:
    examples = [
        ("Trypsin-Ca$^{2+}$", 0.968254, 126, CLASS_COLORS["Metal/cations"]),
        ("Insulin-Zn$^{2+}$", 0.937500, 64, CLASS_COLORS["Metal/cations"]),
        ("Proteinase K-Ca$^{2+}$", 0.920635, 63, CLASS_COLORS["Metal/cations"]),
        ("Lysozyme-NO$_3^-$", 0.804878, 82, CLASS_COLORS["Anions"]),
        ("Lysozyme-Cl$^-$", 0.578554, 401, CLASS_COLORS["Anions"]),
        ("Trypsin-Na$^+$", 0.095238, 126, CLASS_COLORS["Metal/cations"]),
        ("Insulin-Na$^+$", 0.069264, 231, CLASS_COLORS["Metal/cations"]),
        ("Lysozyme-PEG", 0.000000, 402, CLASS_COLORS["PEG / IL-DES"]),
    ]

    y_pos = np.array([0, 1, 2, 3, 4, 6, 7, 8], dtype=float)
    labels = [e[0] for e in examples]
    rates = [e[1] for e in examples]
    n_counts = [e[2] for e in examples]
    colors = [e[3] for e in examples]

    ax.barh(y_pos, rates, color=colors, height=0.58, alpha=0.92)
    ax.axhline(5.0, color=LIGHT, linewidth=0.8)
    ax.text(0.02, -0.72, "Observed examples", fontsize=6.7, fontweight="bold", color=INK, ha="left", va="center")
    ax.text(0.02, 5.28, "Non-observed examples", fontsize=6.7, fontweight="bold", color=INK, ha="left", va="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Structure-observed rate")
    ax.set_ylabel("Protein", labelpad=Y_LABELPAD_PT)
    ax.set_ylim(8.75, -0.95)
    open_axes(ax, grid="x")
    for y, rate, n in zip(y_pos, rates, n_counts):
        x_text = min(rate + 0.025, 0.91)
        ax.text(x_text, y, f"n={n}", va="center", fontsize=6.5, color=INK)


def draw_method_panel(ax: plt.Axes, method_table: pd.DataFrame) -> None:
    y = np.arange(len(PROTEINS))
    left = np.zeros(len(PROTEINS))
    for method in METHOD_ORDER:
        sub = method_table[method_table["method_group"] == method].set_index("protein").loc[PROTEINS]
        values = sub["fraction"].to_numpy()
        ax.barh(
            y,
            values,
            left=left,
            height=0.60,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.45,
            label=METHOD_LEGEND_LABELS[method],
        )
        for yi, segment_left, value in zip(y, left, values):
            if value > 0.20:
                ax.text(
                    segment_left + value / 2,
                    yi,
                    f"{100 * value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6.7,
                    color=text_color_for_fill(METHOD_COLORS[method]),
                )
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([PROTEIN_LABELS[p] for p in PROTEINS])
    ax.tick_params(axis="y", labelsize=6.2, pad=1)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Crystallization method composition (%)")
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50", "100"])
    open_axes(ax, "x")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.25),
        ncol=5,
        fontsize=6.7,
        handlelength=0.75,
        handletextpad=0.25,
        columnspacing=0.35,
        borderaxespad=0.0,
    )


def draw_coverage_panel(ax: plt.Axes, coverage: pd.DataFrame) -> None:
    columns = ["pH", "Temperature", "Seeding", "Cryoprotectant"]
    matrix = coverage.set_index("protein").loc[PROTEINS, columns]
    im = ax.imshow(matrix.to_numpy(), cmap=HEATMAP_CMAP, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(PROTEINS)))
    ax.set_yticklabels([PROTEIN_LABELS[p] for p in PROTEINS])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PROTEINS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i, protein in enumerate(PROTEINS):
        for j, col in enumerate(columns):
            value = matrix.loc[protein, col]
            text_color = "white" if value > 65 else INK
            ax.text(j, i, f"{value:.0f}%", ha="center", va="center", fontsize=6.0, color=text_color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.055, pad=0.025)
    cbar.set_label("Metadata coverage (%)", fontsize=7.4)
    cbar.set_ticks([0, 50, 100])
    cbar.ax.tick_params(labelsize=6.1, length=2)


def make_figure() -> Path:
    apply_style()
    df_agg = load_summary()
    condition_level = build_condition_level()
    method_table = build_method_table(condition_level)
    coverage = build_coverage(condition_level)

    fig = plt.figure(figsize=(6.50, 4.30))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.68, 1.0],
        width_ratios=[2.0, 1.0, 1.0],
        hspace=0.47965,
        wspace=0.53958,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    draw_panel_a(ax_a, fig, df_agg)
    draw_examples_panel(ax_b)
    draw_method_panel(ax_c, method_table)
    draw_coverage_panel(ax_d, coverage)

    fig.subplots_adjust(left=0.095, right=0.965, top=0.955, bottom=0.245)
    stem = OUT / "figure2_condition_landscape_v5"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig((OUT / "figure2_condition_landscape_v5_hank").with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig((OUT / "figure2_condition_landscape_v5_hank").with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.png')}")
    return stem


if __name__ == "__main__":
    make_figure()


