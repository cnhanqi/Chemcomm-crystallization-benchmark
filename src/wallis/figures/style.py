from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
WALLIS_FIGURES_DIR = ROOT / "results" / "figures"
WALLIS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "ink": "#23313D",
    "blue": "#4C78A8",
    "teal": "#4E9A8B",
    "orange": "#D98C5F",
    "green": "#7A9E59",
    "purple": "#8D7AAE",
    "gold": "#C9A66B",
    "slate": "#B8BEC7",
    "light_grid": "#E8EDF2",
}

PANEL_LABELS_ENABLED = False
ANNOTATION_FONTSIZE = 10
ANNOTATION_FONTSIZE_SMALL = 9
ANNOTATION_FONTWEIGHT = "normal"
HEATMAP_ANNOT_KWS = {
    "fontsize": ANNOTATION_FONTSIZE_SMALL,
    "fontweight": ANNOTATION_FONTWEIGHT,
    "color": PALETTE["ink"],
}

FEATURE_SET_COLORS = {
    "combined": PALETTE["teal"],
    "chemistry_only": PALETTE["orange"],
    "protein_only": PALETTE["slate"],
}

STATE_COLORS = {
    "yes": "#0F766E",
    "no": PALETTE["slate"],
}

PROTEIN_ORDER = ["lysozyme", "ribonuclease", "trypsin", "insulin", "proteinase k"]
PROTEIN_LABELS = {
    "lysozyme": "Lysozyme",
    "ribonuclease": "Ribonuclease",
    "trypsin": "Trypsin",
    "insulin": "Insulin",
    "proteinase k": "Proteinase K",
}
PROTEIN_COLORS = {
    "lysozyme": PALETTE["blue"],
    "ribonuclease": PALETTE["teal"],
    "trypsin": PALETTE["green"],
    "insulin": PALETTE["purple"],
    "proteinase k": PALETTE["gold"],
}

ML_STORY_EXPORTS = {
    "ml_figure1_dataset_overview_v1": "figure7_ml_dataset_overview_v1",
    "ml_figure2_model_comparison_panel_v1": "figure8_ml_model_comparison_v1",
    "ml_figure3_classification_diagnostics_v1": "figure9_ml_retention_classification_v1",
    "ml_figure4_regression_diagnostics_v1": "figure10_ml_solvent_regression_v1",
    "ml_figure5_feature_importance_panel_v1": "figure11_ml_feature_importance_v1",
}


def apply_publication_style(grid: bool = False) -> None:
    sns.set_theme(style="ticks")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.weight": "normal",
            "axes.labelsize": 18,
            "axes.titlesize": 19,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
            "figure.dpi": 300,
            "axes.linewidth": 1.4,
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "lines.linewidth": 2.2,
            "axes.grid": grid,
            "grid.color": PALETTE["light_grid"],
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def style_axes(ax, rotation: int | None = None) -> None:
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for label in ax.get_xticklabels():
        label.set_fontweight("normal")
        if rotation is not None:
            label.set_rotation(rotation)
            label.set_ha("right")
            label.set_rotation_mode("anchor")
    for label in ax.get_yticklabels():
        label.set_fontweight("normal")
    ax.grid(False)


def panel_label(ax, label: str) -> None:
    if not PANEL_LABELS_ENABLED:
        return
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def save_figure_bundle(fig: plt.Figure, out_path: str | Path, save_svg: bool = True) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if save_svg:
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")


def export_story_figure(source_path: str | Path, story_stem: str, save_svg: bool = True) -> None:
    source_path = Path(source_path)
    story_png = WALLIS_FIGURES_DIR / f"{story_stem}.png"
    shutil.copy2(source_path, story_png)
    if save_svg:
        svg_source = source_path.with_suffix(".svg")
        if svg_source.exists():
            shutil.copy2(svg_source, WALLIS_FIGURES_DIR / f"{story_stem}.svg")
