from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wallis.figures.style import PALETTE, PROTEIN_COLORS, PROTEIN_LABELS, PROTEIN_ORDER, STATE_COLORS  # noqa: E402


DATA = ROOT / "data" / "benchmark_tables"
ML_DATA = ROOT / "data" / "ml"
OUT = ROOT / "results" / "figures" / "main_text_redesign"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#202326"
MUTED = "#6B7280"
LIGHT = "#E8EDF2"
LIGHTER = "#F7FAFC"
ARROW = "#536879"
TEAL = STATE_COLORS["yes"]
GREY = STATE_COLORS["no"]


def apply_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.4,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.6,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.65,
            "axes.grid": False,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.fontsize": 7.0,
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


def panel_label(ax: plt.Axes, label: str, x: float = -0.04, y: float = 1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.6, fontweight="bold", ha="left", va="top", color="black")


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str = "",
    face: str = LIGHTER,
    edge: str = "#2D536B",
    title_size: float = 7.1,
    body_size: float = 6.4,
    title_weight: str = "bold",
) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.024",
            transform=ax.transAxes,
            linewidth=0.75,
            edgecolor=edge,
            facecolor=face,
        )
    )
    ax.text(x + 0.035 * w, y + h - 0.20 * h, title, transform=ax.transAxes, fontsize=title_size, fontweight=title_weight, color=INK, va="top")
    if body:
        ax.text(x + 0.035 * w, y + 0.16 * h, body, transform=ax.transAxes, fontsize=body_size, color=MUTED, va="bottom", linespacing=1.15)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], scale: float = 8.0) -> None:
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=scale, color=ARROW, linewidth=0.8))


def protein_icon(ax: plt.Axes, cx: float, cy: float, color: str, scale: float = 0.72) -> None:
    base = [(-0.028, 0.012), (-0.006, 0.028), (0.020, 0.018), (0.026, -0.012), (0.000, -0.026), (-0.025, -0.012)]
    offsets = [(dx * scale, dy * scale) for dx, dy in base]
    pts = [(cx + dx, cy + dy) for dx, dy in offsets]
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        ax.plot([x1, x2], [y1, y2], transform=ax.transAxes, color=color, linewidth=0.8, alpha=0.75)
    for x, y in pts:
        ax.add_patch(Circle((x, y), 0.008 * scale, transform=ax.transAxes, facecolor=color, edgecolor="white", linewidth=0.4))


def tidy_open_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(length=2.8, pad=2, direction="out")


def draw_panel_a(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title("Five model proteins", loc="left", pad=4)
    panel_label(ax, "a")
    cards = [
        ("lysozyme", "Lysozyme", "anions and small solvents"),
        ("ribonuclease", "Ribonuclease", "mixed ions"),
        ("trypsin", "Trypsin", r"Ca$^{2+}$-linked stabilization"),
        ("insulin", "Insulin", r"Zn$^{2+}$-linked assembly"),
        ("proteinase k", "Proteinase K", r"Ca$^{2+}$-linked stabilization"),
    ]
    positions = [(0.02, 0.61), (0.51, 0.61), (0.02, 0.37), (0.51, 0.37), (0.25, 0.13)]
    for (protein, label, tag), (x, y) in zip(cards, positions):
        color = PROTEIN_COLORS[protein]
        rounded_box(ax, (x, y), (0.43, 0.18), label, tag, face="#FFFFFF", edge=color, title_size=6.35, body_size=5.75)
        protein_icon(ax, x + 0.37, y + 0.124, color)


def draw_panel_b(ax: plt.Axes, pdb_count: int) -> None:
    ax.set_axis_off()
    ax.set_title("PDB structure and\ncrystallization records", loc="left", pad=3)
    panel_label(ax, "b")
    rounded_box(ax, (0.18, 0.70), (0.64, 0.17), "PDB/mmCIF records", f"{pdb_count:,} PDB IDs", face="#FFFFFF", edge=PALETTE["blue"], title_size=6.9, body_size=6.0)
    rounded_box(ax, (0.08, 0.44), (0.34, 0.14), "Crystallization text", "", face="#FFFFFF", edge=PALETTE["blue"], title_size=6.1)
    rounded_box(ax, (0.48, 0.44), (0.40, 0.14), "Modeled non-polymer\nspecies", "", face="#FFFFFF", edge=TEAL, title_size=5.65)
    rounded_box(ax, (0.15, 0.16), (0.67, 0.16), "Resolution | solvent content |\nMatthews coefficient", "", face="#FFFFFF", edge=PALETTE["purple"], title_size=5.6)
    arrow(ax, (0.33, 0.70), (0.25, 0.58), scale=7)
    arrow(ax, (0.67, 0.70), (0.68, 0.58), scale=7)
    arrow(ax, (0.50, 0.70), (0.48, 0.32), scale=7)


def draw_panel_c(ax: plt.Axes, condition_rows: int, additive_rows: int, species_rows: int) -> None:
    ax.set_axis_off()
    ax.set_title("From condition text\nto specific species", loc="left", pad=3)
    panel_label(ax, "c")
    rounded_box(ax, (0.05, 0.72), (0.88, 0.15), "Crystallization condition", "PEG, NaCl, nitrate, pH 6.5", face="#FFFFFF", edge=PALETTE["blue"], title_size=6.8, body_size=5.8)
    rounded_box(ax, (0.11, 0.45), (0.76, 0.14), "Recognized chemical species", r"Na$^+$ | Cl$^-$ | NO$_3^-$ | PEG", face=LIGHTER, edge=TEAL, title_size=6.45, body_size=5.9)
    rounded_box(ax, (0.05, 0.20), (0.88, 0.13), "Specific-species records", "protein + PDB ID + crystal + species", face="#FFFFFF", edge=PALETTE["purple"], title_size=6.8, body_size=5.55)
    arrow(ax, (0.49, 0.72), (0.49, 0.59))
    arrow(ax, (0.49, 0.45), (0.49, 0.33))
    badges = [
        (f"{condition_rows:,}", "condition\nrecords", PALETTE["blue"], 0.02),
        (f"{additive_rows:,}", "additive-level\nrecords", TEAL, 0.35),
        (f"{species_rows:,}", "specific-species\nrecords", PALETTE["purple"], 0.68),
    ]
    for value, label, color, x in badges:
        ax.add_patch(Rectangle((x, 0.03), 0.24, 0.09, transform=ax.transAxes, facecolor="#FFFFFF", edgecolor=color, linewidth=0.65))
        ax.text(x + 0.02, 0.083, value, transform=ax.transAxes, fontsize=6.35, fontweight="bold", color=INK, va="center")
        ax.text(x + 0.02, 0.047, label, transform=ax.transAxes, fontsize=4.85, color=MUTED, va="center", linespacing=0.9)


def draw_panel_d(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title("Three linked data layers", loc="left", pad=4)
    panel_label(ax, "d")
    layers = [
        ("Condition chemistry", "species | conc. | pH | temp. | method", PALETTE["blue"], 0.67),
        ("Modeled structural species", "modeled? | ligand IDs | instance counts", TEAL, 0.41),
        ("Crystal outputs", "solvent content | Matthews coefficient | resolution", PALETTE["purple"], 0.15),
    ]
    for title, body, color, y in layers:
        rounded_box(ax, (0.06, y), (0.88, 0.17), title, body, face="#FFFFFF", edge=color, title_size=6.8, body_size=5.25)
    arrow(ax, (0.50, 0.67), (0.50, 0.58), scale=7)
    arrow(ax, (0.50, 0.41), (0.50, 0.32), scale=7)
    ax.add_patch(Rectangle((0.07, 0.02), 0.86, 0.08, transform=ax.transAxes, facecolor="#FFF7ED", edgecolor="#D98C5F", linewidth=0.65))
    ax.text(0.50, 0.06, "Present in condition != modeled in structure", transform=ax.transAxes, ha="center", va="center", fontsize=6.45, fontweight="bold", color="#7C2D12")


def draw_panel_e(ax: plt.Axes, ml_rows: pd.Series) -> None:
    ax.set_title("Data coverage by protein", loc="left", pad=4)
    panel_label(ax, "e")
    rows = ml_rows.reindex(PROTEIN_ORDER)
    y = np.arange(len(rows))
    colors = [PROTEIN_COLORS[p] for p in rows.index]
    ax.barh(y, rows.values, color=colors, edgecolor="white", height=0.68)
    ax.set_yticks(y, [""] * len(rows))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Additive-level records")
    ax.invert_yaxis()
    for yi, protein, val in zip(y, rows.index, rows.values):
        label = PROTEIN_LABELS[protein]
        if val >= 600:
            ax.text(40, yi, label, va="center", ha="left", fontsize=6.6, color="white", fontweight="bold")
            ax.text(val + 35, yi, f"{int(val):,}", va="center", fontsize=6.7, color=INK)
        else:
            ax.text(val + 35, yi, f"{label}  {int(val):,}", va="center", fontsize=6.7, color=INK)
    ax.set_xlim(0, max(rows.values) * 1.22)
    tidy_open_axis(ax)


def draw_panel_f(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title("Data-guided\nscreening outputs", loc="left", pad=3)
    panel_label(ax, "f")
    boxes = [
        ("Model input table", 0.70, PALETTE["blue"]),
        ("Predict species modeled in structure", 0.49, TEAL),
        ("Protein-specific species ranking", 0.28, PALETTE["green"]),
        ("Concentration-window ranking", 0.07, PALETTE["gold"]),
    ]
    for title, y, color in boxes:
        rounded_box(ax, (0.08, y), (0.80, 0.13), title, "", face="#FFFFFF", edge=color, title_size=6.6)
    for y1, y2 in [(0.70, 0.62), (0.49, 0.41), (0.28, 0.20)]:
        arrow(ax, (0.48, y1), (0.48, y2), scale=7)


def make_figure() -> None:
    apply_style()

    conditions = pd.read_csv(DATA / "pdb_core_model_proteins_condition_entries_v2.csv")
    master = pd.read_csv(DATA / "master_benchmark_table_v1.csv")
    species_status = pd.read_csv(DATA / "pdb_core_model_proteins_species_entry_status_v3.csv")
    ml_df = pd.read_csv(ML_DATA / "master_benchmark_screening_ml_v1.csv")

    fig = plt.figure(figsize=(6.73, 5.55))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0, 1.0], hspace=0.48, wspace=0.38)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    draw_panel_a(axes[0])
    draw_panel_b(axes[1], conditions["pdb_id"].nunique())
    draw_panel_c(axes[2], len(conditions), len(master), len(species_status))
    draw_panel_d(axes[3])
    draw_panel_e(axes[4], ml_df["protein"].value_counts())
    draw_panel_f(axes[5])

    fig.text(0.02, 0.988, "Building a chemical-species dataset for protein crystallization", ha="left", va="top", fontsize=9.4, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.070, right=0.985, top=0.902, bottom=0.085)

    stem = OUT / "figure1_benchmark_architecture_v3"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.png')}")


if __name__ == "__main__":
    make_figure()

