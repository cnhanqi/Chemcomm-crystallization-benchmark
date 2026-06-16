from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_figure4_ml_all_panels_v3.py"
OUT = ROOT / "results" / "figures" / "revised_submission_figures_v1" / "main_figures"

COMMON_TEXT_SIZE_PT = 8.0
HEATMAP_VALUE_SIZE_PT = 9.5
SUPPORT_COUNT_SIZE_PT = 6.0
DECIMAL_TEXT_RE = re.compile(r"^\d+\.\d+$")


def load_fig3_module():
    spec = importlib.util.spec_from_file_location("fig3_ml_source", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def mm_rect(fig_width_mm: float, fig_height_mm: float, x_mm: float, y_mm: float, w_mm: float, h_mm: float) -> list[float]:
    return [x_mm / fig_width_mm, y_mm / fig_height_mm, w_mm / fig_width_mm, h_mm / fig_height_mm]


def set_axis_text_size(ax: plt.Axes, size_pt: float) -> None:
    text_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
    text_items.extend(ax.get_xticklabels())
    text_items.extend(ax.get_yticklabels())
    text_items.extend(ax.texts)
    for text in text_items:
        text.set_fontsize(size_pt)
    legend = ax.get_legend()
    if legend is not None:
        legend.get_title().set_fontsize(size_pt)
        for text in legend.get_texts():
            text.set_fontsize(size_pt)


def apply_bcde_font_overrides(ax_b: plt.Axes, ax_c: plt.Axes, ax_d: plt.Axes, ax_e: plt.Axes, cbar_axes: list[plt.Axes]) -> None:
    for ax, label in [(ax_b, "b"), (ax_c, "c"), (ax_d, "d"), (ax_e, "e")]:
        for text in list(ax.texts):
            if text.get_text().strip() == label:
                text.remove()
    for text in list(ax_c.texts):
        if abs(float(text.get_rotation()) - 90.0) < 0.1:
            text.remove()

    for ax in [ax_b, ax_c, ax_d, ax_e, *cbar_axes]:
        set_axis_text_size(ax, COMMON_TEXT_SIZE_PT)

    for text in ax_b.texts:
        if DECIMAL_TEXT_RE.match(text.get_text().strip()):
            text.set_fontsize(HEATMAP_VALUE_SIZE_PT)

    for text in ax_d.texts:
        label = text.get_text().strip()
        if label.startswith("n="):
            text.set_text(label[2:])
            text.set_fontsize(SUPPORT_COUNT_SIZE_PT)
        elif DECIMAL_TEXT_RE.match(label):
            text.set_fontsize(HEATMAP_VALUE_SIZE_PT)


def main() -> None:
    fig3 = load_fig3_module()
    OUT.mkdir(parents=True, exist_ok=True)

    validation = pd.read_csv(fig3.OUT / "fig4_feature_set_validation.csv")
    species = pd.read_csv(fig3.OUT / "fig4_species_priority_ml.csv")
    classes = pd.read_csv(fig3.OUT / "fig4_class_signal_ml.csv")
    contrib = pd.read_csv(fig3.OUT / "fig4_feature_contribution_ml.csv")

    fig3.apply_style("hank")

    # Preserve the current panel scale and approximately the current visual 35 mm gaps.
    fig_w_mm = 332.0
    fig_h_mm = 118.0
    panel_h_mm = 66.8
    bottom_mm = 34.0
    gap_mm = 35.0
    cd_gap_mm = gap_mm
    cbar_pad_mm = 1.25
    cbar_w_mm = 2.65

    b_w_mm = 50.38
    c_w_mm = 55.34
    d_w_mm = 50.38
    e_w_mm = 37.48

    b_x = 16.0
    b_cbar_x = b_x + b_w_mm + cbar_pad_mm
    c_x = b_cbar_x + cbar_w_mm + gap_mm
    d_x = c_x + c_w_mm + cd_gap_mm
    d_cbar_x = d_x + d_w_mm + cbar_pad_mm
    e_x = d_cbar_x + cbar_w_mm + gap_mm

    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    ax_b = fig.add_axes(mm_rect(fig_w_mm, fig_h_mm, b_x, bottom_mm, b_w_mm, panel_h_mm))
    ax_c = fig.add_axes(mm_rect(fig_w_mm, fig_h_mm, c_x, bottom_mm, c_w_mm, panel_h_mm))
    ax_d = fig.add_axes(mm_rect(fig_w_mm, fig_h_mm, d_x, bottom_mm, d_w_mm, panel_h_mm))
    ax_e = fig.add_axes(mm_rect(fig_w_mm, fig_h_mm, e_x, bottom_mm, e_w_mm, panel_h_mm))

    fig3.draw_b(ax_b, validation)
    fig3.draw_c(ax_c, species)
    fig3.draw_d(ax_d, classes)
    fig3.draw_e(ax_e, contrib)

    fig.canvas.draw()
    cbar_axes = [ax for ax in fig.axes if ax not in {ax_b, ax_c, ax_d, ax_e}]
    if len(cbar_axes) >= 2:
        cbar_axes[0].set_position(mm_rect(fig_w_mm, fig_h_mm, b_cbar_x, bottom_mm + 5.0, cbar_w_mm, panel_h_mm - 10.0))
        cbar_axes[1].set_position(mm_rect(fig_w_mm, fig_h_mm, d_cbar_x, bottom_mm + 5.0, cbar_w_mm, panel_h_mm - 10.0))

    ax_b.set_position(mm_rect(fig_w_mm, fig_h_mm, b_x, bottom_mm, b_w_mm, panel_h_mm))
    ax_c.set_position(mm_rect(fig_w_mm, fig_h_mm, c_x, bottom_mm, c_w_mm, panel_h_mm))
    ax_d.set_position(mm_rect(fig_w_mm, fig_h_mm, d_x, bottom_mm, d_w_mm, panel_h_mm))
    ax_e.set_position(mm_rect(fig_w_mm, fig_h_mm, e_x, bottom_mm, e_w_mm, panel_h_mm))

    apply_bcde_font_overrides(ax_b, ax_c, ax_d, ax_e, cbar_axes)
    fig.canvas.draw()

    stem = OUT / "Figure3_ml_guided_ion_additive_prioritization_bcde_row"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.svg')}")


if __name__ == "__main__":
    main()






