# -*- coding: utf-8 -*-
"""Single combined SI figure (Fig. S10) for the pooled cross-protein signal check.
Consolidates the former three pooled SI figures into one compact 2x2 panel,
built directly from the existing comparison/importance CSV tables.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "results" / "ml" / "species_aware_v2"
OUT = ROOT / "results" / "figures" / "ml_outputs_chemistry_first" / "si_pooled"
OUT.mkdir(parents=True, exist_ok=True)

GREY = "#9AA3AD"
TEAL = "#2F8F86"
BLUE = "#3A6EA5"
INK = "#1F2933"

FS_LABEL = {
    "protein_only": "Protein only\n(baseline)",
    "protein_and_species_only": "Protein + species",
    "species_chemistry_only": "Chemistry\n(no protein)",
    "combined": "Combined\n(all features)",
}
FS_ORDER = ["protein_only", "protein_and_species_only", "species_chemistry_only", "combined"]
MODEL_LABEL = {
    "logistic_regression": "Logistic reg.",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boost.",
    "ridge": "Ridge",
}
MODEL_COLOR = {
    "logistic_regression": BLUE,
    "ridge": BLUE,
    "random_forest": TEAL,
    "gradient_boosting": "#C57B3F",
}
FEAT_LABEL = {
    "additive_group": "Additive group",
    "concentration_unit_simple": "Conc. unit",
    "protein": "Protein",
    "temp_k": "Temperature",
    "condition_concentration_values": "Conc. value",
    "p_h": "pH",
    "method_family": "Method",
    "condition_groups_count": "Condition groups",
    "is_ion_like": "Ion-like",
    "has_concentration_value": "Has conc. value",
    "species_name": "Species identity",
    "concentration_numeric_mean": "Conc. (mean)",
    "molar_equivalent_mean": "Molar equiv.",
}


def feat_label(x: str) -> str:
    return FEAT_LABEL.get(x, x.replace("_", " "))


def grouped_metric_panel(ax, df, metric, title, xlabel, xlim):
    rows = []
    for fs in FS_ORDER:
        sub = df[df["feature_set"] == fs]
        for _, r in sub.iterrows():
            rows.append((fs, r["model"], r[metric]))
    # order: feature set blocks, models within
    y = 0
    yticks, ylabels = [], []
    for fs in FS_ORDER:
        sub = [r for r in rows if r[0] == fs]
        # keep a stable model order
        order = ["logistic_regression", "ridge", "random_forest", "gradient_boosting"]
        sub = sorted(sub, key=lambda r: order.index(r[1]) if r[1] in order else 9)
        for fs_, model, val in sub:
            color = GREY if fs == "protein_only" else MODEL_COLOR.get(model, TEAL)
            ax.barh(y, val, height=0.74, color=color, edgecolor="white", linewidth=0.4)
            y += 1
        yticks.append(y - len(sub) / 2.0 - 0.5 + 0.5)
        ylabels.append(FS_LABEL[fs])
        y += 0.6
    ax.set_yticks(yticks, ylabels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4)
    ax.tick_params(axis="x", labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def importance_panel(ax, df, color, title):
    top = df.sort_values("importance_mean", ascending=False).head(8).iloc[::-1]
    y = range(len(top))
    ax.barh(list(y), top["importance_mean"], xerr=top["importance_std"],
            color=color, edgecolor="white", height=0.7,
            error_kw={"lw": 0.6, "capsize": 2, "ecolor": "#5B6670"})
    ax.set_yticks(list(y), [feat_label(f) for f in top["feature"]], fontsize=7)
    ax.set_xlabel("Permutation importance", fontsize=7.5)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4)
    ax.tick_params(axis="x", labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    cls = pd.read_csv(TBL / "classification_model_comparison.csv")
    reg = pd.read_csv(TBL / "regression_model_comparison.csv")
    cls_imp = pd.read_csv(TBL / "classification_feature_importance.csv")
    reg_imp = pd.read_csv(TBL / "regression_feature_importance.csv")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    grouped_metric_panel(ax_a, cls, "roc_auc_mean",
                         "Classification: structure-modelled species",
                         "Grouped CV ROC-AUC", (0.45, 1.0))
    ax_a.axvline(0.5, color="#C2C8CF", lw=0.8, ls="--", zorder=0)
    grouped_metric_panel(ax_b, reg, "r2_mean",
                         "Regression: solvent content",
                         "Grouped CV R虏", (0.0, 0.7))
    importance_panel(ax_c, cls_imp, TEAL, "Classification drivers")
    importance_panel(ax_d, reg_imp, BLUE, "Regression drivers")

    for ax, lab in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]:
        ax.text(-0.16, 1.06, lab, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", color=INK)

    fig.suptitle("Pooled cross-protein machine-learning signal check",
                 fontsize=9.5, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.92, bottom=0.085,
                        hspace=0.42, wspace=0.62)
    stem = OUT / "figureS10_pooled_signal_check_combined"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print("Wrote", stem.with_suffix(".png"))
    print("Wrote", stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()

