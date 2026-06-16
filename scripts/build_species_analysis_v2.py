#!/usr/bin/env python3
"""Build v2 species-level analysis tables for the Hank benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "benchmark_tables"
ASSETS = ROOT / "results" / "figure_assets"

COND = TABLES / "pdb_core_model_proteins_condition_species_inventory_v2.csv"
EXPL = TABLES / "pdb_core_model_proteins_explicit_nonpolymer_inventory_v2.csv"
MASTER = TABLES / "master_benchmark_table_v1.csv"

OUT_ENTRY = TABLES / "pdb_core_model_proteins_species_entry_status_v2.csv"
OUT_RETENTION = TABLES / "pdb_core_model_proteins_species_retention_summary_v2.csv"
OUT_TOP = TABLES / "pdb_core_model_proteins_top_species_by_protein_v2.csv"
OUT_METAL = TABLES / "pdb_core_model_proteins_metal_cation_summary_v2.csv"
OUT_NOTES = TABLES / "pdb_core_model_proteins_species_analysis_notes_v2.md"

FIG_TOP = ASSETS / "figure2d_top_species_by_protein_v2.csv"
FIG_RETENTION = ASSETS / "figure3d_species_retention_summary_v2.csv"
FIG_METAL = ASSETS / "figure3e_metal_cation_summary_v2.csv"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cond = pd.read_csv(COND)
    expl = pd.read_csv(EXPL)
    master = pd.read_csv(MASTER, usecols=["protein", "pdb_id", "solvent_content_percent", "resolution_high_a"])
    return cond, expl, master


def build_entry_status(cond: pd.DataFrame, expl: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    cond_u = cond[["protein", "pdb_id", "species_name", "species_major_class", "species_minor_class"]].drop_duplicates().copy()
    cond_u["condition_present"] = "yes"
    expl_u = expl[["protein", "pdb_id", "species_name", "species_major_class", "species_minor_class"]].drop_duplicates().copy()
    expl_u["explicit_present"] = "yes"
    merged = cond_u.merge(
        expl_u,
        on=["protein", "pdb_id", "species_name", "species_major_class", "species_minor_class"],
        how="outer",
    )
    merged["condition_present"] = merged["condition_present"].fillna("no")
    merged["explicit_present"] = merged["explicit_present"].fillna("no")
    merged["entry_status"] = merged.apply(
        lambda r: "condition_and_explicit"
        if r["condition_present"] == "yes" and r["explicit_present"] == "yes"
        else "condition_only"
        if r["condition_present"] == "yes"
        else "explicit_only",
        axis=1,
    )
    merged = merged.merge(master.drop_duplicates(["protein", "pdb_id"]), on=["protein", "pdb_id"], how="left")
    return merged.sort_values(
        ["protein", "species_major_class", "species_name", "pdb_id"], ascending=[True, True, True, True]
    )


def build_retention_summary(entry_df: pd.DataFrame) -> pd.DataFrame:
    cond_rows = entry_df[entry_df["condition_present"] == "yes"].copy()
    summary = (
        cond_rows.groupby(["protein", "species_name", "species_major_class", "species_minor_class"], as_index=False)
        .agg(
            condition_entries=("pdb_id", "nunique"),
            retained_entries=("explicit_present", lambda s: int((s == "yes").sum())),
            condition_only_entries=("explicit_present", lambda s: int((s == "no").sum())),
            median_solvent_content=("solvent_content_percent", "median"),
            median_resolution=("resolution_high_a", "median"),
        )
    )
    summary["retention_rate"] = summary["retained_entries"] / summary["condition_entries"]
    return summary.sort_values(
        ["protein", "condition_entries", "retention_rate", "species_name"],
        ascending=[True, False, False, True],
    )


def build_top_species(entry_df: pd.DataFrame) -> pd.DataFrame:
    cond = entry_df[entry_df["condition_present"] == "yes"].copy()
    top = (
        cond.groupby(["protein", "species_name", "species_major_class", "species_minor_class"], as_index=False)
        .agg(
            condition_entries=("pdb_id", "nunique"),
            retained_entries=("explicit_present", lambda s: int((s == "yes").sum())),
        )
    )
    top["retention_rate"] = top["retained_entries"] / top["condition_entries"]
    top["protein_rank"] = (
        top.sort_values(["protein", "condition_entries", "retention_rate"], ascending=[True, False, False])
        .groupby("protein")
        .cumcount()
        + 1
    )
    return top.sort_values(["protein", "protein_rank", "species_name"])


def build_metal_summary(entry_df: pd.DataFrame) -> pd.DataFrame:
    metals = {"zinc", "magnesium", "calcium", "sodium", "potassium", "lithium", "ammonium", "copper", "nickel", "cobalt", "manganese", "cadmium", "cesium"}
    metal_df = entry_df[(entry_df["condition_present"] == "yes") & (entry_df["species_name"].isin(metals))].copy()
    summary = (
        metal_df.groupby(["protein", "species_name"], as_index=False)
        .agg(
            condition_entries=("pdb_id", "nunique"),
            retained_entries=("explicit_present", lambda s: int((s == "yes").sum())),
            median_solvent_content=("solvent_content_percent", "median"),
            median_resolution=("resolution_high_a", "median"),
        )
    )
    summary["retention_rate"] = summary["retained_entries"] / summary["condition_entries"]
    return summary.sort_values(["protein", "condition_entries", "retention_rate"], ascending=[True, False, False])


def write_notes(retention: pd.DataFrame, top: pd.DataFrame, metal: pd.DataFrame) -> None:
    top_lines = []
    for protein in top["protein"].drop_duplicates():
        subset = top[(top["protein"] == protein) & (top["protein_rank"] <= 8)]
        top_lines.append(f"### {protein}")
        for row in subset.itertuples(index=False):
            top_lines.append(
                f"- {row.species_name} ({row.species_major_class}): condition {int(row.condition_entries)}, retained {int(row.retained_entries)}, rate {row.retention_rate:.2f}"
            )
        top_lines.append("")

    metal_lines = []
    for protein in metal["protein"].drop_duplicates():
        subset = metal[metal["protein"] == protein].head(8)
        metal_lines.append(f"### {protein}")
        for row in subset.itertuples(index=False):
            metal_lines.append(
                f"- {row.species_name}: condition {int(row.condition_entries)}, retained {int(row.retained_entries)}, rate {row.retention_rate:.2f}"
            )
        metal_lines.append("")

    text = "\n".join(
        [
            "# Species Analysis Notes v2",
            "",
            "This file summarizes the species-level analysis layer added after the initial additive-group benchmark.",
            "",
            "## What changed",
            "",
            "- We now track species names directly rather than only collapsed additive groups.",
            "- Entry status distinguishes `condition_only`, `explicit_only`, and `condition_and_explicit`.",
            "- Metal cations and small molecules can now be discussed explicitly in downstream figures and writing.",
            "",
            "## Top species by protein",
            "",
            *top_lines,
            "## Metal cation summary",
            "",
            *metal_lines,
            "## Recommended figure updates",
            "",
            "- Keep the current additive-group figures for the main benchmark narrative.",
            "- Add a species-aware supplement or updated panel showing top species by protein.",
            "- Add a metal-cation-focused panel or table to capture zinc, magnesium, calcium, sodium, potassium, lithium, and ammonium behavior.",
            "- When discussing retention, use species-level wording for metal ions and small molecules instead of only additive-group labels.",
        ]
    )
    OUT_NOTES.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    cond, expl, master = load_data()
    entry_df = build_entry_status(cond, expl, master)
    retention = build_retention_summary(entry_df)
    top = build_top_species(entry_df)
    metal = build_metal_summary(entry_df)

    entry_df.to_csv(OUT_ENTRY, index=False)
    retention.to_csv(OUT_RETENTION, index=False)
    top.to_csv(OUT_TOP, index=False)
    metal.to_csv(OUT_METAL, index=False)

    top[top["protein_rank"] <= 12].to_csv(FIG_TOP, index=False)
    retention[(retention["condition_entries"] >= 10) & (~retention["species_major_class"].isin(["buffer"]))].to_csv(FIG_RETENTION, index=False)
    metal.to_csv(FIG_METAL, index=False)

    write_notes(retention, top, metal)
    print(
        {
            "entry_rows": len(entry_df),
            "retention_rows": len(retention),
            "top_rows": len(top),
            "metal_rows": len(metal),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


