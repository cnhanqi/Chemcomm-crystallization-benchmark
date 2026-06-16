#!/usr/bin/env python3
"""Build species-aware downstream assets for Figures 4-6.

This extends the v2 species tables into the later manuscript figures so that
metal cations and explicit molecular species are retained throughout the
condition -> retention -> outcome story.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "benchmark_tables"
ASSETS = ROOT / "results" / "figure_assets"

COND = TABLES / "pdb_core_model_proteins_condition_species_inventory_v2.csv"
EXPL = TABLES / "pdb_core_model_proteins_explicit_nonpolymer_inventory_v2.csv"
MASTER = TABLES / "master_benchmark_table_v1.csv"

OUT_ENTRY = TABLES / "pdb_core_model_proteins_species_entry_status_v3.csv"
OUT_OUTCOME = TABLES / "pdb_core_model_proteins_species_outcome_summary_v3.csv"
OUT_CONTRAST = TABLES / "pdb_core_model_proteins_species_contrast_summary_v3.csv"
OUT_SPACEGROUP = TABLES / "pdb_core_model_proteins_species_spacegroup_summary_v3.csv"
OUT_NOTES = TABLES / "pdb_core_model_proteins_species_downstream_notes_v3.md"

FIG4 = ASSETS / "figure4_species_outcomes_by_state_v3.csv"
FIG5A = ASSETS / "figure5a_species_matched_contrasts_v3.csv"
FIG5B = ASSETS / "figure5b_selected_species_systems_v3.csv"
FIG6 = ASSETS / "figure6_species_space_group_diversity_v3.csv"

SELECTED_SPECIES = [
    "sodium",
    "sulfate",
    "chloride",
    "calcium",
    "magnesium",
    "nitrate",
    "zinc",
    "ethylene glycol",
    "glycerol",
    "acetate",
]

SELECTED_SYSTEMS = [
    ("insulin", "zinc"),
    ("trypsin", "calcium"),
    ("proteinase k", "calcium"),
    ("ribonuclease", "magnesium"),
    ("ribonuclease", "calcium"),
    ("lysozyme", "sodium"),
    ("lysozyme", "chloride"),
    ("lysozyme", "nitrate"),
]

EXCLUDE_MAJOR = {"buffer", "buffer_or_additive"}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cond = pd.read_csv(COND)
    expl = pd.read_csv(EXPL)
    master = pd.read_csv(MASTER)
    return cond, expl, master


def build_master_entry(master: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "protein",
        "pdb_id",
        "crystal_id",
        "solvent_content_percent",
        "matthews_density",
        "resolution_high_a",
        "space_group_hm",
        "method_family",
        "p_h",
        "temp_k",
    ]
    entry = (
        master[keep_cols]
        .groupby(["protein", "pdb_id", "crystal_id"], as_index=False)
        .agg(
            solvent_content_percent=("solvent_content_percent", "first"),
            matthews_density=("matthews_density", "first"),
            resolution_high_a=("resolution_high_a", "first"),
            space_group_hm=("space_group_hm", "first"),
            method_family=("method_family", "first"),
            p_h=("p_h", "first"),
            temp_k=("temp_k", "first"),
        )
    )
    return entry


def build_entry_status(cond: pd.DataFrame, expl: pd.DataFrame, master_entry: pd.DataFrame) -> pd.DataFrame:
    cond_u = cond[
        [
            "protein",
            "pdb_id",
            "crystal_id",
            "species_name",
            "species_major_class",
            "species_minor_class",
        ]
    ].drop_duplicates()
    cond_u["condition_present"] = "yes"

    expl_u = expl[
        ["protein", "pdb_id", "species_name", "species_major_class", "species_minor_class"]
    ].drop_duplicates()
    expl_u["explicit_present"] = "yes"

    merged = cond_u.merge(
        expl_u,
        on=["protein", "pdb_id", "species_name", "species_major_class", "species_minor_class"],
        how="left",
    )
    merged["explicit_present"] = merged["explicit_present"].fillna("no")
    merged["observed_in_structure"] = merged["explicit_present"].map({"yes": "yes", "no": "no"})
    merged["entry_status"] = merged["observed_in_structure"].map(
        {"yes": "condition_and_explicit", "no": "condition_only"}
    )
    merged = merged.merge(master_entry, on=["protein", "pdb_id", "crystal_id"], how="left")
    return merged.sort_values(
        ["protein", "species_major_class", "species_name", "pdb_id", "crystal_id"]
    )


def build_outcome_summary(entry_df: pd.DataFrame) -> pd.DataFrame:
    cond_df = entry_df[~entry_df["species_major_class"].isin(EXCLUDE_MAJOR)].copy()
    summary = (
        cond_df.groupby(
            ["species_name", "species_major_class", "species_minor_class", "observed_in_structure"],
            as_index=False,
        )
        .agg(
            n_entries=("pdb_id", "size"),
            n_unique_entries=("pdb_id", "nunique"),
            n_solvent_content_percent=("solvent_content_percent", "count"),
            mean_solvent_content_percent=("solvent_content_percent", "mean"),
            median_solvent_content_percent=("solvent_content_percent", "median"),
            n_matthews_density=("matthews_density", "count"),
            mean_matthews_density=("matthews_density", "mean"),
            median_matthews_density=("matthews_density", "median"),
            n_resolution_high_a=("resolution_high_a", "count"),
            mean_resolution_high_a=("resolution_high_a", "mean"),
            median_resolution_high_a=("resolution_high_a", "median"),
            unique_space_groups=("space_group_hm", "nunique"),
        )
    )
    return summary.sort_values(
        ["n_entries", "n_unique_entries", "species_name"], ascending=[False, False, True]
    )


def build_contrast_summary(entry_df: pd.DataFrame) -> pd.DataFrame:
    cond_df = entry_df[~entry_df["species_major_class"].isin(EXCLUDE_MAJOR)].copy()
    summary = (
        cond_df.groupby(
            ["protein", "species_name", "species_major_class", "species_minor_class", "observed_in_structure"],
            as_index=False,
        )
        .agg(
            entries=("pdb_id", "size"),
            unique_entries=("pdb_id", "nunique"),
            space_groups=("space_group_hm", "nunique"),
            mean_solvent_content=("solvent_content_percent", "mean"),
            mean_matthews_density=("matthews_density", "mean"),
            mean_resolution=("resolution_high_a", "mean"),
            median_solvent_content=("solvent_content_percent", "median"),
            median_resolution=("resolution_high_a", "median"),
        )
    )
    piv = summary.pivot_table(
        index=["protein", "species_name", "species_major_class", "species_minor_class"],
        columns="observed_in_structure",
        values=[
            "entries",
            "unique_entries",
            "space_groups",
            "mean_solvent_content",
            "mean_matthews_density",
            "mean_resolution",
            "median_solvent_content",
            "median_resolution",
        ],
        aggfunc="first",
    )
    piv.columns = [f"{metric}_{state}" for metric, state in piv.columns]
    out = piv.reset_index()
    out["delta_mean_solvent_content"] = (
        out["mean_solvent_content_yes"] - out["mean_solvent_content_no"]
    )
    out["delta_mean_resolution"] = out["mean_resolution_yes"] - out["mean_resolution_no"]
    out["delta_space_groups"] = out["space_groups_yes"] - out["space_groups_no"]
    return out.sort_values(
        ["entries_yes", "entries_no"], ascending=[False, False], na_position="last"
    )


def build_spacegroup_summary(entry_df: pd.DataFrame) -> pd.DataFrame:
    cond_df = entry_df[~entry_df["species_major_class"].isin(EXCLUDE_MAJOR)].copy()
    summary = (
        cond_df.groupby(
            ["species_name", "species_major_class", "species_minor_class", "observed_in_structure"],
            as_index=False,
        )
        .agg(
            entries=("pdb_id", "size"),
            unique_entries=("pdb_id", "nunique"),
            unique_space_groups=("space_group_hm", "nunique"),
            mean_solvent_content=("solvent_content_percent", "mean"),
            mean_resolution=("resolution_high_a", "mean"),
        )
    )
    return summary.sort_values(["entries", "species_name"], ascending=[False, True])


def write_notes(entry_df: pd.DataFrame, outcome: pd.DataFrame, contrast: pd.DataFrame) -> None:
    lines = [
        "# Species Downstream Notes v3",
        "",
        "This pass extends the species-aware analysis into Figures 4-6.",
        "",
        "## What changed",
        "",
        "- Later figures are no longer limited to collapsed additive groups.",
        "- Crystal-aware species rows now propagate into structural-outcome summaries and matched contrasts.",
        "- Metal cations such as zinc, calcium, magnesium, sodium, and potassium remain visible after Figure 3.",
        "",
        "## Selected species retained for Figure 4-6",
        "",
    ]
    for species in SELECTED_SPECIES:
        sub = outcome[outcome["species_name"] == species]
        if len(sub) == 0:
            continue
        yes = sub[sub["observed_in_structure"] == "yes"]
        no = sub[sub["observed_in_structure"] == "no"]
        lines.append(
            f"- {species}: observed {int(yes['n_entries'].sum()) if len(yes) else 0}, condition-only {int(no['n_entries'].sum()) if len(no) else 0}"
        )
    lines += ["", "## Selected matched systems for Figure 5", ""]
    for protein, species in SELECTED_SYSTEMS:
        sub = contrast[(contrast["protein"] == protein) & (contrast["species_name"] == species)]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]
        lines.append(
            f"- {protein} | {species}: retained {int(row.get('entries_yes', 0) or 0)}, condition-only {int(row.get('entries_no', 0) or 0)}"
        )
    OUT_NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    cond, expl, master = load_inputs()
    master_entry = build_master_entry(master)
    entry_df = build_entry_status(cond, expl, master_entry)
    outcome = build_outcome_summary(entry_df)
    contrast = build_contrast_summary(entry_df)
    spacegroup = build_spacegroup_summary(entry_df)

    entry_df.to_csv(OUT_ENTRY, index=False)
    outcome.to_csv(OUT_OUTCOME, index=False)
    contrast.to_csv(OUT_CONTRAST, index=False)
    spacegroup.to_csv(OUT_SPACEGROUP, index=False)

    fig4 = outcome[
        outcome["species_name"].isin(SELECTED_SPECIES)
        & outcome["observed_in_structure"].isin(["yes", "no"])
    ].copy()
    fig4.to_csv(FIG4, index=False)

    fig5a = contrast[
        contrast["species_name"].isin(SELECTED_SPECIES)
        & (contrast["entries_yes"].fillna(0) >= 4)
        & (contrast["entries_no"].fillna(0) >= 4)
    ].copy()
    fig5a.to_csv(FIG5A, index=False)

    selected_rows = []
    for protein, species in SELECTED_SYSTEMS:
        sub = contrast[(contrast["protein"] == protein) & (contrast["species_name"] == species)]
        if len(sub):
            selected_rows.append(sub.iloc[0])
    fig5b = pd.DataFrame(selected_rows)
    fig5b.to_csv(FIG5B, index=False)

    fig6 = spacegroup[
        spacegroup["species_name"].isin(SELECTED_SPECIES)
        & spacegroup["observed_in_structure"].isin(["yes", "no"])
    ].copy()
    fig6.to_csv(FIG6, index=False)

    write_notes(entry_df, outcome, contrast)
    print(
        {
            "entry_rows_v3": len(entry_df),
            "outcome_rows_v3": len(outcome),
            "contrast_rows_v3": len(contrast),
            "spacegroup_rows_v3": len(spacegroup),
            "figure4_rows": len(fig4),
            "figure5a_rows": len(fig5a),
            "figure5b_rows": len(fig5b),
            "figure6_rows": len(fig6),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

