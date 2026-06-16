from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmark_tables"
OUT_DIR = ROOT / "data" / "ml"
OUT_CSV = OUT_DIR / "species_aware_screening_ml_v2.csv"
OUT_JSON = OUT_DIR / "species_aware_screening_ml_v2_metadata.json"
OUT_TASKS_MD = ROOT / "reports" / "drafts" / "species_aware_ml_task_assessment_v1.md"

ENTRY_STATUS = BENCH / "pdb_core_model_proteins_species_entry_status_v3.csv"
CONDITION_SPECIES = BENCH / "pdb_core_model_proteins_condition_species_inventory_v2.csv"
EXPLICIT_SPECIES = BENCH / "pdb_core_model_proteins_explicit_nonpolymer_inventory_v2.csv"

METAL_CATIONS = {
    "zinc",
    "calcium",
    "magnesium",
    "copper",
    "nickel",
    "cobalt",
    "manganese",
    "cadmium",
    "iron",
}
MONOVALENT_CATIONS = {
    "sodium",
    "potassium",
    "ammonium",
    "lithium",
    "cesium",
}
DIVALENT_CATIONS = {
    "zinc",
    "calcium",
    "magnesium",
    "copper",
    "nickel",
    "cobalt",
    "manganese",
    "cadmium",
}
ANION_SCREEN_SET = {
    "nitrate",
    "sulfate",
    "chloride",
    "acetate",
    "formate",
    "phosphate",
    "citrate",
    "iodide",
}


def _split_multi(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _parse_numeric_list(value: object) -> list[float]:
    out: list[float] = []
    for item in _split_multi(value):
        try:
            out.append(float(item))
        except ValueError:
            continue
    return out


def _normalize_unit(unit: str) -> str:
    unit = unit.strip().lower()
    mapping = {
        "m": "M",
        "mm": "mM",
        "mmolar": "mM",
        "mmol": "mM",
        "mmol/l": "mM",
        "mg/ml": "mg/mL",
        "mg/mls": "mg/mL",
        "mg/ml ": "mg/mL",
        "% v/v": "% v/v",
        "% w/v": "% w/v",
        "%": "%",
    }
    if unit in mapping:
        return mapping[unit]
    if unit == "m":
        return "M"
    if unit == "mM".lower():
        return "mM"
    return unit


def _unit_family(unit: str) -> str:
    normalized = _normalize_unit(unit)
    if normalized in {"M", "mM"}:
        return "molar"
    if normalized in {"%", "% v/v", "% w/v"}:
        return "percent"
    if normalized == "mg/mL":
        return "mass_per_volume"
    if normalized:
        return "other"
    return "missing"


def _to_molar(value: float, unit: str) -> float | None:
    normalized = _normalize_unit(unit)
    if normalized == "M":
        return value
    if normalized == "mM":
        return value / 1000.0
    return None


def _summarize_concentration(group: pd.DataFrame) -> pd.Series:
    raws: list[str] = []
    nums: list[float] = []
    units: list[str] = []
    molar_vals: list[float] = []
    for row in group.itertuples(index=False):
        raws.extend(_split_multi(getattr(row, "all_concentration_raws")))
        row_nums = _parse_numeric_list(getattr(row, "all_concentration_values"))
        row_units = [_normalize_unit(u) for u in _split_multi(getattr(row, "all_concentration_units"))]
        nums.extend(row_nums)
        units.extend(row_units)
        if row_nums and row_units:
            for value, unit in zip(row_nums, row_units):
                molar = _to_molar(value, unit)
                if molar is not None:
                    molar_vals.append(molar)

    unique_raws = list(dict.fromkeys(raws))
    unique_units = list(dict.fromkeys(units))
    unit_families = sorted({_unit_family(u) for u in unique_units if u})

    if len(unit_families) == 0:
        unit_family = "missing"
    elif len(unit_families) == 1:
        unit_family = unit_families[0]
    else:
        unit_family = "mixed"

    primary_unit = unique_units[0] if len(unique_units) == 1 else ("mixed" if unique_units else "missing")
    primary_raw = unique_raws[0] if len(unique_raws) == 1 else ("mixed" if unique_raws else "")

    if nums:
        num_min = float(np.min(nums))
        num_max = float(np.max(nums))
        num_mean = float(np.mean(nums))
    else:
        num_min = math.nan
        num_max = math.nan
        num_mean = math.nan

    if molar_vals:
        molar_min = float(np.min(molar_vals))
        molar_max = float(np.max(molar_vals))
        molar_mean = float(np.mean(molar_vals))
    else:
        molar_min = math.nan
        molar_max = math.nan
        molar_mean = math.nan

    if pd.notna(molar_mean):
        if molar_mean < 0.01:
            conc_bin = "trace_to_low"
        elif molar_mean < 0.1:
            conc_bin = "low_to_mid"
        elif molar_mean < 0.5:
            conc_bin = "mid"
        elif molar_mean < 1.5:
            conc_bin = "high"
        else:
            conc_bin = "very_high"
    elif pd.notna(num_mean):
        conc_bin = "nonmolar_numeric"
    else:
        conc_bin = "missing"

    return pd.Series(
        {
            "condition_segment_count": int(len(group)),
            "concentration_raw_primary": primary_raw,
            "concentration_raw_all": " | ".join(unique_raws[:10]),
            "concentration_value_count": int(len(nums)),
            "has_numeric_concentration": int(len(nums) > 0),
            "concentration_numeric_min": num_min,
            "concentration_numeric_max": num_max,
            "concentration_numeric_mean": num_mean,
            "concentration_unit_primary": primary_unit,
            "concentration_unit_family": unit_family,
            "concentration_units_all": "; ".join(unique_units),
            "molar_equivalent_min": molar_min,
            "molar_equivalent_max": molar_max,
            "molar_equivalent_mean": molar_mean,
            "concentration_bin": conc_bin,
        }
    )


def _species_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["species_name"] = out["species_name"].astype(str).str.lower()
    out["is_ion_like"] = out["species_major_class"].eq("ion").astype(int)
    out["is_anion"] = out["species_minor_class"].eq("anion").astype(int)
    out["is_cation"] = out["species_minor_class"].eq("cation").astype(int)
    out["is_metal_cation"] = out["species_name"].isin(METAL_CATIONS).astype(int)
    out["is_monovalent_cation"] = out["species_name"].isin(MONOVALENT_CATIONS).astype(int)
    out["is_divalent_cation"] = out["species_name"].isin(DIVALENT_CATIONS).astype(int)
    out["is_screen_relevant_anion"] = out["species_name"].isin(ANION_SCREEN_SET).astype(int)
    out["is_buffer_like"] = out["species_major_class"].isin({"buffer", "buffer_or_additive", "organic_ion_or_buffer"}).astype(int)
    out["is_solvent_like"] = out["species_major_class"].isin({"solvent_or_cosolvent", "ionic_liquid_or_des"}).astype(int)
    out["is_polymeric_precipitant"] = out["species_major_class"].eq("polymeric_solvent_or_precipitant").astype(int)
    out["is_polyol_like"] = out["species_minor_class"].eq("polyol").astype(int)
    out["is_ionic_liquid_or_des"] = out["species_major_class"].eq("ionic_liquid_or_des").astype(int)
    out["species_role_class"] = np.select(
        [
            out["is_metal_cation"].eq(1),
            out["is_cation"].eq(1),
            out["is_anion"].eq(1),
            out["is_polymeric_precipitant"].eq(1),
            out["is_polyol_like"].eq(1),
            out["is_solvent_like"].eq(1),
            out["is_buffer_like"].eq(1),
            out["is_ionic_liquid_or_des"].eq(1),
        ],
        [
            "metal_cation",
            "cation",
            "anion",
            "polymeric_precipitant",
            "polyol",
            "solvent_or_cosolvent",
            "buffer_like",
            "ionic_liquid_or_des",
        ],
        default="other",
    )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    status = pd.read_csv(ENTRY_STATUS)
    cond = pd.read_csv(CONDITION_SPECIES)
    explicit = pd.read_csv(EXPLICIT_SPECIES)

    status = _species_flags(status)

    cond_rows: list[dict] = []
    for keys, group in cond.groupby(["protein", "pdb_id", "crystal_id", "species_name"], sort=False):
        protein, pdb_id, crystal_id, species_name = keys
        row = _summarize_concentration(group).to_dict()
        row.update(
            {
                "protein": protein,
                "pdb_id": pdb_id,
                "crystal_id": crystal_id,
                "species_name": species_name,
            }
        )
        cond_rows.append(row)
    cond_agg = pd.DataFrame(cond_rows)

    explicit_agg = (
        explicit.groupby(["protein", "pdb_id", "species_name"], as_index=False)
        .agg(
            explicit_entity_count=("explicit_entity_count", "sum"),
            explicit_compound_count=("comp_id", "nunique"),
        )
    )
    explicit_agg["explicit_entity_present_label"] = (explicit_agg["explicit_entity_count"] > 0).astype(int)

    out = status.merge(cond_agg, on=["protein", "pdb_id", "crystal_id", "species_name"], how="left")
    out = out.merge(explicit_agg, on=["protein", "pdb_id", "species_name"], how="left")

    out["group_id"] = out[["protein", "pdb_id", "crystal_id"]].astype(str).agg("|".join, axis=1)
    out["species_group_id"] = out[["protein", "pdb_id", "crystal_id", "species_name"]].astype(str).agg("|".join, axis=1)
    out["observed_in_structure_label"] = out["observed_in_structure"].map({"yes": 1, "no": 0}).fillna(0).astype(int)
    out["explicit_entity_count"] = out["explicit_entity_count"].fillna(0).astype(int)
    out["explicit_compound_count"] = out["explicit_compound_count"].fillna(0).astype(int)
    out["explicit_entity_present_label"] = out["explicit_entity_present_label"].fillna(0).astype(int)
    out["retained_multicopy_label"] = (out["explicit_entity_count"] >= 2).astype(int)

    out["solvent_content_regime"] = pd.cut(
        out["solvent_content_percent"],
        bins=[-np.inf, 40, 55, np.inf],
        labels=["compact", "intermediate", "open"],
    ).astype("object")
    out["resolution_regime"] = pd.cut(
        out["resolution_high_a"],
        bins=[-np.inf, 1.8, 2.5, np.inf],
        labels=["high", "medium", "lower"],
    ).astype("object")

    id_columns = [
        "group_id",
        "species_group_id",
        "pdb_id",
        "crystal_id",
    ]

    feature_columns = [
        "protein",
        "species_name",
        "species_major_class",
        "species_minor_class",
        "species_role_class",
        "is_ion_like",
        "is_anion",
        "is_cation",
        "is_metal_cation",
        "is_monovalent_cation",
        "is_divalent_cation",
        "is_screen_relevant_anion",
        "is_buffer_like",
        "is_solvent_like",
        "is_polymeric_precipitant",
        "is_polyol_like",
        "is_ionic_liquid_or_des",
        "condition_segment_count",
        "has_numeric_concentration",
        "concentration_value_count",
        "concentration_numeric_min",
        "concentration_numeric_max",
        "concentration_numeric_mean",
        "concentration_unit_primary",
        "concentration_unit_family",
        "molar_equivalent_min",
        "molar_equivalent_max",
        "molar_equivalent_mean",
        "concentration_bin",
        "method_family",
        "p_h",
        "temp_k",
    ]

    label_columns = [
        "observed_in_structure",
        "observed_in_structure_label",
        "explicit_entity_present_label",
        "retained_multicopy_label",
        "explicit_entity_count",
        "explicit_compound_count",
        "solvent_content_percent",
        "solvent_content_regime",
        "matthews_density",
        "resolution_high_a",
        "resolution_regime",
        "space_group_hm",
    ]

    out_df = out[id_columns + feature_columns + label_columns].copy()
    out_df.to_csv(OUT_CSV, index=False)

    classification_targets = {
        "primary": [
            "observed_in_structure_label",
            "explicit_entity_present_label",
        ],
        "secondary": [
            "solvent_content_regime",
            "resolution_regime",
        ],
        "exploratory_only": [
            "retained_multicopy_label",
        ],
    }
    regression_targets = [
        "solvent_content_percent",
        "matthews_density",
        "resolution_high_a",
        "explicit_entity_count",
        "explicit_compound_count",
    ]
    ranking_targets = [
        "protein-conditioned species ranking by retained-state probability",
        "protein-conditioned concentration-bin ranking by predicted structural outcome",
    ]

    metadata = {
        "source_files": [str(ENTRY_STATUS), str(CONDITION_SPECIES), str(EXPLICIT_SPECIES)],
        "output_file": str(OUT_CSV),
        "row_count": int(len(out_df)),
        "unique_group_count": int(out_df["group_id"].nunique()),
        "unique_species_group_count": int(out_df["species_group_id"].nunique()),
        "id_columns": id_columns,
        "feature_columns": feature_columns,
        "label_columns": label_columns,
        "recommended_classification_targets": classification_targets,
        "recommended_regression_targets": regression_targets,
        "recommended_ranking_tasks": ranking_targets,
        "notes": [
            "species-aware rows are defined at protein+pdb_id+crystal_id+species_name",
            "condition concentration is aggregated from condition_species_inventory_v2",
            "explicit entity count is aggregated at protein+pdb_id+species_name and should be interpreted cautiously where crystal_id multiplicity exists",
        ],
    }
    OUT_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    coverage = {
        "observed_in_structure_label": int(out_df["observed_in_structure_label"].notna().sum()),
        "explicit_entity_present_label": int(out_df["explicit_entity_present_label"].notna().sum()),
        "retained_multicopy_label": int(out_df["retained_multicopy_label"].notna().sum()),
        "solvent_content_percent": int(out_df["solvent_content_percent"].notna().sum()),
        "solvent_content_regime": int(out_df["solvent_content_regime"].notna().sum()),
        "matthews_density": int(out_df["matthews_density"].notna().sum()),
        "resolution_high_a": int(out_df["resolution_high_a"].notna().sum()),
        "resolution_regime": int(out_df["resolution_regime"].notna().sum()),
        "explicit_entity_count": int(out_df["explicit_entity_count"].notna().sum()),
    }

    md_lines = [
        "# Species-Aware ML Task Assessment v1",
        "",
        f"- rows in `species_aware_screening_ml_v2.csv`: `{len(out_df)}`",
        f"- unique grouped entries: `{out_df['group_id'].nunique()}`",
        f"- unique species-aware grouped rows: `{out_df['species_group_id'].nunique()}`",
        "",
        "## Recommended Classification Tasks",
        "",
        "- `observed_in_structure_label`",
        "  best primary classification label because it directly reflects retained-versus-condition-only state.",
        "- `explicit_entity_present_label`",
        "  nearly equivalent to retained-state presence, useful as a cleaner explicit-entity label.",
        "- `solvent_content_regime`",
        "  useful if a reader wants regime-level prediction rather than continuous regression.",
        "- `resolution_regime`",
        "  secondary and noisier; useful for coarse-quality binning only.",
        "- `retained_multicopy_label`",
        "  exploratory only; currently too sparse for a stable standalone classifier.",
        "",
        "## Recommended Regression Tasks",
        "",
        "- `solvent_content_percent`",
        "  strongest continuous structural-outcome target for this project.",
        "- `matthews_density`",
        "  useful continuous packing descriptor closely related to solvent content but not identical.",
        "- `resolution_high_a`",
        "  useful but influenced by many non-chemical factors; should be treated as secondary.",
        "- `explicit_entity_count`",
        "  potentially useful count-regression target for retained-species extent.",
        "- `explicit_compound_count`",
        "  secondary count target reflecting retained chemical diversity.",
        "",
        "## Recommended Ranking / Recommendation Tasks",
        "",
        "- protein-conditioned ranking of `species_name` for retained-state likelihood.",
        "- protein-conditioned ranking of `concentration_bin` within a chosen species system.",
        "- joint desirability ranking that combines retained-state probability with solvent-content regime.",
        "",
        "## Coverage Snapshot",
        "",
    ]
    for key, value in coverage.items():
        md_lines.append(f"- `{key}` non-missing rows: `{value}`")

    md_lines.extend(
        [
            "",
            "## Bottom-line Assessment",
            "",
            "The best immediate classification task is `observed_in_structure_label`.",
            "The best immediate regression task is `solvent_content_percent`.",
            "`retained_multicopy_label` is currently too sparse to be a main classification endpoint.",
            "The most useful downstream experimental task is not generic pooled prediction, but protein-conditioned ranking of species and concentration bins.",
        ]
    )
    OUT_TASKS_MD.write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

