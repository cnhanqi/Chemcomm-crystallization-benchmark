from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ml_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def get_screening_features(df: pd.DataFrame) -> list[str]:
    return [
        "protein",
        "additive_group",
        "additive_class",
        "is_ion_like",
        "is_solvent_like",
        "is_polymeric_solvent_like",
        "is_ionic_liquid_or_des",
        "condition_concentration_values",
        "concentration_unit_simple",
        "has_concentration_value",
        "condition_row_count",
        "method_family",
        "p_h",
        "temp_k",
        "mentions_seeding_flag",
        "mentions_soaking_flag",
        "mentions_cryoprotectant_flag",
        "condition_groups_count",
    ]


def get_species_screening_features(df: pd.DataFrame) -> list[str]:
    return [
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
