#!/usr/bin/env python3
"""Build an expanded solvent/ion/molecule inventory for the Hank benchmark.

This pass extends the earlier additive-centric tables by:
1. extracting a broader condition-side species inventory from condition text segments
2. fetching all explicit non-polymer entities from RCSB for the current entry set
3. summarizing condition-vs-structure species coverage for downstream analysis
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "benchmark_tables"

CONDITION_ENTRIES = TABLES / "pdb_core_model_proteins_condition_entries_v2.csv"
CONDITION_COMPONENTS = TABLES / "pdb_core_model_proteins_condition_components_v1.csv"

OUT_CONDITION = TABLES / "pdb_core_model_proteins_condition_species_inventory_v2.csv"
OUT_EXPLICIT = TABLES / "pdb_core_model_proteins_explicit_nonpolymer_inventory_v2.csv"
OUT_BRIDGE = TABLES / "pdb_core_model_proteins_condition_explicit_species_bridge_v2.csv"
OUT_COND_COUNTS = TABLES / "pdb_core_model_proteins_condition_species_counts_v2.csv"
OUT_EXPL_COUNTS = TABLES / "pdb_core_model_proteins_explicit_species_counts_v2.csv"
OUT_NOTES = TABLES / "pdb_core_model_proteins_expanded_species_notes_v2.md"


SPECIES_PATTERNS = [
    ("sodium", "ion", "cation", [r"\bsodium\b", r"\bnacl\b", r"\bnano3\b", r"\bna acetate\b", r"\bsodium acetate\b"]),
    ("potassium", "ion", "cation", [r"\bpotassium\b", r"\bkcl\b", r"\bkno3\b"]),
    ("lithium", "ion", "cation", [r"\blithium\b", r"\blicl\b", r"\blino3\b"]),
    ("ammonium", "ion", "cation", [r"\bammonium\b", r"\bnh4\b"]),
    ("magnesium", "ion", "cation", [r"\bmagnesium\b", r"\bmgcl2\b", r"\bmgso4\b", r"\bmagnesium chloride\b", r"\bmagnesium acetate\b"]),
    ("calcium", "ion", "cation", [r"\bcalcium\b", r"\bcacl2\b", r"\bcalcium chloride\b", r"\bcalcium acetate\b"]),
    ("zinc", "ion", "cation", [r"\bzinc\b", r"\bzinc acetate\b", r"\bzinc chloride\b"]),
    ("cobalt", "ion", "cation", [r"\bcobalt\b", r"\bcobalt acetate\b", r"\bcobalt chloride\b"]),
    ("manganese", "ion", "cation", [r"\bmanganese\b", r"\bmanganese chloride\b", r"\bmanganese sulfate\b"]),
    ("nickel", "ion", "cation", [r"\bnickel\b", r"\bnickel chloride\b", r"\bnickel sulfate\b"]),
    ("copper", "ion", "cation", [r"\bcopper\b", r"\bcopper chloride\b", r"\bcopper sulfate\b"]),
    ("cadmium", "ion", "cation", [r"\bcadmium\b", r"\bcadmium chloride\b"]),
    ("cesium", "ion", "cation", [r"\bcesium\b", r"\bcaesium\b", r"\bcscl\b"]),
    ("chloride", "ion", "anion", [r"\bchloride\b"]),
    ("bromide", "ion", "anion", [r"\bbromide\b"]),
    ("iodide", "ion", "anion", [r"\biodide\b", r"\biodine\b"]),
    ("nitrate", "ion", "anion", [r"\bnitrate\b", r"\bno3\b"]),
    ("sulfate", "ion", "anion", [r"\bsulfate\b", r"\bso4\b"]),
    ("phosphate", "ion", "anion", [r"\bphosphate\b", r"\bpo4\b"]),
    ("acetate", "ion", "anion", [r"\bacetate\b", r"\bact\b"]),
    ("formate", "ion", "anion", [r"\bformate\b", r"\bfmt\b"]),
    ("citrate", "organic_ion_or_buffer", "anion_or_buffer", [r"\bcitrate\b"]),
    ("tartrate", "organic_ion_or_buffer", "anion_or_buffer", [r"\btartrate\b"]),
    ("succinate", "organic_ion_or_buffer", "anion_or_buffer", [r"\bsuccinate\b"]),
    ("malate", "organic_ion_or_buffer", "anion_or_buffer", [r"\bmalate\b"]),
    ("cacodylate", "buffer", "buffer", [r"\bcacodylate\b"]),
    ("hepes", "buffer", "buffer", [r"\bhepes\b"]),
    ("tris", "buffer", "buffer", [r"\btris\b", r"\btris-hcl\b"]),
    ("mes", "buffer", "buffer", [r"\bmes\b"]),
    ("bis-tris", "buffer", "buffer", [r"\bbis-tris\b", r"\bbistris\b"]),
    ("imidazole", "buffer_or_additive", "organic_base", [r"\bimidazole\b"]),
    ("peg", "polymeric_solvent_or_precipitant", "polymer", [r"\bpeg\b", r"\bpolyethylene glycol\b"]),
    ("glycerol", "solvent_or_cosolvent", "polyol", [r"\bglycerol\b", r"\bgol\b"]),
    ("mpd", "solvent_or_cosolvent", "polyol", [r"\bmpd\b", r"\bmethylpentanediol\b", r"\b2-methyl-2,4-pentanediol\b"]),
    ("ethylene glycol", "solvent_or_cosolvent", "polyol", [r"\bethylene glycol\b", r"\bedo\b"]),
    ("2-propanol", "solvent_or_cosolvent", "alcohol", [r"\b2-propanol\b", r"\bisopropanol\b", r"\bpropanol\b"]),
    ("acetone", "solvent_or_cosolvent", "ketone", [r"\bacetone\b"]),
    ("ethanol", "solvent_or_cosolvent", "alcohol", [r"\bethanol\b"]),
    ("methanol", "solvent_or_cosolvent", "alcohol", [r"\bmethanol\b"]),
    ("dioxane", "solvent_or_cosolvent", "ether", [r"\bdioxane\b"]),
    ("dmf", "solvent_or_cosolvent", "amide", [r"\bdmf\b", r"\bdimethylformamide\b"]),
    ("phenol", "organic_additive", "aromatic", [r"\bphenol\b"]),
    ("cresol", "organic_additive", "aromatic", [r"\bcresol\b", r"\bm-cresol\b"]),
    ("dtt", "reducing_agent", "thiol", [r"\bdtt\b", r"\bdithiothreitol\b"]),
    ("chaps", "detergent", "detergent", [r"\bchaps\b"]),
    ("ionic liquid", "ionic_liquid_or_des", "ionic_liquid", [r"\bionic liquid\b", r"\bionic liquids\b"]),
    ("ethylammonium nitrate", "ionic_liquid_or_des", "ionic_liquid", [r"\bethylammonium nitrate\b", r"\bean\b"]),
    ("ethylammonium formate", "ionic_liquid_or_des", "ionic_liquid", [r"\bethylammonium formate\b"]),
]


EXPLICIT_RULES = [
    ("zinc", "ion", "cation", ["ZINC ION"], ["ZN"]),
    ("magnesium", "ion", "cation", ["MAGNESIUM ION"], ["MG"]),
    ("calcium", "ion", "cation", ["CALCIUM ION"], ["CA"]),
    ("sodium", "ion", "cation", ["SODIUM ION"], ["NA"]),
    ("potassium", "ion", "cation", ["POTASSIUM ION"], ["K"]),
    ("lithium", "ion", "cation", ["LITHIUM ION"], ["LI"]),
    ("ammonium", "ion", "cation", ["AMMONIUM"], ["NH4"]),
    ("chloride", "ion", "anion", ["CHLORIDE ION"], ["CL"]),
    ("bromide", "ion", "anion", ["BROMIDE ION"], ["BR"]),
    ("iodide", "ion", "anion", ["IODIDE ION"], ["IOD"]),
    ("nitrate", "ion", "anion", ["NITRATE ION"], ["NO3"]),
    ("sulfate", "ion", "anion", ["SULFATE ION"], ["SO4"]),
    ("phosphate", "ion", "anion", ["PHOSPHATE ION"], ["PO4"]),
    ("acetate", "ion", "anion", ["ACETATE ION"], ["ACT"]),
    ("formate", "ion", "anion", ["FORMATE ION"], ["FMT"]),
    ("citrate", "organic_ion_or_buffer", "anion_or_buffer", ["CITRATE"], ["FLC", "CIT"]),
    ("glycerol", "solvent_or_cosolvent", "polyol", ["GLYCEROL"], ["GOL"]),
    ("mpd", "solvent_or_cosolvent", "polyol", ["PENTANEDIOL"], ["MPD"]),
    ("ethylene glycol", "solvent_or_cosolvent", "polyol", ["ETHYLENE GLYCOL"], ["EDO"]),
    ("2-propanol", "solvent_or_cosolvent", "alcohol", ["ISOPROPANOL", "2-PROPANOL"], ["IPA"]),
    ("acetone", "solvent_or_cosolvent", "ketone", ["ACETONE"], ["ACTN"]),
    ("phenol", "organic_additive", "aromatic", ["PHENOL"], ["IPH"]),
    ("cresol", "organic_additive", "aromatic", ["CRESOL"], ["MCR"]),
]


def normalize_text(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split()).lower()


def match_condition_species(text: str) -> list[tuple[str, str, str]]:
    norm = normalize_text(text)
    matches: list[tuple[str, str, str]] = []
    for species, major, minor, patterns in SPECIES_PATTERNS:
        if any(re.search(pattern, norm, flags=re.IGNORECASE) for pattern in patterns):
            matches.append((species, major, minor))
    return matches


def classify_explicit(comp_id: str, comp_name: str) -> tuple[str, str, str]:
    upper_name = (comp_name or "").upper()
    upper_id = (comp_id or "").upper()
    for species, major, minor, name_hits, id_hits in EXPLICIT_RULES:
        if any(hit in upper_name for hit in name_hits) or upper_id in id_hits:
            return species, major, minor
    keyword_map = {
        "ZINC": ("zinc", "ion", "cation"),
        "MAGNESIUM": ("magnesium", "ion", "cation"),
        "CALCIUM": ("calcium", "ion", "cation"),
        "SODIUM": ("sodium", "ion", "cation"),
        "POTASSIUM": ("potassium", "ion", "cation"),
        "LITHIUM": ("lithium", "ion", "cation"),
        "AMMONIUM": ("ammonium", "ion", "cation"),
        "NICKEL": ("nickel", "ion", "cation"),
        "COPPER": ("copper", "ion", "cation"),
        "COBALT": ("cobalt", "ion", "cation"),
        "MANGANESE": ("manganese", "ion", "cation"),
        "CADMIUM": ("cadmium", "ion", "cation"),
        "CESIUM": ("cesium", "ion", "cation"),
        "CAESIUM": ("cesium", "ion", "cation"),
        "CHLORIDE": ("chloride", "ion", "anion"),
        "BROMIDE": ("bromide", "ion", "anion"),
        "IODIDE": ("iodide", "ion", "anion"),
        "NITRATE": ("nitrate", "ion", "anion"),
        "SULFATE": ("sulfate", "ion", "anion"),
        "PHOSPHATE": ("phosphate", "ion", "anion"),
        "ACETATE": ("acetate", "ion", "anion"),
        "FORMATE": ("formate", "ion", "anion"),
    }
    for keyword, result in keyword_map.items():
        if keyword in upper_name:
            return result
    if " ION" in upper_name:
        return comp_name.lower().replace(" ion", ""), "ion", "other_ion"
    return comp_name.lower(), "nonpolymer_molecule", "other_molecule"


def batched(items: Iterable[str], size: int) -> Iterable[list[str]]:
    chunk: list[str] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def build_condition_inventory() -> pd.DataFrame:
    components = pd.read_csv(CONDITION_COMPONENTS)
    rows = []
    for row in components.itertuples(index=False):
        matches = match_condition_species(getattr(row, "component_text", ""))
        if not matches:
            continue
        seen = set()
        for species, major, minor in matches:
            if species in seen:
                continue
            seen.add(species)
            rows.append(
                {
                    "protein": row.protein,
                    "pdb_id": row.pdb_id,
                    "crystal_id": row.crystal_id,
                    "segment_index": row.segment_index,
                    "species_name": species,
                    "species_major_class": major,
                    "species_minor_class": minor,
                    "component_text": row.component_text,
                    "all_concentration_raws": row.all_concentration_raws,
                    "all_concentration_values": row.all_concentration_values,
                    "all_concentration_units": row.all_concentration_units,
                }
            )
    df = pd.DataFrame(rows)
    df.sort_values(["protein", "pdb_id", "crystal_id", "species_major_class", "species_name"], inplace=True)
    return df


def fetch_explicit_inventory(entry_to_protein: dict[str, str]) -> pd.DataFrame:
    query = """
    query($ids:[String!]!){
      entries(entry_ids:$ids){
        rcsb_id
        nonpolymer_entities {
          nonpolymer_comp {
            chem_comp { id name type formula }
          }
        }
      }
    }
    """
    session = requests.Session()
    rows = []
    entry_ids = sorted(entry_to_protein)
    for i, chunk in enumerate(batched(entry_ids, 80), start=1):
        resp = session.post(
            "https://data.rcsb.org/graphql",
            json={"query": query, "variables": {"ids": chunk}},
            timeout=120,
        )
        resp.raise_for_status()
        payload = resp.json()
        for entry in payload.get("data", {}).get("entries", []) or []:
            pdb_id = entry.get("rcsb_id")
            protein = entry_to_protein.get(pdb_id, "")
            entities = entry.get("nonpolymer_entities") or []
            if not entities:
                continue
            counts = Counter()
            details: dict[str, dict[str, str]] = {}
            for ent in entities:
                chem = ((ent or {}).get("nonpolymer_comp") or {}).get("chem_comp") or {}
                comp_id = chem.get("id", "")
                comp_name = chem.get("name", "")
                formula = chem.get("formula", "")
                if not comp_id and not comp_name:
                    continue
                counts[(comp_id, comp_name, formula)] += 1
                details[(comp_id, comp_name, formula)] = {
                    "comp_id": comp_id,
                    "comp_name": comp_name,
                    "formula": formula,
                }
            for key, entity_count in counts.items():
                detail = details[key]
                species_name, major, minor = classify_explicit(detail["comp_id"], detail["comp_name"])
                rows.append(
                    {
                        "protein": protein,
                        "pdb_id": pdb_id,
                        "comp_id": detail["comp_id"],
                        "comp_name": detail["comp_name"],
                        "formula": detail["formula"],
                        "species_name": species_name,
                        "species_major_class": major,
                        "species_minor_class": minor,
                        "explicit_entity_count": entity_count,
                    }
                )
        if i % 10 == 0:
            print(f"Fetched {i * 80} entry slots")
        time.sleep(0.08)
    df = pd.DataFrame(rows)
    df.sort_values(["protein", "pdb_id", "species_major_class", "species_name", "comp_id"], inplace=True)
    return df


def build_bridge(condition_df: pd.DataFrame, explicit_df: pd.DataFrame) -> pd.DataFrame:
    cond = (
        condition_df.groupby(["protein", "species_name", "species_major_class", "species_minor_class"], as_index=False)
        .agg(
            condition_mentions=("pdb_id", "size"),
            condition_unique_entries=("pdb_id", "nunique"),
            condition_examples=("pdb_id", lambda s: "; ".join(pd.unique(s.astype(str))[:10])),
        )
    )
    expl = (
        explicit_df.groupby(["protein", "species_name", "species_major_class", "species_minor_class"], as_index=False)
        .agg(
            explicit_entity_rows=("pdb_id", "size"),
            explicit_unique_entries=("pdb_id", "nunique"),
            explicit_comp_ids=("comp_id", lambda s: "; ".join(pd.unique(s.astype(str))[:10])),
            explicit_comp_names=("comp_name", lambda s: "; ".join(pd.unique(s.astype(str))[:5])),
        )
    )
    bridge = cond.merge(
        expl,
        on=["protein", "species_name", "species_major_class", "species_minor_class"],
        how="outer",
    ).fillna("")
    return bridge.sort_values(
        ["protein", "species_major_class", "condition_unique_entries", "explicit_unique_entries", "species_name"],
        ascending=[True, True, False, False, True],
    )


def write_notes(condition_df: pd.DataFrame, explicit_df: pd.DataFrame, bridge_df: pd.DataFrame) -> None:
    cation_bridge = bridge_df[(bridge_df["species_minor_class"] == "cation")].copy()
    ion_rows = bridge_df[bridge_df["species_major_class"] == "ion"].copy()
    top_cond = (
        condition_df.groupby(["species_name", "species_major_class"], as_index=False)["pdb_id"]
        .nunique()
        .sort_values("pdb_id", ascending=False)
        .head(20)
    )
    top_exp = (
        explicit_df.groupby(["species_name", "species_major_class"], as_index=False)["pdb_id"]
        .nunique()
        .sort_values("pdb_id", ascending=False)
        .head(20)
    )

    lines = [
        "# Expanded Species Notes v2",
        "",
        "This pass extends the earlier additive-centric benchmark tables.",
        "",
        "## Why this was needed",
        "",
        "The first benchmark collapsed many salt conditions into a limited set of high-frequency additive groups",
        "such as chloride, sulfate, nitrate, acetate, PEG, glycerol, and MPD. That was useful for an initial",
        "condition-to-structure benchmark, but it partially hid metal cations and other counterions.",
        "",
        "Example: `zinc acetate` often contributed to an `acetate` row, and `magnesium chloride` often",
        "contributed to a `chloride` row, so the cation chemistry was underrepresented in the high-level analysis.",
        "",
        "## Output files",
        "",
        f"- `{OUT_CONDITION.name}`: expanded condition-side species inventory from condition-text segments",
        f"- `{OUT_EXPLICIT.name}`: explicit non-polymer inventory fetched from RCSB",
        f"- `{OUT_BRIDGE.name}`: condition-versus-structure species bridge for downstream analysis",
        "",
        "## Top condition-side species by unique entries",
        "",
    ]
    for row in top_cond.itertuples(index=False):
        lines.append(f"- {row.species_name} ({row.species_major_class}): {int(row.pdb_id)}")
    lines += ["", "## Top explicit structure species by unique entries", ""]
    for row in top_exp.itertuples(index=False):
        lines.append(f"- {row.species_name} ({row.species_major_class}): {int(row.pdb_id)}")
    lines += ["", "## Metal and simple cation species now visible in the dataset", ""]
    if len(cation_bridge):
        for row in cation_bridge.head(20).itertuples(index=False):
            lines.append(
                f"- {row.protein}: {row.species_name} | condition entries {row.condition_unique_entries or 0} | explicit entries {row.explicit_unique_entries or 0}"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "Condition-side species extraction is vocabulary-guided and can still be expanded.",
        "Explicit structure species are exact non-polymer entities returned by RCSB for the current entry set.",
        "For downstream analyses, explicit species coverage is the stronger layer; condition-side species provide",
        "the chemical context needed to test whether those species were listed, retained, both, or neither.",
    ]
    OUT_NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    entries = pd.read_csv(CONDITION_ENTRIES)
    entry_to_protein = dict(entries[["pdb_id", "protein"]].drop_duplicates().values.tolist())

    condition_df = build_condition_inventory()
    condition_df.to_csv(OUT_CONDITION, index=False)

    explicit_df = fetch_explicit_inventory(entry_to_protein)
    explicit_df.to_csv(OUT_EXPLICIT, index=False)

    cond_counts = (
        condition_df.groupby(["protein", "species_name", "species_major_class", "species_minor_class"], as_index=False)
        .agg(entry_count=("pdb_id", "nunique"), mention_count=("pdb_id", "size"))
        .sort_values(["protein", "entry_count", "mention_count", "species_name"], ascending=[True, False, False, True])
    )
    cond_counts.to_csv(OUT_COND_COUNTS, index=False)

    expl_counts = (
        explicit_df.groupby(["protein", "species_name", "species_major_class", "species_minor_class"], as_index=False)
        .agg(entry_count=("pdb_id", "nunique"), entity_row_count=("pdb_id", "size"))
        .sort_values(["protein", "entry_count", "entity_row_count", "species_name"], ascending=[True, False, False, True])
    )
    expl_counts.to_csv(OUT_EXPL_COUNTS, index=False)

    bridge_df = build_bridge(condition_df, explicit_df)
    bridge_df.to_csv(OUT_BRIDGE, index=False)

    write_notes(condition_df, explicit_df, bridge_df)
    print(json.dumps({
        "condition_rows": int(len(condition_df)),
        "explicit_rows": int(len(explicit_df)),
        "bridge_rows": int(len(bridge_df)),
        "unique_explicit_entries": int(explicit_df["pdb_id"].nunique()) if len(explicit_df) else 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


