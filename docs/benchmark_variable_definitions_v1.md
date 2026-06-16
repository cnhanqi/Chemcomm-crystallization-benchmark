# Benchmark Variable Definitions v1

This note defines the most important variables used in the Hank benchmark and manuscript drafts.

## Why this note matters

Several output variables in crystallography do not mean what they look like at first glance. In particular, `solvent_content_percent` is **not** the same thing as “how many explicit water molecules were modeled in the structure”.

## Condition-side variables

### `additive_group`

A benchmark-level normalized label used to group related crystallization-condition components into broad chemistry classes such as:

- nitrate
- sulfate
- chloride
- acetate
- formate
- glycerol
- MPD
- PEG
- ionic liquid or DES

This is a derived field created for analysis. It does not imply that all members of the same group behave identically.

### `condition_additive_names`

The additive names recognized from the crystallization-condition text.

### `condition_concentration_raws`

The original concentration-like text snippets parsed from the condition string, such as `0.2 M`, `10%`, or `5.7% w/v`.

### `condition_concentration_values` and `condition_concentration_units`

Heuristically parsed concentration numbers and units extracted from free text.

Important caveat:
these are useful structured approximations, not perfect ground truth, because PDB crystallization conditions are recorded as heterogeneous free text.

### `method_family`

A normalized crystallization-method label such as:

- vapor diffusion - hanging drop
- vapor diffusion - sitting drop
- batch
- evaporation
- microbatch

This is derived from the deposited method field and associated condition text.

## Structure-retention variables

### `observed_in_structure`

A derived binary field.

Meaning:
for a given benchmark row, whether the relevant additive group is also represented by an explicitly modeled non-polymer species in the deposited structure.

Interpretation:

- `yes` means the additive group was explicitly retained and modeled in the deposited structure
- `no` means it was not explicitly modeled as a matched non-polymer species

Important caveat:
`no` does **not** prove that the species was absent from the crystallization environment or absent from the crystal altogether. It only means that it was not deposited as an explicit matched non-polymer component.

### `explicit_ligand_ids` and `explicit_ligand_names`

The matched non-polymer component identifiers and names found in the deposited structure, for example `NO3`, `SO4`, `CL`, or `GOL`.

### `explicit_ligand_instance_count`

The number of matched ligand instances associated with the deposited structure record for the retained species.

Practical interpretation:
this gives a stronger structural signal than a simple yes/no label because it distinguishes one retained instance from several retained instances.

## Output-side structural variables

### `solvent_content_percent`

This field comes from the crystallographic percent solvent reported in PDB/mmCIF.

In crystallographic terms, it is the fraction of the crystal unit-cell volume interpreted as solvent rather than ordered macromolecular volume.

What it **does mean**:

- a bulk packing and solvation readout
- an estimate of how much of the crystal volume is not occupied by the ordered macromolecule
- a property of the crystal lattice and unit-cell contents

What it **does not mean**:

- it is not a direct count of modeled `HOH` atoms
- it is not “the percentage of water molecules in the coordinate file”
- it is not limited to strongly bound water only

Does it include water?

- yes, conceptually it mainly corresponds to solvent-filled volume, which in protein crystals is typically dominated by water-rich mother-liquor space
- but it is broader than “water only”, because it reflects solvent-accessible or solvent-occupied crystal volume rather than a strict explicit-water inventory
- therefore, in this benchmark it should be read as **unit-cell solvent fraction**, not as a count of explicit bound waters

Practical Hank interpretation:
this is a bulk crystal-packing/solvation output variable that is highly relevant to a `solvent + ion` framing.

### `matthews_density`

This is the Matthews coefficient reported in crystallographic metadata, typically expressed in `A^3/Da`.

Meaning:
it is the ratio of asymmetric-unit volume to molecular mass, and it is closely related to crystal packing efficiency and solvent content.

Interpretation in this project:

- lower values usually indicate tighter packing
- higher values usually indicate more open packing and, often, higher solvent content
- it is complementary to `solvent_content_percent`, not redundant with it

### `resolution_high_a`

The high-resolution limit of the deposited structure, reported in angstroms.

Interpretation:
smaller values correspond to higher nominal structural resolution.

### `rfree` and `rwork`

Standard crystallographic refinement-quality metrics.

Interpretation in this project:
these are secondary structure-quality readouts and are not the main mechanistic outputs, but they help contextualize whether retained or non-retained states are associated with differences in refined structure quality.

### `space_group_hm`

The Hermann-Mauguin space-group symbol.

Interpretation in this project:
this is useful for crystal-form and polymorph-sensitive analysis, but should be treated as a secondary endpoint rather than the primary benchmark target.

## Recommended reading of key combinations

### `observed_in_structure` + `solvent_content_percent`

Use this pair to ask whether explicit retention of a solvent/ion group is associated with different bulk packing or solvent fraction outcomes.

### `observed_in_structure` + `matthews_density`

Use this pair to ask whether retained species are associated with tighter or looser crystal packing.

### `observed_in_structure` + `resolution_high_a`

Use this pair to ask whether retained species are associated with different refined structural quality.

## Official source notes

The PDB/mmCIF item descriptions behind the two most important output variables are:

- `_exptl_crystal.density_percent_sol`
- `_exptl_crystal.density_Matthews`

For the Hank manuscript, the safest wording is:

- `solvent_content_percent` = crystallographic percent solvent of the crystal cell, not explicit water count
- `matthews_density` = Matthews coefficient, a packing-related volume-per-mass metric

## Source links

- wwPDB PDBx/mmCIF dictionary for percent solvent: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_exptl_crystal.density_percent_sol.html
- wwPDB PDBx/mmCIF dictionary for Matthews coefficient: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_exptl_crystal.density_Matthews.html

