# Master Benchmark Labels v1

## What this label set is for

This file defines practical labels for analysis and first-pass machine learning using `master_benchmark_table_v1.csv`.

The goal is to prefer labels that are:

- interpretable
- already present in the current dataset
- defensible for a data paper

## Label family A: retained-species labels

These are the strongest current labels for your solvent/ion project.

### A1. `observed_in_structure`

Type:
binary classification

Definition:
whether the condition species is also explicitly present as a non-polymer species in the deposited structure.

Why it matters:
this is the cleanest available proxy for retained or ordered solvent/ion association.

Limitation:
`no` does not prove complete absence from the crystallization environment.

### A2. `explicit_ligand_instance_count`

Type:
count / ordinal label

Definition:
instance count of the matched retained species in the deposited structure.

Why it matters:
captures richer information than a simple yes/no retention label.

Suggested bins:

- `0`
- `1`
- `2-3`
- `>=4`

## Label family B: packing and solvation outcome labels

### B1. `solvent_content_percent`

Type:
regression

Why it matters:
highly relevant to your solvent-focused framing; usable as a bulk packing/solvation output.

### B2. `matthews_density`

Type:
regression

Why it matters:
closely related to solvent content and crystal packing.

### B3. `solvent_content_bin`

Type:
ordinal classification

Suggested bins:

- `<35`
- `35-45`
- `45-55`
- `>55`

Why it matters:
good for first-pass interpretable analysis.

## Label family C: structural quality labels

### C1. `resolution_high_a`

Type:
regression

### C2. `high_resolution_bin`

Type:
binary or ordinal classification

Suggested bins:

- binary:
  `<=1.5 A` versus `>1.5 A`
- ordinal:
  `<=1.2`, `1.2-1.8`, `1.8-2.5`, `>2.5`

### C3. `rfree`

Type:
regression

Why it matters:
useful as a structure-quality readout, but secondary to solvent-content and retention labels for your paper.

## Label family D: crystal-form labels

These should be secondary labels.

### D1. `space_group_hm`

Type:
multi-class classification

Use:
descriptive analysis or restricted within-protein classification only.

Limitation:
too sparse and imbalanced to be the main benchmark target for the whole paper.

### D2. `space_group_diversity_proxy`

Type:
derived analysis label

Definition:
use the polymorph proxy summaries to describe whether a protein-additive-retained-state combination spans multiple space groups.

Use:
paper section or subgroup analysis, not the main benchmark target.

## Label family E: recommended paper-ready benchmark tasks

### Task 1

Predict `observed_in_structure` from:

- protein identity
- additive group
- concentration
- method
- pH
- temperature

Why:
best fit to your solvent/ion question.

### Task 2

Model `solvent_content_percent` from:

- protein identity
- additive group
- retained-state
- explicit instance count
- condition parameters

Why:
connects solvent/ion chemistry to bulk packing outcome.

### Task 3

Model `resolution_high_a` from the same feature set.

Why:
good secondary benchmark, but not the main scientific identity of the paper.

## Labels I would not make primary

- direct polymorph prediction across all proteins
- crystal size prediction
- strong-binding versus weak-binding labels

Reason:
the current dataset does not support these labels strongly enough without much heavier curation.

## Recommended order for the paper

1. `observed_in_structure`
2. `explicit_ligand_instance_count`
3. `solvent_content_percent`
4. `matthews_density`
5. `resolution_high_a`
6. `space_group_hm` only as a later focused section
