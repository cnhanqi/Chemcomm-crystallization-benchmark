# Hank ChemComm Curated Minimal Release

This repository contains a reduced public package of summary data tables, figure source data, selected machine-learning outputs and representative non-plotting Python scripts supporting the ChemComm manuscript on a chemical species-resolved benchmark for protein crystallization conditions and structure-observed non-polymer species.

The release is intentionally reduced. It does not contain manuscript drafts, reviewer comments, local agent files, private working notes, cache directories, full row-level working benchmark tables, full model-input matrices, exhaustive species inventories, editable vector artwork, figure-generation code or one-command orchestration scripts that are not required to inspect the reported results.

## Data Source

The primary source data are public crystallographic records and PDBx/mmCIF metadata from the Protein Data Bank:

https://www.rcsb.org/

No new crystallographic structures were determined in this study. The processed tables in this release derive from PDB records for five recurrent model protein systems: lysozyme, ribonuclease systems, trypsin, insulin and proteinase K.

## Release Contents

- `data/benchmark_tables/`: summary-level PDB-derived benchmark tables linking crystallization-condition chemistry, modeled non-polymer species and structural outputs.
- `data/ml/`: schema-level metadata for the species-resolved model input table.
- `results/figure_assets/`: source CSV files used to generate manuscript figures.
- `results/ml/`: summary model-comparison and feature-contribution outputs used in the manuscript and supporting information.
- `results/figures/`: final PNG figure files included for traceability.
- `scripts/`: representative non-plotting scripts used to summarize benchmark layers.
- `docs/benchmark_variable_definitions_v1.md`: variable definitions and interpretation notes.

## Key Tables

- `data/benchmark_tables/master_benchmark_table_v1_summary.md`: row counts and benchmark summary.
- `data/benchmark_tables/master_benchmark_labels_v1.md`: benchmark labels and task definitions.
- `data/benchmark_tables/pdb_core_model_proteins_species_retention_summary_v2.csv`: summary structure-observed rates by species.
- `data/benchmark_tables/pdb_core_model_proteins_species_outcome_summary_v3.csv`: summary structural-outcome table.
- `data/benchmark_tables/pdb_core_model_proteins_top_species_by_protein_v2.csv`: summary protein-specific species ranking table.
- `results/figure_assets/`: main-figure and SI source tables at figure-summary level.

## Reproducibility

Create the Python environment with either:

```bash
conda env create -f environment.yml
```

or:

```bash
python -m pip install -r requirements.txt
```

The scripts were developed with Python 3.11. Paths inside some scripts may need to be adjusted from the original local workspace path to the cloned repository path before rerunning.

This reduced public package is intended for summary-level traceability of the manuscript figures and reported comparisons. It is not a one-command full rerun package, and figure-generation code is not included.

## Included Scripts

- `scripts/build_species_analysis_v2.py`

## Data Availability Statement Template

Use the final DOI and GitHub URLs after release:

The data supporting this article are derived from publicly available crystallographic records in the Protein Data Bank (PDB; https://www.rcsb.org/) and the associated PDBx/mmCIF metadata. The PDB accession codes used in the analysis are listed in the supporting data tables and Supplementary Information. Curated summary benchmark tables, figure source data, summary model-evaluation outputs and selected non-plotting Python scripts used to generate the analyses are available at Chemcomm-crystallization-benchmark at https://github.com/cnhanqi/Chemcomm-crystallization-benchmark, with the archived version of record available at https://doi.org/10.5281/zenodo.20711461. The version of the code used for this study is v1.0.2. Supporting data for the figures and tables are also included in the Supplementary Information. No new crystallographic structures were determined in this study.

## Suggested Citation

Please cite the archived release DOI in the manuscript data availability statement and, if required by the journal, in the reference list: https://doi.org/10.5281/zenodo.20711461.
