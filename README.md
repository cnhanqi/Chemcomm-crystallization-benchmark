# Wallis ChemComm Data and Code Release

This repository contains the curated data tables, figure source data, machine-learning outputs and Python scripts supporting the ChemComm manuscript on a chemical species-resolved benchmark for protein crystallization conditions and structure-observed non-polymer species.

The release is intentionally curated. It does not contain manuscript drafts, reviewer comments, local agent files, private working notes, cache directories or historical exploratory outputs that are not required to support the reported results.

## Data Source

The primary source data are public crystallographic records and PDBx/mmCIF metadata from the Protein Data Bank:

https://www.rcsb.org/

No new crystallographic structures were determined in this study. The processed tables in this release derive from PDB records for five recurrent model protein systems: lysozyme, ribonuclease systems, trypsin, insulin and proteinase K.

## Release Contents

- `data/benchmark_tables/`: processed PDB-derived benchmark tables linking crystallization-condition chemistry, modeled non-polymer species and structural outputs.
- `data/ml/`: model input tables used for species-resolved and protein-specific machine-learning analyses.
- `results/figure_assets/`: source CSV files used to generate manuscript figures.
- `results/ml/`: model comparison, feature-importance, out-of-fold prediction and screening-guide outputs used in the manuscript and supporting information.
- `results/figures/`: final figure files included for traceability.
- `scripts/`: selected scripts used to build benchmark layers, run ML analyses and generate final figures.
- `docs/benchmark_variable_definitions_v1.md`: variable definitions and interpretation notes.

## Key Tables

- `data/benchmark_tables/master_benchmark_table_v1.csv`: integrated benchmark table.
- `data/benchmark_tables/master_benchmark_table_v1_summary.md`: row counts and summary.
- `data/benchmark_tables/master_benchmark_labels_v1.md`: benchmark labels and task definitions.
- `data/ml/species_aware_screening_ml_v2.csv`: species-resolved ML input table.
- `data/ml/per_protein_species_retention_v1/model_input_table.csv`: protein-specific ML input table.

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

## Main Analysis Scripts

- `scripts/build_expanded_species_inventory.py`
- `scripts/build_species_analysis_v2.py`
- `scripts/build_species_aware_ml_table_v2.py`
- `scripts/build_species_downstream_assets_v3.py`
- `scripts/run_species_aware_ml_baselines_v2.py`
- `scripts/run_species_aware_ml_rigorous_v3.py`
- `scripts/run_per_protein_species_retention_fig4_v1.py`
- `scripts/run_figure4_ml_all_panels_v3.py`
- `scripts/make_figure1_benchmark_architecture_v3.py`
- `scripts/make_figure2_condition_landscape_v5.py`
- `scripts/make_figure3_ml_bcde_row_v1.py`
- `scripts/make_figureS10_pooled_combined.py`

## Data Availability Statement Template

Use the final DOI and GitHub URLs after release:

The data supporting this article are derived from publicly available crystallographic records in the Protein Data Bank (PDB; https://www.rcsb.org/) and the associated PDBx/mmCIF metadata. The PDB accession codes used in the analysis are listed in the supporting data tables. Processed benchmark tables, figure source data, machine-learning input tables, model-evaluation outputs and the Python scripts used to generate the analyses and figures are available at [repository name] at [full GitHub URL], with the archived version of record available at [full DOI URL]. The version of the code used for this study is [release tag]. Supporting data for the figures and tables are also included in the Supplementary Information. No new crystallographic structures were determined in this study.

## Suggested Citation

After archiving the GitHub release with Zenodo, cite the archive DOI in the manuscript data availability statement and, if required by the journal, in the reference list.

