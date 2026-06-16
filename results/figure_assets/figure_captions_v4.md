# Figure Captions v4

## Main Text Figures

### Figure 1

**Figure 1. Benchmark construction workflow linking crystallization-condition composition, structure-observed species, and structural outputs.**

This schematic summarizes the benchmark architecture used throughout the study. Five recurrent model proteins were used to assemble a dataset in which crystallization-condition text, explicitly modeled non-polymer species, and structure-level outputs were integrated into a common analytical framework. Here, `crystallization-condition composition` means the chemical makeup of the deposited crystallization condition, and `structure-observed species` means species present in that condition and also modeled as non-polymer components in the deposited structure. The figure emphasizes that the benchmark is not a condition-only database: it explicitly links input chemistry to modeled structural species and to output metrics such as solvent content, Matthews density, resolution, and space group.

### Figure 2

**Figure 2. Protein-specific ion and additive landscape across the five model proteins.**

Panel A shows a protein-by-species heatmap of the highest-coverage non-buffer species in the crystallization-condition layer. Panel B summarizes method-family composition by protein, preserving the procedural context of the benchmark. Panel C shows metadata completeness for pH, temperature, seeding, and cryoprotectant annotations normalized within protein. Together, these panels show that the benchmark retains specific ions and additives such as sodium, sulfate, chloride, calcium, magnesium, zinc, glycerol, ethylene glycol, and citrate rather than collapsing all chemistry to broad additive groups.

### Figure 3

**Figure 3. Condition composition versus structure-observed species.**

Panel A compares the most frequent condition species split into `condition only` versus `condition and explicit` states, highlighting which species are common in cocktails but rarely modeled and which are frequently modeled in the deposited structure. Panel B summarizes species-level structure-observed efficiency, with point size proportional to the number of condition entries. Panel C is a dedicated metal-cation panel in which point size reflects condition-entry count and color encodes the structure-observed rate across proteins. This figure makes visible metal- and ion-specific behavior that is blurred in additive-group-only summaries, including strong zinc signals in insulin, strong calcium signals in trypsin and proteinase K, and intermediate magnesium signals in ribonuclease.

### Figure 4

**Figure 4. Structural outcomes for structure-observed versus condition-only ions and molecules.**

Panels A-C compare median solvent content, Matthews density, and resolution for selected species that occur in both structure-observed and condition-only states. Lines connect the two states for each species, and point size reflects the number of contributing entries. The figure shows that the output-side structural layer is not only an anion story: sodium, calcium, magnesium, zinc, chloride, sulfate, nitrate, acetate, ethylene glycol, and glycerol remain directly interpretable in relation to packing-related outcomes.

### Figure 5

**Figure 5. Matched protein-species contrasts in selected ion-sensitive systems.**

Panels A and B compare solvent-content and resolution contrasts within matched protein-species systems, while Panel C compares state-wise space-group diversity in the same systems. The selected systems include insulin-zinc, trypsin-calcium, proteinase K-calcium, ribonuclease-magnesium, ribonuclease-calcium, lysozyme-sodium, lysozyme-chloride, and lysozyme-nitrate. By holding system identity fixed, this figure shows that protein-specific ion behavior remains interpretable after moving beyond global pooled summaries.

### Figure 6

**Figure 6. Protein-specific top species ranking across the five model proteins.**

Each panel shows the highest-ranked species for one protein using a protein-specific ranking baseline. Bars represent the combined ranking score, which prioritizes species with strong predicted modeled-structure behavior and favorable predicted solvent-content regime. Support counts are shown at the bar ends. The figure translates the benchmark from retrospective chemistry mining into first-pass screening guidance, showing that the most promising species differ substantially across lysozyme, ribonuclease, trypsin, insulin, and proteinase K.

### Figure 7

**Figure 7. Concentration-bin ranking for the five case systems.**

Each panel shows the highest-ranked concentration bins for one targeted protein-species system: insulin-zinc, trypsin-calcium, ribonuclease-magnesium, lysozyme-nitrate, and proteinase K-calcium. Bars represent the combined ranking score, with support counts and approximate molar-equivalent values annotated where available. This figure extends the recommendation logic beyond species identity alone and shows that concentration-window prioritization is feasible once protein context and species identity are fixed.

## Supporting Information Figures

### Figure S1

**Figure S1. Space-group diversity summary at the level of specific ions and additives.**

This figure compares unique space-group diversity for the selected species set in structure-observed versus condition-only states, with point size proportional to entry count. The analysis is informative for polymorph-sensitive interpretation, but it is presented as supporting information because it is less central than the condition-to-structure-to-ranking progression emphasized in the main text.

### Figure S2

**Figure S2. Protein-specific drivers of modeled-structure ranking.**

This heatmap summarizes protein-specific permutation-importance values from the protein-specific ranking models. The figure highlights which variables dominate ranking within each protein context, including species identity, method family, pH, temperature, concentration bin, and molar-equivalent features. It is placed in supporting information because it supports interpretation and later SHAP-style analysis rather than carrying the main narrative on its own. This is therefore a protein-specific driver summary prepared for later SHAP-style interpretation, not a SHAP result itself.
