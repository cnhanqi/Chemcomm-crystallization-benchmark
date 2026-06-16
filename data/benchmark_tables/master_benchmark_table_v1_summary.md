# Master Benchmark Table v1 Summary

This master table merges the main layers built so far:

- condition-level metadata
- additive-aware condition parsing
- condition-to-structure retention comparison
- output-side structural quality and packing metrics
- polymorph proxy summaries for the same protein/additive/retained-state

- total rows: 4716
- observed in structure: 1160
- not observed in structure: 3556

## Rows by protein

- `insulin`: 346
- `lysozyme`: 2019
- `proteinase k`: 199
- `ribonuclease`: 1318
- `trypsin`: 834

## Rows by additive group

- `acetate`: 1193
- `chloride`: 835
- `formate`: 74
- `glycerol`: 269
- `ionic_liquid_or_des`: 4
- `mpd`: 114
- `nitrate`: 107
- `peg`: 1330
- `sulfate`: 790

## Recommended use

- use this as the main table for figures and descriptive statistics
- for ML, this is the best starting point if each row is treated as `protein + condition entry + additive group`
- if you later want entry-level labels only, aggregate from this table back to `protein + pdb_id + crystal_id`