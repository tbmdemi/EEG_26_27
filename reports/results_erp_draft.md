# Results – ERP (draft for group review)

**Status:** descriptive results use the released v2 cohort. R2 records summary QC review for all 35 participants and verifies montage provenance for sub-12/sub-24. The P300 reporting rule was approved on 2026-11-22. The excluded-IC example was regenerated from saved v2 ICA; the matched raw/clean PSD still needs the unavailable raw data.

After subject exclusions and pipeline rejection, 35 of 42 participants contributed 20,600 behaviorally correct epochs: 15,744 standard, 2,477 target and 2,379 distractor. Group ERPs use equal participant weights after averaging trials within participants. Uncertainty bands represent SEM across participants; difference-wave SEM uses paired within-participant differences.

The grand-average target response showed a larger late positivity than the standard at Pz. At 500 ms, target, standard and distractor amplitudes were respectively 9.38 µV (SEM 0.51), 0.76 µV (SEM 0.27) and 7.11 µV (SEM 0.45); the paired target-minus-standard amplitude was 8.61 µV (SEM 0.48). Draft scalp maps showed a posterior/central positive distribution over 250-600 ms (88 recorded samples, 252-600 ms). Maps use name-aligned mean electrode positions in head coordinates. The log identifies 33 CapTrak montages and two BrainVision `.bvef` montages; their source provenance is recorded in the R2 handoff.

An exploratory maximum-signed-voltage search in the proposed fixed 300–600 ms interval produced the following Pz candidate measurements across all 35 participants, including flagged observations:

| Condition | Candidate amplitude, median [IQR] (µV) | Candidate latency, median [IQR] (ms) |
|---|---|---|
| Target | 10.34 [9.03–12.60] | 484 [448–562] |
| Distractor | 7.91 [7.02–9.27] | 504 [440–540] |

Five target candidates and one distractor candidate at Pz required boundary-related review; sub-38 reached the 600 ms upper limit. No additional participants were excluded. These descriptive candidates do not establish P300 identity or statistical significance. Motor-response contributions remain unresolved. The approved reporting rule retains flagged candidates and reports their count.

Source tables: `results/erp/`; draft F2 figures: `figures/` (generated locally, gitignored). No classification/F3 results are included.
