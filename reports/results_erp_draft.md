# Results – ERP (draft for group review)

**Status:** provisional, recomputed from v2 during the R3 review of `b97a0684` plus uncommitted validation fixes. All 35 participants await R2 visual QC; the peak rule and coordinate provenance of sub-12/sub-24 also await review.

After subject exclusions and pipeline rejection, 35 of 42 participants contributed 20,600 behaviorally correct epochs: 15,744 standard, 2,477 target and 2,379 distractor. Group ERPs use equal participant weights after averaging trials within participants. Uncertainty bands represent SEM across participants; difference-wave SEM uses paired within-participant differences.

The grand-average target response showed a larger late positivity than the standard at Pz. At 500 ms, target, standard and distractor amplitudes were respectively 9.38 ± 0.51, 0.76 ± 0.27 and 7.11 ± 0.45 µV (mean ± SEM); the paired target-minus-standard amplitude was 8.61 ± 0.48 µV. Draft scalp maps showed a posterior/central positive distribution over 250–600 ms (88 recorded samples, 252–600 ms). Maps use name-aligned mean electrode positions in head coordinates. The log identifies 33 CapTrak montages and two BrainVision `.bvef` montages, requiring R2 confirmation.

An exploratory maximum-signed-voltage search in the proposed fixed 300–600 ms interval produced the following Pz candidate measurements across all 35 participants, including flagged observations:

| Condition | Candidate amplitude, median [IQR] (µV) | Candidate latency, median [IQR] (ms) |
|---|---|---|
| Target | 10.34 [9.03–12.60] | 484 [448–562] |
| Distractor | 7.91 [7.02–9.27] | 504 [440–540] |

Five target candidates and one distractor candidate at Pz required boundary-related review; sub-38 reached the 600 ms upper limit. No additional participants were excluded. These descriptive candidates do not establish P300 identity or statistical significance. Motor-response contributions remain unresolved; final interpretation requires QC and group approval of the measurement rule.

Source tables: `results/erp/`; draft F2 figures: `figures/` (generated locally, gitignored). No classification/F3 results are included.
