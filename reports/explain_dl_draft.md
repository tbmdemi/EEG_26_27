# Methods and Results: Explainability of the DL models (R1 draft, for report section 3.6 / RQ3)

> Draft 2026-11-20, data v2 (35 usable subjects), task `tgt_std`, R5 checkpoints of 2026-11-20
> (`results/checkpoints/eegnet/`). Produced by `python scripts/05_explain_dl.py` (code: `src/eegproj/explain/dl.py`, figures: `src/eegproj/explain/plots.py`).
> Still missing for protocol section 9: LDA patterns (9.4, needs R4's `lda_tw`) and the control tasks `dis_std` / `tgt_dis`.

## 3.6 Model explanation

**Models and trials explained.** We explained the fitted EEGNet models of Section 3.4 (the best DL model) exactly as
they were evaluated: in the cross-subject setting, each subject was explained by the fold model for which it was a
test subject (5 models); in the within-subject setting, by the five models that had its trials in their test folds
(175 models). Before each analysis, the test-set decisions of every model were recomputed and matched with the
stored test predictions (largest absolute difference of the logit difference: 0.00011), which guarantees that the
same data, channel order and preprocessing were used. Attributions were computed on the behaviourally correct
target trials that the model classified correctly (cross/full: 2193 of 2477 targets; cross/early: 1946; within/full:
2234; within/early: 2060).

**EEGNet spatial patterns.** The depthwise convolution of EEGNet learns D = 2 spatial filters for each of the F1 = 8
temporal filters (16 per model). Because filter weights are not interpretable as scalp topographies (Haufe et al.,
2014), each filter w was converted into an activation pattern a = C w / (w^T C w), where C is the channel
covariance of the output of the temporal filter feeding w, estimated over all trials of the model's training
subjects. Patterns were multiplied by the channel standard deviations of the input standardisation to express them
in microvolts. The polarity of a pattern is arbitrary (it is absorbed by later layers); we displayed it with a
positive spatial correlation to the group target-minus-standard ERP map (250 to 600 ms, same window as Figure F2)
and report the absolute correlation |r|. The importance of a filter was measured on the fold's test subjects as the
drop in AUC when its output was replaced by its training mean (BatchNorm running mean), averaged over subjects.

**Integrated Gradients.** For each correctly classified target trial we computed Integrated Gradients (Sundararajan
et al., 2017) of the logit difference (target minus standard) with respect to the standardised input, with a zero
baseline (the training-set mean of every channel) and a 64-step midpoint Riemann sum. The completeness error (sum
of attributions compared with the change of the logit difference) was below 0.1% of the output change for every
trial. Attributions were averaged per subject (signed and absolute values) and then across subjects. Because the
signed attributions of neighbouring channels often have opposite signs (gradients behave like spatial filters),
maps and regional summaries use the absolute attribution |IG|.

**Regional and temporal summary (protocol 9.5).** To decide whether the decisions rely on the P300 or on motor
activity from the right-hand button press, we compared a parieto-central P300 region (Pz, CPz, POz, CP1, CP2, P1, P2)
with a left central motor region (C3, C1, C5, FC3, CP3) in four time bins (0-250, 250-450, 450-600 and 600-800 ms;
450 ms is the end of the `early` window). For each subject and bin we computed the enrichment of each region: its
share of the |IG| mass divided by its share of the channels (1 means as much as an average channel). Regions were
compared with a two-sided Wilcoxon signed-rank test over subjects (descriptive, not corrected for the number of
bins).

## Results: RQ3

Numbers from `results/explain/` (run of 2026-11-20, commit 17b72dd plus uncommitted R1 code). Median RT of correct
targets: 536 ms (10% of responses before 412 ms, 25% before 461 ms).

**Spatial patterns (Figure F5a, cross-subject).** In the `full` window, the most important spatial filter of every
fold had a centro-parietal positive pattern that resembled the target-minus-standard ERP map (|r| from 0.61 to 0.91,
median 0.83; positive peaks at CCP2h, PPO1h, PPO2h), and so did 9 of the 10 two most important filters (|r| of at
least 0.5). In the `early` window, the top filter of three folds peaked at Pz or CP1 with |r| of 0.90 or more, whereas
fold 0 relied most on a right parieto-occipital pattern (|r| = 0.03) and fold 3 on a left fronto-central one (FCC1h,
|r| = 0.53). Ablating a single filter reduced test AUC by at most 0.018 (sum over all 16 filters: 0.02 to 0.04 per
fold), and importance was not related to P300 similarity across all filters (Spearman rho = 0.19, p = 0.09 for
`full`; rho = -0.12, p = 0.29 for `early`). The networks thus spread the information over several redundant filters,
the most useful of which have a P300-like topography.

**Integrated Gradients (Figure F5b).** In the `full` models, attribution rose from about 250 ms and peaked at 495 ms
(cross) and 511 ms (within), that is, before the median button press. The signed attribution of the P300 region was
positive (evidence for "target") and peaked at 472 ms (cross) and 456 ms (within), when the target P3b is maximal in
the ERP. About half of the |IG| mass fell after 450 ms (median share 0.53 cross, 0.51 within), but in this late
part the P300 region carried more attribution than the left motor region (enrichment median 1.20 vs 1.05 in cross,
p = 0.002; 1.19 vs 1.00 in within, p = 0.010). Between 600 and 800 ms, after most button presses, the motor region
received no more than an average channel (0.99 cross, 1.00 within). In the `early` models, attribution grew towards
the end of the window (peak 441 ms cross, 402 ms within) and the P300 region was enriched from 250 to 450 ms (1.35 vs
1.23 for the motor region in cross, p = 0.003; 1.18 vs 1.13 in within, p = 0.74).

| Setting / window | Time bin | Share of \|IG\| | P300 enrichment | Motor enrichment | p (Wilcoxon) |
|---|---|---|---|---|---|
| cross / full | 0-250 ms | 0.18 | 1.07 [0.97, 1.16] | 1.26 [1.14, 1.40] | 0.001 |
| cross / full | 250-450 ms | 0.30 | 1.38 [1.14, 1.56] | 1.30 [1.19, 1.44] | 0.35 |
| cross / full | 450-800 ms | 0.53 | 1.20 [1.14, 1.45] | 1.05 [0.99, 1.21] | 0.002 |
| cross / early | 250-450 ms | 0.62 | 1.35 [1.22, 1.54] | 1.23 [1.08, 1.34] | 0.003 |
| within / full | 450-800 ms | 0.51 | 1.19 [1.11, 1.33] | 1.00 [0.93, 1.20] | 0.010 |
| within / early | 250-450 ms | 0.62 | 1.18 [1.03, 1.34] | 1.13 [0.98, 1.28] | 0.74 |

Median [IQR] over 35 subjects; an enrichment of 1 equals the attribution of an average channel. Full table:
`results/explain/saliency_summary_tgt_std.csv`.

**Interpretation.** Both analyses point to the P300 as the main basis of EEGNet's decisions. Its most important
spatial filters reproduce the centro-parietal P3b topography, and its attributions peak over the parietal midline at
about 450 to 500 ms, before most responses, with no late, motor-specific increase over the left central region. Two
observations temper this conclusion. First, two left central channels (C1, FCC3h) received about twice the
attribution of an average channel in the cross-subject `full` model in every time bin, already at 0-250 ms. This is
not time-locked to the response and so is unlikely to be motor execution, but it may reflect early central ERP
components or motor preparation in fast trials (the lateralised readiness potential starts before the press, and a
quarter of responses occurred before 461 ms). Second, gradient-based attributions are not activation patterns:
channels can receive attribution because they help cancel noise rather than because they carry the signal. The
control tasks (`dis_std`: no response in either class; `tgt_dis`: both classes rare) are needed to settle the motor
question, and LDA patterns of the classical model (protocol 9.4) will give a linear reference.

## Limitations (for the Discussion)

- One explanation per fitted model. Filters are not matched across folds (different initialisations), so the
  spatial patterns are shown per fold rather than averaged.
- The ROIs and time bins are fixed a priori from protocol section 9.5 and `docs/02_dataset.md` (right-hand
  responses); the Wilcoxon tests are descriptive and uncorrected.
- Single-filter ablation underestimates the role of filters that are redundant with others.

## Reproduction

```bash
python scripts/00_get_data.py      # data v2
python scripts/04_get_dl_results.py   # R5 checkpoints + predictions from Drive (or retrain: scripts/04_train_dl.py)
python scripts/05_explain_dl.py    # results/explain/, figures/F5a_*, figures/F5b_*
python -m unittest tests.test_r1_explain -v
```

## References

- Haufe, S., et al. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging.
  *NeuroImage*, 87, 96-110.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*.
- Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer
  interfaces. *Journal of Neural Engineering*, 15, 056013.
