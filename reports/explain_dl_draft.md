# Methods and Results: Explainability of EEGNet and LDA (R1 draft, for report section 3.6 / RQ3)

> Draft 2026-11-21, data v2 (35 usable subjects), main task `tgt_std`, R5 checkpoints of 2026-11-20
> (`results/checkpoints/eegnet/`). Produced by `python scripts/05_explain_dl.py` (code: `src/eegproj/explain/dl.py`, figures: `src/eegproj/explain/plots.py`)
> and, for the LDA patterns, `python scripts/06_explain_lda.py` (code: `src/eegproj/explain/lda.py`).
> Control tasks: EEGNet trained on `dis_std` and `tgt_dis` with the R5 code (commit f865c97, same folds and
> hyperparameters) on a Kaggle Tesla T4 (2026-11-21), then explained with `python scripts/05_explain_dl.py --task <task>`.

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

**LDA activation patterns (protocol 9.4).** As a linear reference we refitted the classical `lda_tw` model (50 ms
bin means per channel, standardisation, shrinkage LDA with equal priors) on the same five cross-subject folds and
converted its weights into activation patterns (Haufe et al., 2014, eq. 7): a = Cov(z, s) / Var(s), where z are the
bin features of the training trials and s = w^T z is the LDA output, so that a is in microvolts per unit of output.
The refit reproduced the classical results exactly (mean balanced accuracy 0.8605 for `full` and 0.7910 for
`early`; mean AUC 0.9391 and 0.8799). Unlike EEGNet filters, LDA patterns live in a fixed feature space (channel by
50 ms bin), so they were averaged over the five folds. For each bin we computed the spatial correlation with the
P300 reference map and the same regional enrichment as for |IG|, using the absolute pattern |a|.

**Control tasks (protocol 2 and 9.5).** After the behavioural filter every target trial contains a right-hand button
press and no standard or distractor trial does, so a target classifier could use motor activity instead of the P300.
We therefore trained EEGNet, with the same code, folds and hyperparameters, on two control tasks: distractor vs
standard (`dis_std`: a rare class but no response in either class) and target vs distractor (`tgt_dis`: both classes
rare, only targets have a response). The models were explained with the same Integrated Gradients procedure, on the
correctly classified trials of the positive class (distractors for `dis_std`, targets for `tgt_dis`). The `dis_std`
model gives the attribution that the motor region receives when no response is involved; motor-region attribution
above this level in `tgt_std` or `tgt_dis` points to response-related activity.

## Results: RQ3

Numbers from `results/explain/` (runs of 2026-11-21 with the code of commit ac98248). Median RT of correct
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

**LDA patterns (Figure F5c, cross-subject).** The patterns were almost identical across folds (correlation between
the patterns of any two folds 0.98 or higher). In the `full` model, 71% of the absolute pattern lay after 450 ms,
and its topography matched the P300 reference map from 350 ms onwards (r = 0.92 for 250-450 ms, 0.99 for 450-600 ms,
0.95 for 600-800 ms). The P300 region peaked at 0.58 µV in the 500-550 ms bin and the largest channels after 450 ms
were Pz, CPz, CPP1h and CPP2h. The P300 region was enriched about twofold after 450 ms, whereas the left motor
region stayed below an average channel in every bin (enrichment 0.54 to 0.70).
The `early` model had the same centro-parietal pattern, growing towards the end of the window (0.95 µV at
400-450 ms).

| Window | Time bin | Share of \|pattern\| | r with P300 map | P300 enrichment | Motor enrichment |
|---|---|---|---|---|---|
| full | 0-250 ms | 0.07 | 0.50 | 0.96 | 0.57 |
| full | 250-450 ms | 0.22 | 0.92 | 1.94 | 0.54 |
| full | 450-600 ms | 0.35 | 0.99 | 2.32 | 0.70 |
| full | 600-800 ms | 0.37 | 0.95 | 2.09 | 0.61 |
| early | 0-250 ms | 0.23 | 0.54 | 1.07 | 0.52 |
| early | 250-450 ms | 0.77 | 0.92 | 2.01 | 0.54 |

Fold-mean pattern; per 50 ms bin: `results/explain/lda_pattern_summary_tgt_std.csv`.

**Control tasks: decoding.** Distractors were decoded from standards almost as well as targets, although neither
class involves a response, and the `early` window kept most of the performance. Target vs distractor was the
hardest task and lost the most in the `early` window, close to chance within subjects.

| Task | Setting | EEGNet full | EEGNet early | Drop full to early | Classical best, full (R4) |
|---|---|---|---|---|---|
| `tgt_std` | within | 0.941 | 0.875 | 0.066 | 0.949 (`xdawn_rg`) |
| `tgt_std` | cross | 0.924 | 0.830 | 0.094 | 0.872 (`xdawn_rg`) |
| `dis_std` | within | 0.866 | 0.824 | 0.042 | 0.857 (`xdawn_rg`) |
| `dis_std` | cross | 0.839 | 0.783 | 0.056 | 0.830 (`xdawn_rg`) |
| `tgt_dis` | within | 0.684 | 0.549 | 0.135 | 0.784 (`xdawn_rg`) |
| `tgt_dis` | cross | 0.723 | 0.620 | 0.103 | 0.715 (`xdawn_rg`) |

Median balanced accuracy over 35 subjects; classical values from `reports/r4_evaluation_review.md`.

**Control tasks: attributions (cross-subject, `full`).** With no response in either class (`dis_std`), the left
motor region received the attribution of an average channel after 250 ms (enrichment 0.96 to 1.01), while the P300
region was enriched (1.16 to 1.30). In both tasks where the positive class contains a button press, the motor region
rose above this baseline from 250 to 600 ms (1.27 to 1.30 at 250-450 ms, 1.15 at 450-600 ms). In `tgt_dis` it became
the most attributed region: its |IG| peaked at 456 ms, before the median response (536 ms), at 1.71 times the
average channel, and the strongest channels at 450-600 ms were FCC3h, FC1, FCC1h and C1, over the left
(contralateral) motor cortex. In `tgt_std` the motor region peaked at about the same time (464 ms, 1.31 times the
average channel) but the P300 region stayed at least as high (472 ms, 1.37 times) and remained enriched after the
response (600-800 ms: 1.26 vs 0.99).

| Task | 0-250 ms | 250-450 ms | 450-600 ms | 600-800 ms |
|---|---|---|---|---|
| `tgt_std` (response in target class) | 1.07 / 1.26 | 1.38 / 1.30 | 1.27 / 1.15 | 1.26 / 0.99 |
| `dis_std` (no response) | 1.10 / 0.76 | 1.16 / 1.01 | 1.30 / 0.96 | 1.15 / 1.01 |
| `tgt_dis` (response in target class) | 0.87 / 1.62 | 1.08 / 1.27 | 1.27 / 1.15 | 1.10 / 1.14 |

P300 / motor enrichment of |IG|, median over 35 subjects. Within-subject models showed the same direction with
smaller differences (450-800 ms, motor: 1.00 `tgt_std`, 1.01 `dis_std`, 1.03 `tgt_dis`). Full tables:
`results/explain/saliency_summary_<task>.csv`; figures `figures/F5b_eegnet_saliency_<setting>_<task>`.

**Interpretation.** Both analyses point to the P300 as the main basis of EEGNet's decisions. Its most important
spatial filters reproduce the centro-parietal P3b topography, and its attributions peak over the parietal midline at
about 450 to 500 ms, before most responses, with no motor-specific increase over the left central region after
the response (600-800 ms). Two
observations temper this conclusion. First, two left central channels (C1, FCC3h) received about twice the
attribution of an average channel in the cross-subject `full` model in every time bin, already at 0-250 ms. This is
not time-locked to the response and so is unlikely to be motor execution, but it may reflect early central ERP
components or motor preparation in fast trials (the lateralised readiness potential starts before the press, and a
quarter of responses occurred before 461 ms). Second, gradient-based attributions are not activation patterns:
channels can receive attribution because they help cancel noise rather than because they carry the signal.

The LDA patterns, which are activation patterns and do not have this problem, give the same answer for the
classical model: the signal it uses is the centro-parietal P3b, peaking at 500-550 ms, without a left-central
motor component (after 450 ms the pattern is 0.14 µV at C3 against 0.55 µV at Pz). Because an LDA pattern closely
follows the target-minus-standard difference in its features, this also shows that a late left-central difference
is small in these data, which makes a mainly motor strategy unlikely for either model.

The control tasks qualify this answer. Response-related activity is used: the left central attribution in
`tgt_std` is above the no-response baseline of `dis_std`, it peaks shortly before the median press, and when the
P300 differs less between the two classes (`tgt_dis`, two rare stimuli) EEGNet relies on the same left central
channels more than on the P300 region. Because this activity starts before the press, it is more likely motor
preparation (the lateralised readiness potential) than the movement itself, and part of it falls inside the
`early` window. For the main task, however, the P300 region carries at least as much attribution, the distractor
task without any response reaches 0.84 to 0.87 balanced accuracy, and the `early` window keeps 0.83 to 0.88. We
therefore conclude that the target vs standard decision rests mainly on the P300, with a smaller contribution
from motor preparation that cannot be removed by the `early` window alone.

## Limitations (for the Discussion)

- One explanation per fitted model. Filters are not matched across folds (different initialisations), so the
  spatial patterns are shown per fold rather than averaged.
- The ROIs and time bins are fixed a priori from protocol section 9.5 and `docs/02_dataset.md` (right-hand
  responses); the Wilcoxon tests are descriptive and uncorrected.
- Single-filter ablation underestimates the role of filters that are redundant with others.
- Only EEGNet (the DL-best model) was trained on the control tasks, on a different GPU (Kaggle Tesla T4) than the
  main-task models; code, folds and hyperparameters were identical. The control tasks were not tested statistically
  against the classical models.
- The motor ROI is a coarse proxy: attribution there before the response is consistent with motor preparation but
  could also include other central ERP components that differ between the classes.

## Reproduction

```bash
python scripts/00_get_data.py      # data v2
python scripts/04_get_dl_results.py   # R5 checkpoints + predictions from Drive (or retrain: scripts/04_train_dl.py)
python scripts/05_explain_dl.py    # results/explain/, figures/F5a_*, figures/F5b_*
python scripts/04_train_dl.py --models eegnet --tasks dis_std tgt_dis   # control tasks (81 min on a T4)
python scripts/05_explain_dl.py --task dis_std && python scripts/05_explain_dl.py --task tgt_dis
python scripts/06_explain_lda.py   # results/explain/lda_*, figures/F5c_lda_patterns
python scripts/07_plot_pipeline.py # figures/F1_pipeline
python -m unittest tests.test_r1_explain -v
```

## References

- Haufe, S., et al. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging.
  *NeuroImage*, 87, 96-110.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*.
- Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer
  interfaces. *Journal of Neural Engineering*, 15, 056013.
