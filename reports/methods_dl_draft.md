# Methods – Deep Learning (R5 draft, for report §3.4)

> Draft 2026-11-20, data v2 (35 usable subjects), main task `tgt_std`. Results from the full run of
> `python scripts/04_train_dl.py` (2026-11-19/20, branch `feat/dl-r5` on top of `main` b20d9bcd — replace by the final
> commit hash after merge). Methods describe `src/eegproj/models/dl.py`, `src/eegproj/models/dl_runner.py` and
> `configs/config.yaml` → `models.dl`.

## 3.4 Deep learning models

**Input.** Deep networks received the same behaviourally correct, artefact-cleaned epochs as the classical
models (Section 3.1). For each binary task (main task: target vs standard), the whole epoch (−200 to 800 ms) was
resampled from 250 Hz to 128 Hz and only then cropped to the analysis window (`full`: 0–800 ms, 103 samples;
`early`: 0–450 ms, 58 samples), so that resampling edge effects did not fall inside the window. All 127 EEG
channels were used, in microvolts, without further filtering. Each channel was z-scored with the mean and standard
deviation computed over the trials and time points of the training data of the current fold only (excluding the
validation subset); the same statistics were applied to the validation and test trials.

**Architectures.** Two convolutional networks were implemented in PyTorch following the original publications,
with the same layer layout as the braindecode reference implementations:

- **EEGNet-8,2** (Lawhern et al., 2018): a temporal convolution (F1 = 8 filters, 0.5 s = 65 samples, "same"
  padding) followed by batch normalisation; a depthwise spatial convolution across all channels (D = 2 spatial
  filters per temporal filter, max-norm 1), batch normalisation, ELU, average pooling (1 × 4) and dropout (p = 0.25);
  a separable convolution (16-sample depthwise + pointwise, F2 = 16), batch normalisation, ELU, average pooling
  (1 × 8) and dropout; and a linear classifier (max-norm 0.25). The model has ~3.3 k parameters.
- **ShallowConvNet / ShallowFBCSPNet** (Schirrmeister et al., 2017): a temporal convolution (40 filters, 0.1 s =
  13 samples), a spatial convolution across all channels (40 filters), batch normalisation, squaring, average
  pooling (0.3 s = 38 samples, stride 0.06 s = 8 samples), logarithm, dropout (p = 0.5) and a linear classifier
  (~204 k parameters). Kernel and pooling lengths were specified in seconds so that the published values
  (25/75/15 samples at 250 Hz) carry over to 128 Hz.

Writing the networks out explicitly, rather than importing them, keeps every layer accessible for the
explainability analysis (EEGNet spatial filters, Integrated Gradients; Section 3.6).

**Training.** Networks were trained with Adam (learning rate 10⁻³, no weight decay), mini-batches of 64 trials and a
class-weighted cross-entropy loss (weights inversely proportional to class frequency in the training data, i.e.
"balanced"), which addresses the ~1:6 target/standard imbalance without resampling. Training ran for at most 200
epochs, with early stopping when the validation loss had not decreased by at least 10⁻⁴ for 20 consecutive epochs;
the weights with the lowest validation loss were restored. The validation set was always taken from the training
part of the fold and followed the structure of the test split: in the within-subject setting, the last 20% of the
trials of every training segment (so validation trials form contiguous stretches rather than being interleaved with
neighbouring training trials); in the cross-subject setting, 20% of the training subjects (seeded random choice per
fold), so early stopping rewarded generalisation to unseen people. The test fold was never used for any decision.
Hyper-parameters were fixed a priori from the original papers and not tuned. The model of fold *k* was initialised
with seed 42 + *k*; predictions were labelled positive when the softmax probability of the positive class exceeded
0.5.

**Evaluation.** Deep models used exactly the same splits as the classical models: within-subject 5-fold CV over five
contiguous segments of equal trial count in recording order (no shuffling), and cross-subject GroupKFold with 5 folds
over the 35 subjects. Equal segments were used instead of the blocks between the `S 10` rest breaks because those
blocks are very unequal: the last one contains only ~40–50 trials with 0–4 targets (none for sub-13), which would
leave a fold undefined or scored on a single target. Cross-subject folds were frozen in `results/cv_cross_subject_folds.csv` and shared with the classical pipeline. In the cross-subject
setting one network was trained per fold on all trials of the ~28 training subjects and evaluated separately on each
held-out subject. Metrics (balanced accuracy, ROC-AUC, target-class F1) were computed per subject and fold and
written in the common format (protocol §7), so the statistics of RQ2 treat classical and deep models identically.
Training ran on one NVIDIA GeForce RTX 4060 Laptop GPU (8 GB; PyTorch 2.15, CUDA 13.2); the complete run of both
networks over both settings and both windows took 4.6 h. Per-fold training logs (number of epochs, best epoch, run time) are stored in
`results/dl_runs/`.

## Results (for Section 4.2–4.3)

Per-subject scores (within: mean over the 5 folds; cross: score of the subject when held out), N = 35, target vs
standard. No missing values.

| Model | Setting | Window | Balanced acc. median | Balanced acc. mean ± SD | AUC mean ± SD | F1 target | Epochs (median) |
|---|---|---|---|---|---|---|---|
| EEGNet | within | full | **0.941** | 0.917 ± 0.073 | 0.966 ± 0.051 | 0.795 | 101 |
| EEGNet | within | early | 0.875 | 0.851 ± 0.102 | 0.914 ± 0.098 | 0.662 | 77 |
| EEGNet | cross | full | 0.924 | 0.900 ± 0.069 | 0.966 ± 0.038 | 0.745 | 62 |
| EEGNet | cross | early | 0.830 | 0.819 ± 0.086 | 0.904 ± 0.079 | 0.594 | 36 |
| ShallowConvNet | within | full | 0.901 | 0.860 ± 0.099 | 0.927 ± 0.089 | 0.717 | 45 |
| ShallowConvNet | within | early | 0.773 | 0.735 ± 0.145 | 0.808 ± 0.162 | 0.525 | 41 |
| ShallowConvNet | cross | full | 0.905 | 0.887 ± 0.080 | 0.964 ± 0.046 | 0.749 | 27 |
| ShallowConvNet | cross | early | 0.792 | 0.778 ± 0.102 | 0.869 ± 0.102 | 0.543 | 31 |

**EEGNet vs ShallowConvNet** (Wilcoxon signed-rank on per-subject balanced accuracy, two-sided, Holm over the 4
comparisons; Δ = median EEGNet − Shallow; r = matched-pairs rank-biserial):

| Setting | Window | Δ median | EEGNet better in | W | p (Holm) | r |
|---|---|---|---|---|---|---|
| within | full | +0.045 | 34/35 | 2 | 7.0 × 10⁻¹⁰ | 0.99 |
| within | early | +0.116 | 32/35 | 18 | 4.4 × 10⁻⁸ | 0.94 |
| cross | full | +0.008 | 21/35 | 208 | 0.081 | 0.34 |
| cross | early | +0.039 | 28/35 | 65 | 1.9 × 10⁻⁵ | 0.79 |

Key observations:

- **Best DL model: EEGNet** (highest median balanced accuracy in every setting × window) → this is the "DL-best" for
  the RQ2 comparison with the best classical model (protocol §8).
- EEGNet keeps most of its performance in the `early` window (median drop full → early: 0.051 within, 0.079 cross;
  only sub-07 below 0.6), so its decisions do not rely only on activity after the typical button press.
  ShallowConvNet drops more (≈ 0.10) and is near chance in the within/early setting for 10 subjects
  (sub-01, 06, 07, 10, 21, 24, 25, 28, 32, 38), where validation loss stopped improving almost at once (median best epoch 5 vs 24 for the other subjects).
- ShallowConvNet benefits strongly from pooling subjects (within → cross, full: 0.901 → 0.905 median, mean 0.860 → 0.887;
  AUC 0.927 → 0.964) and then matches EEGNet in cross/full (difference not significant). EEGNet, with ~3 k parameters,
  is already data-efficient within a single subject.
- Hardest subject for all models: sub-07 (many rejected epochs, ~3 targets per test fold).

## Points for the Discussion / limitations

- Within-subject training sets are small (~430 trials, ~60 targets per fold); the low-parameter EEGNet coped well,
  whereas ShallowConvNet (~204 k parameters) needed the pooled cross-subject data.
- ShallowConvNet was designed for oscillatory (band-power) features: squaring and log-pooling discard the polarity of
  slow ERP deflections such as the P300; this fits its weaker within/early results.
- The `full`–`early` gap (≈ 0.05–0.10) is consistent with either a late P3b (~400–600 ms in this task, R3) or motor
  activity after ~450 ms contributing to `full`; the saliency / spatial-filter analysis (R1) and control tasks are
  needed to separate the two. R1's analysis (`reports/explain_dl_draft.md`) favours the P3b: EEGNet attributions
  peak over the parietal midline at about 470 to 510 ms, before the median response, with no late left-central increase;
  the control tasks are still to be run.
- No hyper-parameter search was performed (fixed published settings), which is conservative for DL and keeps the
  comparison with the classical pipelines fair but may underestimate the best achievable DL performance.
- Validation on a contiguous tail per block is stricter than a random stratified split; on one pilot subject both gave
  similar balanced accuracy (0.72 vs 0.73), so the choice is unlikely to drive the results.
- Because all correct targets include a button press (and no standard does), the `early` window and the control tasks
  are required to argue that the networks rely on the P300 rather than on motor activity (protocol §3).

## Reproduction

```bash
python scripts/00_get_data.py          # data v2
python scripts/04_train_dl.py          # results/metrics/{eegnet,shallow}_{within,cross}_{full,early}_tgt_std.csv
python -m unittest tests.test_r5_dl -v
```

References: Lawhern V.J. et al. (2018) EEGNet. *J. Neural Eng.* 15:056013. Schirrmeister R.T. et al. (2017) Deep
learning with convolutional neural networks for EEG decoding and visualization. *Hum. Brain Mapp.* 38:5391–5420.
