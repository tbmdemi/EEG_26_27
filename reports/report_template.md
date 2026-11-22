# Single-trial Classification of Frequent versus Target Stimuli in a Visual Oddball EEG Paradigm: Classical Machine Learning versus Deep Learning

**Group:** <name> — **Members:** <names, student IDs>
**Course:** Neuroimaging: EEG — **Date:** <date>

---

## Abstract (≤ 200 words)
Background · Data (Nencki-Symfonia, N = …) · Methods (preprocessing, models, evaluation) · Main results (best balanced accuracy / AUC; classical vs DL; explainability) · Conclusion.

## 1. Introduction
- Oddball paradigm and the P300 component (latency, topography, functional interpretation).
- Single-trial ERP classification: why it is hard (low SNR, class imbalance).
- Classical approaches (shrinkage LDA, xDAWN + Riemannian geometry) vs deep learning (EEGNet, ShallowConvNet).
- Research questions RQ1–RQ3.

## 2. Data
- Participants, recording setup, task design, event proportions (cite Dzianok et al., 2022; DOI 10.5524/100990).
- Behavioural scoring: only correct trials kept (target answered 100–1400 ms after onset; standard/distractor without response); behavioural summary (hit rate, false alarms, RT) vs the paper.
- Subjects excluded and why: a priori sub-02/14/17 (misunderstood task, per the paper) + pipeline criteria; flagged subjects; epochs retained per class (table from `results/preproc_log.csv`).

## 3. Methods
### 3.1 Preprocessing

The team-provided preprocessing release `preproc_v2_20261118.zip` was used, with archive integrity verified by SHA-256. All 42 participants were considered for the oddball task. Participants sub-02, sub-14 and sub-17 were excluded a priori because they misunderstood the response instructions. Behavioural correctness was determined from the BIDS `event_type` and onset columns: targets required a response 100-1400 ms after stimulus onset, whereas standards and distractors required no response within that interval. Only correct trials were retained, and the first target-response latency was stored in epoch metadata. Participants with fewer than 40 correct targets were ineligible; hit rate below 80% was a review flag rather than an exclusion criterion.

The 127 recorded EEG channels were used without adding a fictitious FCz channel. Subject-specific CapTrak electrode positions were used where available. In the retained cohort, sub-12 and sub-24 used the corresponding subject-specific BrainVision `.bvef` files, read in head coordinates with MNE's default head-size normalization. These coordinates were independently matched to the released FIF files. Non-EEG channels were excluded. Bad channels were detected using robust z-scores of log channel standard deviation (absolute z > 5) and a flat-channel threshold of 0.01 microvolt; detected channels were interpolated using spherical splines before average referencing. Recorded `New Segment` joins after time zero were marked BAD with 0.5 s padding on each side, preventing affected data from entering filtering segments, ICA fitting and accepted epochs.

Signals were filtered at 0.1-40 Hz using zero-phase FIR filters and re-referenced to the common average. No separate 50-Hz notch was applied in the released pipeline because the notch frequency lay outside the 40-Hz passband. Extended Infomax ICA was fitted to a copy high-pass filtered at 1 Hz, retaining principal components explaining 99% of variance, with decimation by four, seed 42, a maximum of 512 iterations and a weight-change threshold of 1e-7. ICA was fitted to each participant's complete unlabelled recording. Components assigned eye-blink or muscle-artifact probabilities of at least 0.8 by ICLabel were excluded, and the decomposition was applied to the 0.1-Hz data. ICA on the complete recording is an unsupervised transductive preprocessing choice; ICLabel's training band differs from the 1-40 Hz input and is a limitation.

After ICA, data and event samples were resampled to 250 Hz. Epochs extended from -200 to 800 ms relative to stimulus onset and were baseline corrected over -200 to 0 ms, yielding 251 samples per epoch. Epochs exceeding 150 microvolts peak-to-peak in any EEG channel or overlapping BAD intervals were rejected. Participants retaining fewer than 50% of their behaviourally correct target trials were excluded. All three stimulus classes were saved, together with participant, original onset, block and target RT metadata, to support the main and control tasks.

The released cohort contains 35 participants and 20,600 epochs (15,744 standard, 2,477 target and 2,379 distractor). Subject-level and stage-wise accounting is provided in [the R2 subject/epoch table](r2_subject_epoch_table.md). Reloaded FIFs passed automatic checks of dimensions, finite samples/electrode positions, sampling rate, baseline, average reference, amplitude, class counts and metadata. Retained events and behavioural summaries were matched to source event tables for all relevant participants. Summary amplitude maps, Pz averages, retained-epoch spectra and representative high-amplitude-channel traces were visually inspected for all 35 participants; this does not constitute exhaustive interactive inspection of every raw channel. Residual channel noise in sub-10/OI1h and sub-13/AFF1h triggered separate candidate reprocessing and prevents describing the released data as artifact-free. Candidate results are not silently substituted for the versioned group release.

### 3.2 Classification tasks and time windows
Main task (target vs standard); control tasks (distractor vs standard, target vs distractor); `full` vs `early` windows and the motor-response rationale.
### 3.3 Classical models
`lda_tw`, `xdawn_rg` (+ optional) — features and hyperparameters.
### 3.4 Deep learning models
Architectures, input normalisation, training (optimizer, early stopping, class weighting).
### 3.5 Evaluation
Within-subject blocked CV (5 blocks); cross-subject GroupKFold (5 folds by subject); leakage safeguards; metrics; Wilcoxon signed-rank across subjects (one value per subject) with Holm correction.
### 3.6 Explainability
EEGNet spatial filters, Integrated Gradients, LDA activation patterns (Haufe et al., 2014).

## 4. Results
### 4.1 ERPs (Fig. F2)
### 4.2 Classification performance — RQ1 (Table 1, Fig. F3)
| Model | Family | Setting | Window | Balanced acc. (mean ± SD) | AUC (mean ± SD) |
|---|---|---|---|---|---|
### 4.3 Classical vs deep learning — RQ2 (Fig. F4, statistics)
### 4.4 What did the models learn? — RQ3 (Fig. F5, control tasks)

We explained the fitted EEGNet models exactly as they were evaluated (Section 3.6) and used the activation patterns of the classical `lda_tw` model as a linear reference. The question was whether the decisions rest on the P300, a centro-parietal positivity that peaked at 484 ms at Pz in the grand-average target ERP (Section 4.1), or on activity related to the right-hand button press, which every correct target trial and no standard trial contains (median correct RT 536 ms; 25% of responses before 461 ms). Regional summaries compare a P300 region (Pz, CPz, POz, CP1, CP2, P1, P2) with a left central motor region (C3, C1, C5, FC3, CP3). Enrichment is a region's share of the absolute attribution divided by its share of the channels, so a value of 1 equals the attribution of an average channel.

**Spatial filters (Fig. F5a).** In the cross-subject `full` models, the most important spatial filter of every fold had a positive centro-parietal activation pattern that resembled the target-minus-standard ERP map (|r| = 0.61 to 0.91, median 0.83; peaks at CCP2h, PPO1h and PPO2h). Nine of the ten filters ranked first or second across the five folds had |r| of at least 0.5. Removing any single filter reduced test AUC by at most 0.018, so the networks spread the information over several redundant filters, the most useful of which are P300-like. In the `early` models, three folds again relied most on a Pz or CP1 pattern (|r| of 0.90 or more), whereas the other two relied on a right parieto-occipital and a left fronto-central pattern.

**Attributions over time (Fig. F5b).** In the `full` models, Integrated Gradients attribution rose from about 250 ms and peaked at 495 ms (cross-subject) and 511 ms (within-subject), before the median response. The signed attribution of the P300 region favoured the target class and peaked at 472 and 456 ms, close to the P3b peak of the ERP. After 450 ms the P300 region was more enriched than the motor region (median 1.20 vs 1.05 cross-subject, p = 0.002; 1.19 vs 1.00 within-subject, p = 0.010; two-sided Wilcoxon over 35 subjects, descriptive). Between 600 and 800 ms, after most responses, the motor region received no more than an average channel (0.99 and 1.00). The one interval in which the motor region led was 0-250 ms (1.26 vs 1.07 cross-subject, p = 0.001), driven mainly by C1 and FCC3h. This attribution is not time-locked to the response and therefore unlikely to reflect motor execution.

**Linear reference (Fig. F5c).** The cross-subject LDA patterns were almost identical across folds (r of at least 0.98 between any two folds). In the `full` model, 71% of the absolute pattern lay after 450 ms. Its topography matched the P300 map from 350 ms onwards (r = 0.92, 0.99 and 0.95 for 250-450, 450-600 and 600-800 ms) and reached 0.58 µV over the P300 region at 500-550 ms. The motor region stayed below an average channel in every bin (enrichment 0.54 to 0.70); after 450 ms the pattern was 0.14 µV at C3 against 0.55 µV at Pz. Unlike gradient attributions, activation patterns cannot assign weight to a channel merely because it suppresses noise (Haufe et al., 2014). The LDA result therefore shows that the classical model uses the P3b and that a late left-central class difference is small in these data.

**Control tasks.** Distractors, which require no response, were decoded from standards almost as well as targets, and the `early` window kept most of this performance (Table 3). Target vs distractor, where both classes are rare and only targets involve a response, was the hardest task and lost the most in the `early` window. With no response in either class (`dis_std`), the motor region received the attribution of an average channel after 250 ms, while the P300 region was enriched (Table 4). In both tasks whose positive class contains a button press, the motor region rose above this no-response baseline between 250 and 600 ms. In `tgt_dis` it became the most attributed region, peaking at 456 ms at 1.71 times the average channel, with the strongest channels (FCC3h, FC1, FCC1h, C1) over the left, contralateral motor cortex. In `tgt_std` the motor region peaked at a similar time (464 ms, 1.31), but the P300 region stayed at least as high (472 ms, 1.37) and remained enriched after the response (600-800 ms: 1.26 vs 0.99).

**Table 3.** EEGNet balanced accuracy on the main and control tasks (median over 35 subjects).

| Task | Response in positive class | Within, `full` | Within, `early` | Cross, `full` | Cross, `early` |
|---|---|---:|---:|---:|---:|
| `tgt_std` (target vs standard) | yes | 0.941 | 0.875 | 0.924 | 0.830 |
| `dis_std` (distractor vs standard) | no | 0.866 | 0.824 | 0.839 | 0.783 |
| `tgt_dis` (target vs distractor) | yes | 0.684 | 0.549 | 0.723 | 0.620 |

**Table 4.** Enrichment of absolute Integrated Gradients attribution, P300 region / motor region (cross-subject, `full`, median over 35 subjects; 1 = average channel).

| Task | 0-250 ms | 250-450 ms | 450-600 ms | 600-800 ms |
|---|---:|---:|---:|---:|
| `tgt_std` | 1.07 / 1.26 | 1.38 / 1.30 | 1.27 / 1.15 | 1.26 / 0.99 |
| `dis_std` | 1.10 / 0.76 | 1.16 / 1.01 | 1.30 / 0.96 | 1.15 / 1.01 |
| `tgt_dis` | 0.87 / 1.62 | 1.08 / 1.27 | 1.27 / 1.15 | 1.10 / 1.14 |

**Answer to RQ3.** The features discovered by EEGNet are interpretable and physiologically plausible. Its most useful spatial filters reproduce the centro-parietal P3b topography, its attributions peak over the parietal midline near the P3b latency and before most button presses, and the classical LDA relies on the same P3b signal. The control tasks show that response-related activity is also used. Left-central attribution exceeds the no-response baseline, peaks shortly before the median response, and dominates when the P300 differs little between the classes (`tgt_dis`). Because this activity begins before the press, it more likely reflects motor preparation (the lateralised readiness potential) than the movement itself, and part of it falls inside the `early` window. We conclude that the target vs standard decision rests mainly on the P300, with a smaller contribution from motor preparation that the `early` window alone cannot remove. These conclusions rest on a priori regions and uncorrected descriptive tests, and gradient attributions are not activation patterns (Section 5).

## 5. Discussion
- Interpretation of main findings; comparison with literature.
- Motor confound: what the `early` window and control tasks show.
- Limitations (single dataset, hyperparameter budget, ICA fit on full recording, correct-trial selection makes target ≡ button press, …).
- Future work.

## 6. Conclusion
One paragraph answering RQ1–RQ3 directly.

## 7. Contributions
| Member | Role(s) | Main contributions | % |
|---|---|---|---|
| | | | |
| **Total** | | | **100%** |

## 8. Code and data availability
Repository URL, commit hash of final results, data source.

## References
- Dzianok, P. et al. (2022). The Nencki-Symfonia EEG/ERP dataset. *GigaScience*, 11. DOI 10.5524/100990.
- Gramfort, A. et al. (2013). MEG and EEG data analysis with MNE-Python. *Front. Neurosci.*
- Lawhern, V. J. et al. (2018). EEGNet. *J. Neural Eng.*
- Schirrmeister, R. T. et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Hum. Brain Mapp.*
- Rivet, B. et al. (2009). xDAWN algorithm to enhance evoked potentials. *IEEE TBME.*
- Barachant, A. et al. (2012). Multiclass BCI classification by Riemannian geometry. *IEEE TBME.*
- Haufe, S. et al. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. *NeuroImage.*
- Pion-Tonachini, L. et al. (2019). ICLabel. *NeuroImage.*
- Polich, J. (2007). Updating P300. *Clin. Neurophysiol.*
