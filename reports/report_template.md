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
