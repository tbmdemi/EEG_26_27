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
- Subjects/epochs excluded and why (table from `results/preproc_log.csv`).

## 3. Methods
### 3.1 Preprocessing
Filtering, bad channels, re-referencing, ICA + ICLabel, resampling, epoching, rejection — exact parameters.
### 3.2 Classification tasks and time windows
Main task (target vs standard); control tasks (distractor vs standard, target vs distractor); `full` vs `early` windows and the motor-response rationale.
### 3.3 Classical models
`lda_tw`, `xdawn_rg` (+ optional) — features and hyperparameters.
### 3.4 Deep learning models
Architectures, input normalisation, training (optimizer, early stopping, class weighting).
### 3.5 Evaluation
Within-subject blocked CV; cross-subject GroupKFold; leakage safeguards; metrics; Wilcoxon signed-rank with Holm correction.
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
- Limitations (single dataset, hyperparameter budget, ICA fit on full recording, …).
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
