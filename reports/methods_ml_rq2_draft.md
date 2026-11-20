# Methods & Results — Classical ML and RQ2 Comparison (R4 draft, for report §3.3, §3.5, §4.2, §4.3 & §5)

> Draft 2026-11-20, Pipeline v2 (`preproc_v2_20261118.zip`, N = 35 usable subjects), synchronized with R5 Deep Learning
> (`r5_dl_results_20261120_fc0cfaf.zip`, `evaluation.within.split: equal_segments`, and fixed `results/cv_cross_subject_folds.csv`).

---

## 3.3 Classical models

Two complementary classical evoked-response decoding pipelines were implemented in `src/eegproj/models/classical.py` and `src/eegproj/features/time_windows.py`, operating on the preprocessed 127-channel epochs sampled at 250 Hz and converted from Volts to microvolts (µV) prior to feature extraction:

1. **Time-Window Shrinkage LDA (`lda_tw`):** A stateless `TimeWindowAverager` averages amplitudes (µV) per channel within contiguous 50 ms bins (`configs/config.yaml`: `bin_ms = 50`), yielding 16 bins over `full` (`[0.0, 0.8] s`, 201 samples $\rightarrow$ 2,032 features) and 9 bins over `early` (`[0.0, 0.45] s`, 113 samples $\rightarrow$ 1,143 features). Features are standardized (`StandardScaler`) and classified with shrinkage Linear Discriminant Analysis (`LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5])`). Equal priors compensate for the ~6.4:1 standard-to-target imbalance without oversampling. For RQ3 explainability (protocol §9.4), `extract_lda_haufe_patterns` transforms classifier weights $w$ into forward activation patterns $A = \Sigma_Z w / \operatorname{Var}(Z w)$ (Haufe et al., 2014).
2. **Rank-Safe xDAWN + Riemannian Tangent Space (`xdawn_rg`):** Supervised spatial filtering via xDAWN (`XdawnCovariances(nfilter=4)`; Rivet et al., 2009) learns 4 filters per class (8 virtual channels total) and augments single trials with the class-evoked templates to form 16 × 16 symmetric positive-definite (SPD) covariance matrices (`estimator='lwf'`). Matrices are projected onto the Riemannian tangent space (`TangentSpace(metric='riemann')`; Barachant et al., 2012; 136 features) and classified via L2 logistic regression (`LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)`). Because average referencing and ICA component removal (`7.51 ± 4.01` ICs removed per subject) render the 127-channel EEG rank-deficient ($r < 127$), applying isotropic shrinkage (`'lwf'`) directly to the short evoked template $P$ injects variance into the null space of the background covariance $C_X$ and destabilizes $\operatorname{eigh}(C_P, C_X)$. Our `_RankSafeXdawnEstimator` uses unregularized sample covariance (`'scm'`) for the evoked template $P$ (`x.shape[1] <= 1000`) while retaining Ledoit–Wolf shrinkage (`'lwf'`) for the concatenated signal matrix $X$.

---

## 3.5 Evaluation

All models were evaluated on the **35 usable Pipeline v2 subjects** (`pipeline_version == 2`, `excluded_subject == False`) using shared, leakage-free splitting and scoring modules in `src/eegproj/evaluation/` (`splits.py`, `folds.py`, `metrics.py`, `stats.py`):

- **Within-subject 5-fold blocked CV (`within_cv_metadata`, `iter_within_folds`):** Trials remain strictly ordered chronologically (`onset_s` monotonic increasing, never shuffled). Per the updated team protocol (`evaluation.within.split: equal_segments`), each subject's recording is partitioned into 5 contiguous time-ordered segments of equal trial count (preserving the original `S 10` block labels in `s10_block`) so that all 35 subjects—including subjects whose final short `S 10` segment lacked positive trials (`sub-13`, `sub-27`, `sub-10`)—yield 5 valid two-class folds.
- **Cross-subject 5-fold GroupKFold (`cross_subject_folds`, `results/cv_cross_subject_folds.csv`):** The 35 usable subjects are partitioned into 5 subject-disjoint folds (7 held-out test subjects and 28 training subjects per fold), frozen in `results/cv_cross_subject_folds.csv` so classical ML and deep learning share identical training/test partitions. Crucially, `src/eegproj/models/runner.py` reorders channels by electrode name (`_align_eeg_channels`) prior to cross-subject concatenation because `sub-13` and `sub-42` store the 127 EEG channels in a different row order than the remaining cohort. Scores are computed individually per held-out test subject (protocol §7).
- **Leakage safeguards, metrics, and RQ2 statistics:** All learned transforms (`StandardScaler`, `XdawnCovariances`, `TangentSpace`, classifiers, DL channel z-scoring, and DL early stopping) fit strictly inside training folds; class imbalance is handled exclusively via priors/class weights. Primary metrics are balanced accuracy and ROC-AUC (chance = 0.5), supplemented by target-class F1 (`f1_target`). For RQ2 (protocol §8), within-subject fold scores are averaged per subject (`N = 35` paired observations); `classical-best` and `dl-best` are selected per `(task, setting)` by **median balanced accuracy on the `full` window** and compared via two-sided paired Wilcoxon signed-rank tests (`pingouin.wilcoxon`) with Holm step-down correction (`pingouin.multicomp`) and rank-biserial correlation (`RBC`).

---

## 4.2 Classification performance — RQ1 (Main and Control Tasks, N = 35)

Across all **35 usable subjects** (24 classical CSVs + 8 DL CSVs, 0 NaNs), both model families achieved high single-trial decoding on the primary `tgt_std` task (Table 1). Within subjects (`full`, 0–800 ms), `xdawn_rg` reached `bal_acc = 0.9173 ± 0.0863` (`median = 0.9488`, `IQR = 0.0736`; `AUC = 0.9774 ± 0.0506`, `median = 0.9946`) and `lda_tw` reached `0.9162 ± 0.0665` (`median = 0.9390`, `AUC = 0.9715 ± 0.0455`). Across subjects (`full`), `xdawn_rg` outperformed `lda_tw` (`bal_acc = 0.8690 ± 0.0829`, `median = 0.8716` vs `0.8605 ± 0.0621`, `median = 0.8551`; `AUC = 0.9543 ± 0.0444` vs `0.9391 ± 0.0488`, Wilcoxon `p = 3.15 × 10⁻⁵`, `RBC = +0.756`), while `eegnet` achieved `0.9002 ± 0.0688` (`median = 0.9236`, `AUC = 0.9662 ± 0.0384`).

**Table 1.** Single-trial classification on the main task (`tgt_std`: Target vs Standard, N = 35; `Mean ± SD` and `Median ± IQR`).

| Model | Family | Setting | Window | Bal. Acc (Mean ± SD) | Bal. Acc (Median ± IQR) | ROC-AUC (Mean ± SD) | ROC-AUC (Median ± IQR) | Target F1 (Mean ± SD) |
|---|---|---|---|---:|---:|---:|---:|---:|
| `lda_tw` | classical | within | `full` | `0.9162 ± 0.0665` | `0.9390 ± 0.0770` | `0.9715 ± 0.0455` | `0.9839 ± 0.0298` | `0.8426 ± 0.1245` |
| `xdawn_rg` | classical | within | `full` | **`0.9173 ± 0.0863`** | **`0.9488 ± 0.0736`** | **`0.9774 ± 0.0506`** | **`0.9946 ± 0.0176`** | **`0.8622 ± 0.1564`** |
| `eegnet` | dl | within | `full` | `0.9168 ± 0.0728` | `0.9412 ± 0.0665` | `0.9662 ± 0.0510` | `0.9812 ± 0.0276` | `0.7949 ± 0.1469` |
| `shallow` | dl | within | `full` | `0.8601 ± 0.0990` | `0.9008 ± 0.1091` | `0.9266 ± 0.0894` | `0.9643 ± 0.0697` | `0.7169 ± 0.1781` |
| `lda_tw` | classical | within | `early` | **`0.8533 ± 0.0952`** | `0.8487 ± 0.1219` | `0.9331 ± 0.0747` | `0.9463 ± 0.0693` | `0.7229 ± 0.1705` |
| `xdawn_rg` | classical | within | `early` | `0.8526 ± 0.1088` | `0.8731 ± 0.1177` | **`0.9371 ± 0.0838`** | **`0.9698 ± 0.0712`** | **`0.7473 ± 0.1919`** |
| `eegnet` | dl | within | `early` | `0.8505 ± 0.1021` | **`0.8746 ± 0.1464`** | `0.9140 ± 0.0984` | `0.9447 ± 0.1155` | `0.6622 ± 0.1664` |
| `shallow` | dl | within | `early` | `0.7347 ± 0.1454` | `0.7734 ± 0.2886` | `0.8081 ± 0.1624` | `0.8643 ± 0.3060` | `0.5253 ± 0.2313` |
| `lda_tw` | classical | cross | `full` | `0.8605 ± 0.0621` | `0.8551 ± 0.0957` | `0.9391 ± 0.0488` | `0.9436 ± 0.0455` | `0.6944 ± 0.1057` |
| `xdawn_rg` | classical | cross | `full` | `0.8690 ± 0.0829` | `0.8716 ± 0.0835` | `0.9543 ± 0.0444` | `0.9614 ± 0.0375` | `0.6973 ± 0.1356` |
| `eegnet` | dl | cross | `full` | **`0.9002 ± 0.0688`** | **`0.9236 ± 0.0822`** | **`0.9662 ± 0.0384`** | **`0.9793 ± 0.0308`** | `0.7448 ± 0.1260` |
| `shallow` | dl | cross | `full` | `0.8872 ± 0.0802` | `0.9046 ± 0.0710` | `0.9644 ± 0.0462` | `0.9784 ± 0.0286` | **`0.7494 ± 0.1388`** |
| `lda_tw` | classical | cross | `early` | `0.7910 ± 0.0881` | `0.7957 ± 0.0975` | `0.8799 ± 0.0861` | `0.8910 ± 0.0831` | `0.5514 ± 0.1208` |
| `xdawn_rg` | classical | cross | `early` | `0.7987 ± 0.0928` | `0.8133 ± 0.1186` | `0.8999 ± 0.0811` | **`0.9221 ± 0.0783`** | `0.5475 ± 0.1330` |
| `eegnet` | dl | cross | `early` | **`0.8195 ± 0.0865`** | **`0.8303 ± 0.0957`** | **`0.9043 ± 0.0791`** | `0.9193 ± 0.1065` | **`0.5938 ± 0.1377`** |
| `shallow` | dl | cross | `early` | `0.7777 ± 0.1016` | `0.7916 ± 0.1502` | `0.8691 ± 0.1024` | `0.8988 ± 0.1449` | `0.5433 ± 0.1455` |

Control tasks (`N = 35`, classical models; Table 1b) confirmed high non-motor decoding in `dis_std` (`within full`: `0.8471 ± 0.0879` `lda_tw`, `AUC = 0.9266`; `cross full`: `0.8299 ± 0.0786` `xdawn_rg`, `AUC = 0.9106`) and a sharp late gain in `tgt_dis` (`within`: `0.6178–0.6361` on `early` vs `0.7362–0.7652` on `full`, $\Delta = +0.1184$ to $+0.1291$, $p \le 3.20 \times 10^{-9}$).

---

## 4.3 Classical vs deep learning — RQ2 (Fig. F4, statistics)

Following protocol §8, **`xdawn_rg`** (`classical-best`) and **`eegnet`** (`dl-best`) were selected in both `within` and `cross` settings by `full`-window median balanced accuracy (Table 2; Figure F4). A clear setting-dependent dissociation emerged:

- **Within-subject (`N = 35`):** `xdawn_rg` and `eegnet` achieved statistically indistinguishable balanced accuracy on both `full` (`0.9488` vs `0.9412`, $\Delta\text{med} = -0.0039$, $W = 294.0$, $p_{\text{Holm}} = 0.740$, $r = -0.067$) and `early` (`0.8731` vs `0.8746`, $p_{\text{Holm}} = 0.369$). However, `xdawn_rg` achieved **significantly higher ROC-AUC** than `eegnet` on both `full` (`0.9946` vs `0.9812`, $\Delta\text{med} = -0.0057$, $p_{\text{Holm}} = 1.03 \times 10^{-5}$, $r = -0.832$) and `early` (`0.9698` vs `0.9447`, $p_{\text{Holm}} = 6.92 \times 10^{-5}$, $r = -0.752`).
- **Cross-subject (`N = 35`):** **`eegnet` significantly outperformed `xdawn_rg`** on `full` for both balanced accuracy (`0.9236` vs `0.8716`, $\Delta\text{med} = +0.0202$, $W = 80.0$, $p_{\text{raw}} = 4.16 \times 10^{-5}$, $p_{\text{Holm}} = 1.66 \times 10^{-4}$, $r = +0.746$) and ROC-AUC (`0.9793` vs `0.9614`, $\Delta\text{med} = +0.0108$, $p_{\text{Holm}} = 2.35 \times 10^{-5}$, $r = +0.800$). On `cross early`, `eegnet` showed a raw advantage in balanced accuracy (`0.8303` vs `0.8133`, $p_{\text{raw}} = 0.0287$) that did not survive Holm correction ($p_{\text{Holm}} = 0.0862$, $r = +0.422$).

**Table 2.** Paired Wilcoxon signed-rank tests (`DL − Classical`, N = 35, Holm-corrected across the 4 setting × window comparisons per metric; `results/rq2_wilcoxon_holm.csv` and `results/rq2_wilcoxon_holm_auc.csv`).

| Metric | Setting | Window | `xdawn_rg` (Med ± IQR) | `eegnet` (Med ± IQR) | Δ Med (`DL − Cls`) | *W* | *p* (raw) | *p* (Holm) | RBC (*r*) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bal_acc` | `within` | `full` | `0.9488 ± 0.0736` | `0.9412 ± 0.0665` | `-0.0039` | `294.0` | `0.740` | `0.740` | `-0.067` |
| `bal_acc` | `within` | `early` | `0.8731 ± 0.1177` | `0.8746 ± 0.1464` | `-0.0112` | `233.0` | `0.184` | `0.369` | `-0.260` |
| `bal_acc` | `cross` | `full` | `0.8716 ± 0.0835` | **`0.9236 ± 0.0822`** | **`+0.0202`** | `80.0` | `4.16e-5` | **`1.66e-4`** | **`+0.746`** |
| `bal_acc` | `cross` | `early` | `0.8133 ± 0.1186` | `0.8303 ± 0.0957` | `+0.0158` | `182.0` | `0.0287` | `0.0862` | `+0.422` |
| `auc` | `within` | `full` | **`0.9946 ± 0.0176`** | `0.9812 ± 0.0276` | **`-0.0057`** | `53.0` | `2.58e-6` | **`1.03e-5`** | **`-0.832`** |
| `auc` | `within` | `early` | **`0.9698 ± 0.0712`** | `0.9447 ± 0.1155` | **`-0.0158`** | `78.0` | `3.46e-5` | **`6.92e-5`** | **`-0.752`** |
| `auc` | `cross` | `full` | `0.9614 ± 0.0375` | **`0.9793 ± 0.0308`** | **`+0.0108`** | `63.0` | `7.84e-6` | **`2.35e-5`** | **`+0.800`** |
| `auc` | `cross` | `early` | `0.9221 ± 0.0783` | `0.9193 ± 0.1065` | `+0.0056` | `245.0` | `0.259` | `0.259` | `+0.222` |

---

## 5. Discussion points (Classical vs DL & Motor Confound)

- **Sample-efficiency vs cross-subject representation:** Within subjects (~430 training trials per fold, ~57 targets), convex covariance geometry (`xdawn_rg`) matched `eegnet` in balanced accuracy and surpassed it significantly in ROC-AUC (`0.9946` vs `0.9812`, $p_{\text{Holm}} = 1.03 \times 10^{-5}$), because SPD tangent-space estimation requires fewer parameters than gradient-based CNN optimization. Across subjects (28 subjects pooled, ~14.5k training trials), `eegnet` learned subject-invariant spatio-temporal representations that generalized significantly better (`0.9236` vs `0.8716`, $p_{\text{Holm}} = 1.66 \times 10^{-4}$), likely reflecting cross-subject differences in electrode covariance geometry and P300 latency (`rt_mean_ms` spans `417.6–686.4 ms`) that linear tangent-space centering cannot adapt to without per-subject alignment.
- **Motor confound (`early` vs `full`, `dis_std`, `tgt_dis`):** Non-motor `dis_std` decoding reached `0.8471` (`within full`) and `0.8299` (`cross full`), while pre-response `tgt_std early` (`0–450 ms`) preserved `0.8533` (`within`) and `0.7987–0.8195` (`cross`). Extending `dis_std` from `early` to `full` still improved decoding significantly (`+0.0396–0.0804`, $p \le 9.74 \times 10^{-7}$), showing that late cognitive P3b/P3a components persist past 450 ms. However, the steepest `full − early` jump occurred in `tgt_dis` (`+0.1184–0.1291` within subjects), indicating that post-450 ms target separation combines late task-relevant P3b activity with motor preparation/execution around the ~544 ms response.

## References

- Barachant, A., et al. (2012). Multiclass BCI classification by Riemannian geometry. *IEEE TBME*, 59(4), 920–928.
- Haufe, S., et al. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. *NeuroImage*, 87, 96–110.
- Rivet, B., et al. (2009). xDAWN algorithm to enhance evoked potentials. *IEEE TBME*, 56(8), 2035–2043.
