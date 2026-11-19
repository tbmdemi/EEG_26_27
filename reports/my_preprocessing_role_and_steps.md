# R2 preprocessing - current guide and Methods

Updated: 2026-11-19. Long is responsible for R2. All 42 oddball participants are considered; eligibility follows the shared protocol.

## Run the notebook

Open `notebooks/01_preproc_five_subjects.ipynb` and select **EEG oddball (Python 3.11)**, Restart Kernel and Run All. The historical filename is retained. Default Run All audits the published v2 data and creates QC plots; `RUN_PREPROCESS`, `DOWNLOAD_RAW` and `RUN_SYNTHETIC_TESTS` are False.

For new candidates, enable `RUN_PREPROCESS=True` and `DOWNLOAD_RAW=True` if the requested recordings are missing. `SUBJECTS=None` means the configured 42 IDs; `[]` means no work. The current manual decisions for sub-10/OI1h and sub-13/AFF1h are in `configs/config.yaml` and apply to candidate runs. They do not describe the already published v2 preprocessing.

Core logic has been moved to `scripts/r2_preprocessing.py`, release checks to `r2_validation.py`, QC evidence to `r2_qc.py` and optional integration checks to `r2_checks.py`, as authorized by Long on 2026-11-19. The notebook has 24 execution-code lines and imports these functions. CLI: `python scripts/02_r2.py` (audit) or `python scripts/02_r2.py --process --download -s 10 13` (candidates). Dependencies are in `requirements.txt`; `pybv` is listed in requirements for synthetic test-input export only.

## Inputs, outputs and tested status

- Published data: `data/derivatives/preproc/`; 35 subjects, 20,600 epochs; no FIFs were overwritten.
- Source event checks: all 42 task-specific tables in `data/validation/ds004621/`.
- Raw: sub-01 to sub-05 was deleted earlier as requested; sub-10/sub-13 has now been fetched for targeted QC and reprocessing and is retained unchanged.
- Candidate data: `data/derivatives/preproc/_candidate/`; separate log/provenance and QC under `results/r2_candidate/` and `figures/preprocessing/candidate/`.
- Visual summary QC now covers all 35 retained participants. It is not exhaustive interactive raw-channel inspection or a claim of artifact-free data.
- Actual raw-data reruns for sub-10 and sub-13 passed checks and produced 591 and 648 candidate epochs. These are not yet a versioned group release.
- Source-event, metadata, amplitude and baseline checks passed for released and candidate data. Synthetic integration uses real BrainVision/BIDS, MNE filters, ICA, ONNX ICLabel and FIF I/O. Automated regression includes R3 compatibility.

## Current report material

- [Methods - Preprocessing, v2](r2_methods_v2.md), inserted into report section 3.1.
- [Full subject and epoch accounting](r2_subject_epoch_table.md).
- [QC findings, candidates, coordinate evidence and R3 handoff](r2_qc_and_handoff.md).
- [Prepared Data v2 ready Issue](data_v2_ready_issue.md).

The official report must state its actual release version. Candidate counts cannot be substituted into v2 results. No contribution percentage or expanded ERP/ML/DL responsibility is assigned here.

<details>
<summary>Earlier implementation/audit notes - historical and superseded where inconsistent with the guide above</summary>

# R2 preprocessing: current run guide and evidence

Updated: 2026-11-18. This guide applies to the current saved notebook, not the historical v1 run.

## Start here

1. Open `notebooks/01_preproc_five_subjects.ipynb`. Its filename is historical; the configured scope is all 42 subjects, oddball only.
2. Select the existing **EEG oddball (Python 3.11)** kernel (or the tested `python3` kernel in the same Python 3.11 environment).
3. For the already downloaded v2 release, keep `RUN_PREPROCESS = False`, `DOWNLOAD_RAW = False` and `RUN_SYNTHETIC_TESTS = False`, then Run All. This validates the current release and regenerates audit/QC outputs.
4. To download raw and generate new candidate preprocessing, set both `RUN_PREPROCESS = True` and `DOWNLOAD_RAW = True` before Run All. Raw recordings were deleted previously. This is a large download/processing job, not required to inspect the existing v2 data.
5. For a small real-data trial instead of the full batch, keep the automatic processing flag False, run the definitions, then execute `run(ROOT, subjects=['01'], download=True)` in a separate cell. Set `overwrite=True` only when explicitly reprocessing an existing candidate is intended.

The batch considers 42 IDs but excludes ineligible subjects according to protocol; it does not force all 42 into analysis. Current official data: 35 usable participants and 20,600 epochs.

| Mode | Data/output | What is tested |
|---|---|---|
| Default Run All | Reads `data/derivatives/preproc/` and `data/validation/ds004621/`; writes `results/v2_audit/` and `figures/preprocessing/v2/` | Real-kernel validation passed for all 35 FIFs and all 42 behavioural summaries |
| Candidate processing | Writes `data/derivatives/preproc/_candidate/`, `results/r2_candidate/`, `figures/preprocessing/candidate/` | Real synthetic BrainVision/BIDS -> filtering/ICA/ICLabel -> FIF integration passed; full real-data reprocessing has not been run |
| Optional `test_r2()` | Temporary synthetic inputs; evidence in `results/v2_audit/r2_code_tests.json` | Requires `pybv` for test-input export only; installed locally |

Dependencies are listed in `requirements.txt`; `environment.yml` is no longer the setup source. All personal R2 implementation is in the notebook. The shared main-branch scripts/modules were not updated by this notebook change, and their remaining issues are documented below.

The official v2 release and its exclusion CSV remain unchanged. Candidate outputs require QC and a new team release before replacing official report data. Summary-plot review is not interactive channel review; AFF1h in sub-13 still needs inspection.

## Operational R2 notebook update - 2026-11-18

The R2 implementation in `notebooks/01_preproc_five_subjects.ipynb` is now
operational, rather than reference code with an unconditional processing block.
The historical filename is retained; the configured scope is all 42 participants.
No shared Python script or `src/` module was changed, following Long's notebook-only
implementation constraint. The shared CLI still has the earlier limitations;
this updated notebook is the personal implementation to inspect and test.

### Implemented behaviour

- Trial correctness uses the configured response window with a numerical boundary
  tolerance; target RT is the first response, and non-response conditions retain NaN RT.
- A-priori exclusions, minimum correct-target count and target-retention exclusions
  are logged explicitly. Acquisition/stimulus-count and hit-rate deviations remain flags.
  The nominal 660 stimulus count is now read from `dataset.expected_stimuli` in config;
  moving this existing value did not change an analysis threshold.
- Only the requested subject's oddball files and necessary BIDS/coordinate metadata
  are fetched, and only when downloading is explicitly enabled. No raw deletion is offered.
- CapTrak `.bvct` is preferred; `.bvef` is supported when CapTrak is absent. The latter
  uses MNE's default head-size normalization and head coordinate frame, matching saved
  sub-12 electrode positions within 1e-8 m. This is normalized geometry, not a claim that
  the original BrainVision radii are preserved unchanged.
- Latin-1 sidecars are decoded in a temporary read-only BIDS view. The original files
  are unchanged. Loaded MNE data are detached from temporary filenames before cleanup.
- Every positive `New Segment` marker is read directly from BrainVision sample positions
  and the header sampling interval. BAD padding is applied before filtering and ICA;
  epochs intersecting BAD regions are rejected. This also handles a first segment marker
  occurring after sample one, which annotation-reader normalization can otherwise hide.
- Filtering, bad-channel interpolation, average reference, extended Infomax ICA,
  ICLabel thresholds, resampling, baseline and PTP rejection read existing config values.
  All three event conditions and `subject/onset_s/block/rt` metadata are preserved.
- Resume requires a complete summary, matching scientific configuration, implementation
  revision/notebook SHA-256 and intact FIF checksums. Available source-file hashes are
  checked too. A missing or changed artifact forces reprocessing. If raw was intentionally
  removed, intact, compatible candidate results can still be resumed from recorded provenance.
- Batch logs distinguish `pending`, `complete` and `failed`. A processing failure has
  no scientific exclusion label. All configured IDs appear in the candidate log, including
  not-yet-processed subjects. An explicit empty subject list performs no work.

### Execution and outputs

Run All defaults to validating the official v2 release. To generate candidates,
set `RUN_PREPROCESS = True`; set `DOWNLOAD_RAW = True` only when raw downloading is
intended, since local raw recordings were deleted at the user's request.
The default batch examines all configured IDs and skips the three a-priori exclusions
before fetching recordings. A selected batch can be called with
`run(ROOT, subjects=['01'], download=True)`.

Candidate FIFs and summaries are written under `data/derivatives/preproc/_candidate/`;
rejected participants' FIFs are placed in its `_excluded/` directory. Prior candidate
artifacts are archived in `_history/` on reprocessing. Candidate logs are written to
`results/r2_candidate/preproc_log.csv`, QC to `figures/preprocessing/candidate/`.
The official v2 FIFs, published CSV and exclusion reasons are never overwritten.
Candidate results require review and the team's release process before replacing data
used in the report. Updating this code does not establish a new official release.

### Validation and limits

The notebook contains optional `test_r2()` integration tests. They export temporary
synthetic BrainVision/BIDS data and run actual MNE filtering, bad-channel interpolation,
extended Infomax ICA, ONNX ICLabel, epoch rejection and FIF writing. `pybv` is needed
only for synthetic-input export and was installed locally without dependencies.
Tests cover inclusive RT boundaries, empty/single onsets, shuffled event rows,
read-only Latin-1 input, `.bvef` coordinates, a BAD join overlapping a target,
a-priori/behaviour/retention exclusions, missing-raw failures, artifact/config resume
invalidation, empty batches and protection of the published output path.
The synthetic recording is intentionally short and triggers a long-FIR warning;
it tests execution and control logic, not physiological signal-recovery accuracy.

Evidence is saved to `results/v2_audit/r2_code_tests.json`. The real-kernel notebook
run also checks all 35 released FIFs and behavioural summaries for all 42 participants.
The released dataset contains 20,600 epochs and remains unchanged by these tests.
This does not constitute reprocessing the complete real 42-participant dataset with
the updated notebook, because the raw recordings are not locally available.
Visual/interactive QC, including AFF1h in sub-13, remains pending; neither bad-channel
choices nor the current release were silently altered.


## V2 release audit ? 2026-11-18

### Data and scope

The team-provided `preproc_v2_20261118.zip` was downloaded and its SHA-256 matched
`c53f040786b6c23a9b5d135a31f0bef966088018434409a1ca44a2e944c7a6ef`.
All 115 archive files were extracted to a staging directory before installation.
The active dataset is now `data/derivatives/preproc/`; per-file SHA-256 values are
recorded in `results/v2_audit/release_integrity.json`.
This audit did not regenerate the release or fit ICA. Existing raw recordings were not changed.
Only oddball `events.tsv` files were additionally downloaded for all 42 participants
(866,450 bytes total), under `data/validation/ds004621/`, with source URLs and hashes.
No other tasks or additional raw EEG recordings were downloaded.

All 42 participants are considered. The released analysis set contains 35 usable
participants. Sub-02, sub-14 and sub-17 are excluded a priori for misunderstanding
the task; sub-08, sub-23, sub-35 and sub-41 fail the target-retention criterion.
Requiring all 42 participants to enter the classifier would contradict the current protocol.

The 35 usable participants contain **20,600 retained epochs**: **15,744 standard**,
**2,477 target** and **2,379 distractor**. The main classification task uses the
18,221 target/standard epochs; distractors support the control tasks.
These counts replace the historical five-participant results for any v2 reporting.

### Verification and notebook changes

`configs/config.yaml` now matches the current main-branch configuration, including
all 42 IDs, behavioural criteria and five cross-subject folds.
The historical notebook filename is retained. All personal preprocessing
and verification code remains in `notebooks/01_preproc_five_subjects.ipynb`; no new
Python scripts/modules were created or edited. Run All validates the released data
and creates audit/QC outputs. Writes to the published release are blocked; explicit preprocessing writes candidates only. Promoting a candidate requires a new team release.

The notebook was executed using a real Jupyter kernel. Every released FIF passed:
127 channels, finite electrode locations and samples, 250 Hz, 251 samples,
?200 to 800 ms, correct baseline, near-zero average reference, peak-to-peak
at most 150 ?V, expected event IDs and per-condition counts, ordered onsets,
participant IDs, block values, and target RT values within 100?1400 ms.
Every retained event was matched against its source `events.tsv`: condition,
correctness, first-response RT and block assignment were checked for all 35 usable
participants. Behavioural summaries and exclusion decisions were independently
checked for all 42 participants and matched the released log.
The release's numerical/method parameters were compared with the local config;
the released CSV and the repository CSV also matched.

Evidence: `results/v2_audit/verification.json`, `behaviour_42.json`,
`notebook_execution.json`, `release_integrity.json` and `visual_review.json`.
The released summaries/logs and FIFs were kept unchanged.

### Code findings and remaining limitations

- **Historical notebook incompatibility (addressed):** v1 kept incorrect trials,
  did not implement current exclusions or RT metadata, and could overwrite the
  shared v2 log when run. The notebook now inspects the v2 reference implementation,
  validates current data and protects the published release from reprocessing.
- **Floating-point boundary handling (addressed in notebook only):** for example,
  `4.4 - 3.0 > 1.4` can incorrectly exclude a response exactly at the upper boundary.
  The notebook scorer uses a 1 ns numerical tolerance and checks empty onset inputs.
  Boundary and empty/single-onset tests passed. This tolerance is numerical only,
  far below EEG timing resolution. The shared `behaviour.py` still uses exact float
  comparisons; this remains a shared-code issue before future reprocessing.
  Importantly, the tolerance-corrected scorer matched the current release's behavioural
  summaries and retained-trial labels for all participants checked here.
- **Montage/reproduction gap (unresolved in shared code):** the current shared
  pipeline reads CapTrak unconditionally, whereas released summaries identify
  BrainVision `.bvef` coordinates for some participants (including sub-12/sub-23/sub-24).
  The current main source cannot reproduce those cases as written when CapTrak is absent.
  This audit checked finite saved electrode coordinates, not the original coordinate
  transformation. The release's generating source revision must be clarified before
  claiming exact end-to-end reproducibility.
- **Resume logic (unresolved in shared code):** pipeline resume checks version 2 in
  summary JSON, without verifying the FIF exists or that the configuration matches.
  `usable_subjects` is also summary-driven. The notebook explicitly reads each FIF,
  checks the complete release manifest and compares scientific parameters instead.
- **Downloader checks (worked around during installation):** its deep check inspects
  only the first FIF and does not validate pipeline version for every row. Same-name
  old files are overwritten during direct extraction. Staging plus full-file checks
  were used for this installation; shared downloader code was not edited.
- **Visual review remains incomplete:** 35 epoch-QC figures were generated under
  `figures/preprocessing/v2/`. Only summary plots for sub-01/07/10/13/24/37/38 were
  inspected in this audit. Sub-13 AFF1h has persistently elevated amplitude
  (median retained-epoch PTP 106.39 ?V, versus 26.26 ?V for the next-largest channel),
  and sub-10 also shows a persistent elevated channel band. These are review findings,
  not confirmed bad-channel diagnoses. Interactive inspection is needed before
  deciding whether a revised release is warranted. Original `visual_review=pending`
  entries were not changed. The data are not described as artifact-free.
- ICA is fit to the full unlabelled recording, and ICLabel is applied to 1?40 Hz data
  despite its 1?100 Hz training range; these limitations remain reportable.
  Downstream models still need the specified blocked/grouped CV, train-only fitting,
  full/early windows and motor-confound controls; this audit did not implement ML/DL.

### Cleanup status ? completed after explicit user confirmation

The user explicitly authorized deleting the original raw recordings for sub-01?05.
Those five subject directories and their source-coordinate directories were removed,
as were the inactive v1 derivatives backup, the extracted release ZIP, 25 v1 QC files,
two v1 verification/execution JSON files and two v1 preprocessing-run logs.
The earlier deletion rejection no longer describes the current cleanup status.
The active v2 derivatives, all 42 source-event tables in the validation directory,
v2 figures/audit evidence and the current exclusion log remain available.
All 115 active release-file hashes matched the manifest after legacy-output removal.
Raw-dependent reprocessing will require downloading raw data again; Run All uses
validation event tables and released FIFs and does not require those deleted raw files.

The local environment also passed `scripts/00_get_data.py --check` and the preprocessing CLI import/help check after installing `eegproj` in editable mode without dependencies. These checks do not constitute an end-to-end preprocessing rerun.




<details>
<summary>Historical v1 implementation and Methods - superseded; do not use as current instructions or v2 results</summary>

## Vai trò trong dự án

Phần công việc đang thực hiện tương ứng **R2 — Signal Preprocessing** trong
`docs/04_roles_and_tasks.md`. Đây là mô tả phần việc cá nhân đang chuẩn bị,
không tự thay đổi phân công chính thức hoặc phần trăm đóng góp của nhóm.

Trách nhiệm: kiểm tra dữ liệu, làm sạch tín hiệu cho sub-01 đến sub-05,
kiểm tra chất lượng và bàn giao epoch cùng log cho người làm ERP, ML và DL.
Notebook thực hiện: `notebooks/01_preproc_five_subjects.ipynb`.

## Phạm vi và các thay đổi cấu hình riêng cho lần chạy

- Chỉ dùng 5 subject theo yêu cầu, không tải toàn bộ 42 subject.
- Đọc số kênh thực tế là 127; không tạo thêm kênh giả để đủ 128.
- Bổ sung cấu hình ngưỡng kênh hỏng, thông số hội tụ/decimation ICA và QC.
- Các thông số đã chốt về band-pass, epoch, baseline, ICLabel và ngưỡng
  loại epoch vẫn theo protocol.
- Tài liệu nhóm được giữ nguyên. Các quyết định triển khai và hạn chế
  của lần chạy này tập trung trong file cá nhân này để nhóm xem xét.

## Nguồn yêu cầu

- `docs/01_project_brief.md`: mục tiêu bài tập, so sánh ML/DL và giải thích.
- `docs/02_dataset.md`: dataset, task oddball, mã sự kiện.
- `docs/03_analysis_protocol.md`, mục 1: các bước tiền xử lý bắt buộc.
- `configs/config.yaml`: tham số chạy thực tế.
- `docs/04_roles_and_tasks.md`, R2: các đầu ra phải bàn giao.

## Chạy

Mở `notebooks/01_preproc_five_subjects.ipynb`, chọn kernel Python 3.11 có
các thư viện trong `environment.yml`, rồi chạy các ô từ trên xuống.
Trên máy hiện tại, chọn kernel **EEG oddball (Python 3.11)** đã được cài.
Ô thực hiện dùng `run(ROOT, subjects=CONFIG['dataset']['subjects'], overwrite=False)`.
Để chạy thử một người, truyền `subjects=['01']`; để chạy lại, đặt `overwrite=True`.
Toàn bộ hàm và kiểm tra đều nằm trong notebook, không phụ thuộc script `.py` riêng.

Đầu vào: `data/raw/ds004621`. Bổ sung từ nguồn OpenNeuro của cùng
dataset: `task-oddball_eeg.json` và 5 tệp
`sourcedata/sub-XX/coords/sub-XX_captrak.bvct`.
Không tải thêm bản ghi EEG hay subject ngoài tập con được yêu cầu.

## Xử lý và quyết định triển khai

1. Đọc bằng MNE-BIDS, chỉ giữ EEG thực có (127 kênh).
2. Dùng tọa độ CapTrak riêng từng subject; đổi nhãn tai TP9/TP10 để khớp.
3. Tạo lại annotations từ cột `event_type` của events.tsv. Cột `trial_type`
   chỉ có stimulus/response, không đủ để phân biệt ba kích thích.
4. Phát hiện kênh phẳng hoặc độ lệch chuẩn bất thường qua robust z của log(std).
   Ngưỡng nằm trong config; xuất bảng điểm và hình để kiểm tra bằng mắt.
   Có thể thêm kênh xác nhận thủ công trong `bad_channels.manual` rồi chạy lại.
5. FIR zero-phase 0.1–40 Hz; nội suy spline; average reference.
   Không notch 50 Hz mặc định vì đã nằm ngoài dải low-pass 40 Hz.
6. Extended Infomax trên bản sao high-pass 1 Hz (1–40 Hz), 99% phương sai.
   Fit trên toàn recording không dùng nhãn; lấy mỗi mẫu thứ 4 sau lọc để
   giảm chi phí tính toán. Dừng với lỗi nếu chạm giới hạn số vòng lặp.
   Infomax dùng tối đa 512 vòng, ngưỡng thay đổi trọng số 1e-7 trong config;
   kiểm tra hội tụ từ log bước lặp thực tế. MNE Infomax có thể trả `n_iter_`
   bằng `max_iter` ngay khi hội tụ, nên không dùng riêng trường đó để kết luận.
7. ICLabel backend ONNX; loại thành phần có xác suất eye blink hoặc muscle
   artifact >=0.8. Áp dụng ICA lên dữ liệu 0.1–40 Hz.
8. Resample 250 Hz, đồng thời chuyển vị trí events; epoch -0.2 đến 0.8 s,
   baseline -0.2 đến 0 s; loại peak-to-peak >150 uV.
9. Giữ cả standard/target/distractor; metadata giữ onset gốc và block xác
   định từ các lần nghỉ S10. Subject giữ dưới 50% target được gắn cờ loại.

ICLabel được huấn luyện cho 1–100 Hz; chạy 1–40 Hz là lựa chọn theo protocol
hiện tại và cần ghi hạn chế này trong report.

## Đầu ra và nghiệm thu

- `data/derivatives/preproc/sub-XX_task-oddball_epo.fif`: epoch sau xử lý.
- Cùng thư mục: ICA để tái kiểm tra, summary JSON, provenance của lần chạy.
- `results/preproc_log.csv`: số epoch trước/sau, tỷ lệ giữ từng lớp,
  kênh nội suy, IC loại, subject loại và lý do.
- `figures/preprocessing/`: PSD, ví dụ tín hiệu trước/sau, bảng kênh,
  xác suất ICLabel và hình các IC bị loại.

Không coi một lần chạy là hoàn tất nghiệm thu nếu chưa kiểm tra hình QC.
`visual_review=pending` thể hiện trạng thái cần kiểm tra, không phải chứng
nhận dữ liệu đã sạch. Không dùng subject được gắn `excluded_subject=True`
cho phân loại chính.

Phạm vi chỉ gồm tiền xử lý. Thiết lập cross-subject 6 fold trong config
chưa được đổi; phải dùng tối đa 5 fold khi triển khai phân loại tập con.

## Kết quả đã thực hiện ngày 2026-11-13

| Subject | Standard còn lại / 498 | Target còn lại / 81 | Distractor còn lại / 81 | IC loại |
|---|---:|---:|---:|---:|
| sub-01 | 491 | 80 | 80 | 8 |
| sub-02 | 474 | 77 | 76 | 6 |
| sub-03 | 471 | 78 | 80 | 8 |
| sub-04 | 496 | 80 | 81 | 2 |
| sub-05 | 480 | 75 | 76 | 10 |

Tổng giữ **3.195/3.300 epoch (96,82%)**. Không có subject bị loại.
Không kênh nào vượt ngưỡng phát hiện tự động, nên không có nội suy trong
lần chạy này; nhánh nội suy vẫn có trong notebook cho trường hợp phát hiện kênh hỏng.

Cả 5 tệp FIF đã được đọc lại và vượt kiểm tra tự động: 127 kênh, 250 Hz,
251 mẫu/epoch, baseline, biên độ peak-to-peak, ba lớp kích thích, thứ tự
thời gian và đủ 5 block. Notebook đã chạy 11 ô code từ đầu đến cuối bằng
kernel Jupyter thật, theo chế độ tiếp tục từ các kết quả xử lý đã lưu,
không có lỗi. Code được giữ trong notebook; không còn script `.py` riêng.

Codex đã xem PSD, tín hiệu mẫu trước/sau, phân bố độ lệch chuẩn theo kênh,
topomap IC bị loại, bản đồ peak-to-peak của các epoch giữ lại và ERP Pz
của cả 5 người. Log ghi `summary_plots_reviewed`; đây là kiểm tra hình
tổng hợp, chưa phải duyệt tương tác mọi kênh trên toàn recording.
Sub-02 có nhiễu lớn ở đoạn đầu trước task; các epoch giữ lại vẫn vượt
kiểm tra biên độ. Không diễn giải toàn bộ continuous recording là sạch hoàn toàn.

Các kết quả này phục vụ bước tiền xử lý và bàn giao; chưa phải kết quả
huấn luyện ML/DL hoặc kiểm định thống kê của dự án.

## Bản tiếng Anh để ghép vào report

Đưa phần dưới vào mục **3.1 Preprocessing** của report chung. Phần kết quả
giữ/loại epoch nằm riêng để ghép vào Data hoặc Results; các hạn chế đưa
vào Discussion. Bản nháp này chỉ mô tả các bước đã thực hiện trên 5 người.

### 3.1 Preprocessing

EEG preprocessing was implemented in Python using MNE-Python (v1.13.2),
MNE-BIDS (v0.20.0), and MNE-ICALabel (v0.10.0). An exploratory subset of
five participants (sub-01–sub-05) from the Nencki-Symfonia dataset was
processed. Each recording contained 127 EEG channels sampled at 1,000 Hz.
Recordings were read using MNE-BIDS, and participant-specific digitized
CapTrak coordinates were assigned to the channels. Stimulus identities
were recovered from the `event_type` column of the BIDS events files,
distinguishing standard (S5), target (S6), and distractor (S7) events.

Potentially bad channels were identified from channel standard deviations,
computed using every tenth raw sample. Channels were flagged if their
standard deviation was below 0.01 µV or if the absolute robust z-score of
their log standard deviation exceeded 5. Robust z-scores used the median
and median absolute deviation, with the latter scaled by 1.4826.
Spherical-spline interpolation was specified for flagged channels; none
were flagged in this subset, so no channels were interpolated. Signals
were band-pass filtered between 0.1 and 40 Hz using a zero-phase FIR
filter and re-referenced to the common average. No separate 50-Hz notch
filter was applied because the 40-Hz low-pass filter attenuated this
frequency; attenuation was checked in the quality-control spectra.

Ocular and muscle artifacts were addressed using extended Infomax
independent component analysis (ICA). ICA was fitted to a copy of the
filtered recording with an additional 1-Hz high-pass filter, retaining
principal components explaining 99% of the variance. Every fourth sample
of this filtered copy was used for ICA fitting to reduce computational
cost. The random seed was fixed at 42, the maximum number of iterations
was 512, and the stopping threshold for the squared weight change was
1e-7. Components were classified using ICLabel with its ONNX backend.
Components with an eye-blink or muscle-artifact probability of at least
0.80 were excluded. The fitted ICA decomposition was then applied to
the 0.1–40-Hz recording. ICA was fitted independently to each participant's
entire recording without using stimulus-class labels, before subsequent
cross-validation splits.

After ICA correction, recordings were resampled to 250 Hz, with event
sample indices adjusted simultaneously. Epochs extended from −200 to
800 ms relative to stimulus onset, with baseline correction using the
−200-to-0-ms interval. Epochs were rejected if the peak-to-peak amplitude
exceeded 150 µV in any EEG channel. A participant was to be excluded if
fewer than 50% of the original target epochs remained. All three stimulus
classes were retained in the output. Original stimulus onsets and block
identities derived from the S10 break events were preserved as metadata
to support subsequent evaluation using contiguous temporal blocks.

Processed epochs were saved in FIF format. Quality control included
per-class retention counts, channel-amplitude summaries, spectra and
sample traces before and after processing, topographies of excluded ICA
components, retained-epoch amplitude maps, and ERPs at Pz. Saved epochs
were reloaded to verify sampling frequency, epoch length, baseline
correction, event classes, amplitude limits, and temporal metadata.
Visual inspection was limited to these summary plots rather than an
exhaustive interactive review of every channel throughout each recording.

### Preprocessing outcomes — for Data or Results

Of the 3,300 stimulus epochs across the five participants, 3,195 (96.82%)
were retained: 2,412 standard, 390 target, and 393 distractor epochs.
The number of retained epochs per participant was 651, 627, 629, 657,
and 631, respectively. A total of 34 ICA components were excluded
(8, 6, 8, 2, and 10 per participant). No participant met the exclusion
criterion. These counts describe preprocessing outcomes and do not
establish downstream classification performance.

### Preprocessing limitations — for Discussion

This analysis used five participants rather than the complete dataset,
limiting the generalizability of downstream findings. ICLabel was applied
to a 1–40-Hz recording in accordance with the project protocol, whereas
its recommended training conditions include a 1–100-Hz passband; this
filtering mismatch may affect component classification. In addition,
ICA used each participant's full recording before cross-validation,
including segments later assigned to test folds. Although stimulus-class
labels were not used, this constitutes a transductive preprocessing
choice and should be considered when interpreting generalization.

### References to include in the shared report

- Gramfort et al. (2013), MNE-Python — đã có trong References của report template.
- Pion-Tonachini et al. (2019), ICLabel — đã có trong References của report template.
- Dzianok et al. (2022), Nencki-Symfonia dataset — đã có trong References của report template.

Khi ghép báo cáo, định dạng citation theo cùng kiểu với các phần còn lại.



</details>

</details>
