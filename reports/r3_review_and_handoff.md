# R3 — Bàn giao ERP & Visualization

Ngày cập nhật: **2026-11-23**. Nhánh: `feat/erp-f3-aggregation`.

**Đã triển khai, kiểm chứng và chuẩn bị ZIP output local; chưa upload/tiếp nhận trên Drive và chưa phê duyệt hình F2 cuối.** Không commit/push/PR trong đợt hoàn thiện này. Vẫn dùng release chính thức `preproc_v2_20261118.zip`; không trộn candidate sub-10/sub-13.

## Trạng thái từng phần

| Phần | Triển khai / kiểm chứng | Phê duyệt / bàn giao |
|---|---|---|
| ERP/SEM, difference wave, topomap | Notebook 02–04 và module R3; đã đối chiếu cohort/counts, bảng số và bằng chứng chạy v2 từ backup | R2 đã hoàn tất summary QC; F2 gốc vẫn là draft chờ duyệt hình cuối |
| P300 theo subject | 210 ứng viên, 35 người × 2 điều kiện × 3 kênh; đã tính lại thống kê Pz từ CSV thật và đối chiếu bảng LaTeX | PM duyệt quy tắc mô tả ngày 2026-11-22; không đồng nghĩa xác nhận mọi cực đại là P300 |
| Hành vi | Bảng nhóm 39/35, mean ± SD mẫu qua subject và đối chiếu Table 5; đã tính lại summary từ bảng subject | Đã chuẩn bị trong gói local; chưa khẳng định nhóm đã nhận trên Drive |
| F3 | Audit 32 CSV thật, tổng hợp theo subject, PNG/PDF và số liệu LaTeX đã đối chiếu | Đã chuẩn bị trong gói local; bằng chứng fold từ metrics không phải manifest gốc R4 |
| Hình QC/PSD–ICA | R2 có mô tả nguồn bằng chứng; file chưa có trong workspace/backup được kiểm tra | Phụ thuộc R2 cung cấp; không tạo hình thay thế |

Các checkbox/trạng thái thuộc R1/R2/R4/R5 không được sửa hộ. Mục tổng hợp R1 trong phân công còn thông tin cũ về F3/P300; PM cần tự đồng bộ phần thuộc trách nhiệm của mình.

## ERP và F2

- Cohort v2: **35/42 subject**, **20.600 epoch** = 15.744 standard + 2.477 target + 2.379 distractor; khớp `results/preproc_log.csv` hiện tại.
- Trung bình trial trong mỗi subject rồi trung bình qua người với trọng số bằng nhau. SEM dùng SD mẫu (`ddof=1`) / căn N; difference-wave SEM dựa trên hiệu ghép cặp trong từng subject.
- Bảng grand average có 3.012 dòng; topomap cửa sổ/lát cắt có 508/2.032 dòng; vị trí điện cực có 4.445 dòng (35 × 127).
- Ghép điện cực theo tên; bản đồ nhóm dùng vị trí điện cực trung bình trong hệ tọa độ đầu. Cửa sổ hiển thị 250–600 ms lấy 88 mẫu có sẵn từ 252–600 ms; snapshot 300/400/500/600 ms.
- Tại 500 ms ở Pz, target, standard và distractor lần lượt là **9,38 ± 0,51**, **0,76 ± 0,27**, **7,11 ± 0,45 µV** (mean ± SEM); hiệu ghép cặp target − standard là **8,61 ± 0,48 µV**. ERP target grand average đạt đỉnh 484 ms (9,46 µV, SEM 0,50); distractor đạt đỉnh 504 ms. Các số và diễn giải này đã chuyển từ bản ERP draft; nguồn là `results/erp/grand_average_summary.csv` và Results LaTeX.
- Các bản đồ draft cho thấy phân bố dương vùng sau/trung tâm; đây là điện thế da đầu, không phải kết quả định vị nguồn. R2 đã ghi nguồn 33 montage CapTrak và hai BrainVision `.bvef` tại [bàn giao R2](r2_qc_and_handoff.md).
- Đã kiểm tra đủ tám cặp PNG/PDF của gói, PNG đạt 300 dpi; đã render và xem cả tám PDF. F2 và các bảng gốc được sao chép **nguyên byte** từ `EEG_26_27_backup`, không vẽ lại hoặc đổi số liệu. Code thuật toán ERP/topomap/P300/hành vi hiện tại khớp backup theo cú pháp Python.
- Caption F2 trong LaTeX đã ghi rõ cách ghép kênh, tọa độ, SEM và trạng thái draft. Dòng `visual QC pending` trên hình và các trường `pending` trong CSV/JSON gốc là snapshot lịch sử trước khi R2 hoàn tất summary QC; giữ nguyên để không sửa provenance. Đây không phải bằng chứng QC hiện tại chưa làm hoặc hình đã được duyệt.
- Cần người phụ trách duyệt hình cuối và kiểm tra độ đọc/font ở kích thước in thực tế. Không tự thay v2 bằng candidate để sửa hình.

Bằng chứng lịch sử: notebook 02–06 có **26/26 ô code đạt** và kiểm chứng độc lập trên 35 FIF trong `outputs/r3_review/`. Đợt này kiểm tra lại các bảng/file thật và tính thống kê từ CSV, không tuyên bố đã chạy lại toàn bộ FIF/notebook.

Tái lập ERP/P300/hành vi: cài môi trường theo [README](../README.md), lấy release v2 bằng `python scripts/00_get_data.py`, rồi Restart Kernel/Run All từng notebook **02_erp_exploration → 03_erp_grand_average → 04_erp_topomaps → 05_erp_p300_candidates → 06_erp_behaviour_summary**. Bộ nạp kiểm tra log đủ subject cấu hình ở v2, cohort khớp log bàn giao và metadata; cohort ERP hiện tại phải có 35 người.

## P300

Theo **protocol §9, PM duyệt 2026-11-22**: tìm mẫu điện thế lớn nhất **có dấu** trong 300–600 ms, riêng từng subject/điều kiện/kênh; nếu bằng nhau lấy mẫu đầu. Không lấy trị tuyệt đối, làm trơn thêm, nội suy hoặc đổi cửa sổ theo người. Độ phân giải ở 250 Hz là 4 ms.

Gắn cờ cực đại không dương, tại/gần biên ≤20 ms, không đạt kiểm tra cực đại cục bộ hoặc có mẫu đồng cực đại. **Giữ mọi ứng viên có cờ và không loại thêm subject**. Pz là phạm vi báo cáo chính, CPz/Cz bổ sung; không trộn ba kênh thành quan sát độc lập. Median và Q1/Q3 dùng nội suy tuyến tính qua 35 subject mỗi điều kiện.

| Điều kiện, Pz | Biên độ median [Q1, Q3] (µV) | Latency median [Q1, Q3] (ms) | Có cờ, vẫn giữ |
|---|---|---|---|
| Target | 10,34 [9,03, 12,60] | 484 [448, 562] | 5/35 |
| Distractor | 7,91 [7,02, 9,27] | 504 [440, 540] | 1/35 |

Số trên tính lại từ `p300_candidate_peaks.csv`, khớp summary gốc; output dẫn xuất mới `results/erp/p300_pz_report_summary.csv` có hai dòng. Methods và bảng `tab:p300-candidates` trong LaTeX khớp CSV. Có target sub-38 tại biên 600 ms. Đây là phép đo mô tả ứng viên, không xác nhận mọi cực đại là P300/P3a/P3b hoặc suy ra ý nghĩa thống kê. Giữ nguyên `erp.p300`, thuật toán và tên file `candidate` theo protocol.

Manifest P300 gốc còn tham chiếu `reports/p300_measurement_notes.md`, hiện không có trong cả repo/backup. Không sửa manifest lịch sử để che khoảng trống đó; quy tắc hiện hành được ghi tại protocol §9, tài liệu này và LaTeX Methods.

## Hành vi

Hành vi dùng số đếm trước loại epoch của 39 người sau loại trước sub-02/14/17; nhóm ERP có 35 người, hai nhóm chồng lấp. Hit rate = target đúng/tổng target; false alarm = 1 − correct/total, riêng standard/distractor. Mean/SD nhóm tính qua subject (`ddof=1`), SD của RT từ các `rt_mean_ms`, không trung bình SD trong người. Đã đối chiếu summary với bảng subject và giữ chênh lệch thực tế so với Table 5.

So sánh với Table 5 là mô tả, không phải kiểm định tái lập chính xác hoặc bằng chứng khác biệt có ý nghĩa. Phân tích nhiễu vận động thuộc các phần R1/R4/R5; thống kê ERP/P300 ở đây không xác nhận một thành phần không có đóng góp vận động.

## F3

### Đầu vào và validation

| CSV thật đã audit | Số file | Số dòng |
|---|---:|---:|
| `lda_tw`, `xdawn_rg` × within/cross × full/early × 3 task | 24 | 2.520 |
| `eegnet`, `shallow` × within/cross × full/early × `tgt_std` | 8 | 840 |
| Tổng | **32** | **3.360** |

- Tên file khớp nội dung, đúng thứ tự 13 cột protocol §7. Mỗi file within có 175 dòng; cross có 35 dòng. Log đủ 42 subject v2, 35 dùng được; loại `02, 08, 14, 17, 23, 35, 41`. Mọi tổ hợp dùng đúng cohort đó; không lấy giao, bỏ dòng thiếu hoặc loại thêm người.
- Không missing values, dòng trùng hoặc metric không hữu hạn/ngoài [0, 1]. Fold, seed, n_train/n_test là số nguyên; số trial dương, seed = 42. Within có đúng 5 fold 0–4 mỗi subject; cross có đúng một dòng mỗi subject/condition.
- n_test nhất quán giữa mô hình/cửa sổ, tổng test within bằng cross và khớp số trial giữ lại theo task trong log. `tgt_std` có **18.221 trial**. Cả **3.360 giá trị n_train** khớp outer-train suy ra từ log/fold; theo runner đây là số trial trước khi DL tách validation, không phải kích thước phần fit cuối cùng. SHA256 của cả 32 CSV giữ nguyên trước/sau chạy.

### GroupKFold và provenance

Không có `results/cv_cross_subject_folds.csv` và không tạo file thay thế. Cả **16 tổ hợp cross** (560 dòng) có đủ 35 subject đúng một lần, cùng ánh xạ subject → fold; khớp chính xác cả nhãn fold với `evaluation.splits.iter_cross_folds`: chuẩn hóa/sắp xếp subject, `GroupKFold(n_splits=5)`, không shuffle. Mỗi phép chia hiện tại có 7 người test và 28 người train, không giao nhau.

| Fold | Subject test (bỏ tiền tố `sub-`) |
|---:|---|
| 0 | 06, 12, 19, 25, 30, 36, 42 |
| 1 | 05, 11, 18, 24, 29, 34, 40 |
| 2 | 04, 10, 16, 22, 28, 33, 39 |
| 3 | 03, 09, 15, 21, 27, 32, 38 |
| 4 | 01, 07, 13, 20, 26, 31, 37 |

Đây là bằng chứng từ metrics và **code hiện tại**, không phải manifest gốc hoặc bằng chứng về revision/job fit lịch sử của R4. CSV không đủ để chứng minh toàn bộ lịch sử preprocessing, fit hoặc chống leakage. `results/aggregation/F3_manifest.json` là manifest tổng hợp R3, ghi `assignment_source: metrics_consensus`, `supplied_fold_csv_present: false`, `is_original_r4_manifest: false`, danh sách train/test hiện tại, hash code splitter và phiên bản NumPy/scikit-learn. Có **34 hash nguồn**: 32 CSV, log v2, config; không đưa manifest CV giả vào danh sách nguồn.

### Tổng hợp, hình và số liệu

F3 chỉ dùng **16 CSV `tgt_std`**, 1.680 dòng gốc. Tái sử dụng `stats.aggregate_subject_scores`: within lấy trung bình số học 5 fold trong từng subject, không cân theo n_test; cross giữ một score test. Mọi người có trọng số bằng nhau; không dùng fold làm quan sát độc lập. Quartile nội suy tuyến tính; SD mẫu dùng `ddof=1`.

Số liệu đầy đủ nằm trong bảng RQ1 của [Results LaTeX](latex/chapters/04_results.tex) và `results/aggregation/F3_summary.csv`, không sao chép lại bảng median 16 tổ hợp ở đây. Median balanced accuracy cao nhất: `xdawn_rg` within/full **0,949**; EEGNet cross/full **0,924** và hai setting early **0,875 / 0,830**. Với từng mô hình cố định, median early đều thấp hơn full. Shallow within/early có Q1–Q3 **0,587–0,875**; một số score within/early của EEGNet/shallow dưới 0,5. Đây là mô tả group median, không suy p-value từ boxplot.

Hộp là Q1/median/Q3, râu đến quan sát xa nhất trong 1,5 IQR; mọi subject, kể cả outlier, có chấm. Chance = 0,5; hộp xanh là classical, nâu nhạt là DL. PNG `figures/F3_balanced_accuracy.png` có **1920 × 1650 pixel**, metadata 299,9994 dpi ≈ 300 dpi; PDF cùng tên là vector. Đã xem bố cục/nhãn; kích thước gần chiều rộng văn bản giữ font ít nhất 10 pt khi chèn ở chiều rộng dòng. Bảng subject có **560 dòng**, summary có **48 dòng** (16 tổ hợp × ba metric: balanced accuracy, AUC, F1; N, mean, SD, median, Q1/Q3, IQR, min/max). Tên file và gói bàn giao được liệt kê một lần ở phần dưới.

### Bằng chứng kiểm chứng và tái lập

Các file dưới `outputs/r3_f3_validation/`:

- `real_source_hashes_before.json`, `real_inputs_initial.log`: inventory/hash trước khi chạy.
- `verify_real_f3.py`, `real_pipeline.log`, `real_verification.json`: chạy CLI thật, tính độc lập trực tiếp từ CSV bằng `statistics.fmean`; đối chiếu **1.680 giá trị metric theo subject**, **560 điểm vẽ**, **16 hộp** (median, Q1/Q3, râu) và 48 dòng summary. Sai số tuyệt đối lớn nhất **2,22 × 10⁻¹⁶**. Đã kiểm tra DPI/PDF, hash nguồn và không sinh manifest CV gốc.
- **128 số** trong bảng RQ1 LaTeX khớp CSV ở độ làm tròn 3 chữ số; giữ nguyên bảng, cập nhật caption/nhận xét. `tests_f3_real.log` ghi **12/12 tests F3 đạt** (fallback khi thiếu fold CSV, bất đồng giữa mô hình, consensus lệch GroupKFold, validation task đối chứng). `tests_related_real.log` ghi lượt R3/R4/R5 **38/38 đạt, 63,2 giây**; lượt bàn giao mới hơn ở mục kiểm tra cuối tài liệu.

Từ gốc repo, dùng môi trường có requirements và package `eegproj` hoặc `PYTHONPATH=src`:

```powershell
conda activate eeg-oddball
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python scripts/08_aggregate_results.py
python -m unittest discover -s tests -p 'test_r[345]*.py' -v
python -m ruff check src/eegproj/evaluation/aggregate.py src/eegproj/evaluation/plots.py scripts/08_aggregate_results.py tests/test_r3_aggregate.py
python -m ruff format --check src/eegproj/evaluation/aggregate.py src/eegproj/evaluation/plots.py scripts/08_aggregate_results.py tests/test_r3_aggregate.py
```

CLI audit mọi CSV được cung cấp trước khi xuất. Thiếu 16 CSV chính, lệch cohort/fold hoặc metric lỗi thì dừng; nếu có file fold, kiểm tra nó với metrics và GroupKFold. Không sinh fold hoặc huấn luyện mô hình. Code R3: `src/eegproj/evaluation/aggregate.py`, `src/eegproj/evaluation/plots.py`, `scripts/08_aggregate_results.py`, `tests/test_r3_aggregate.py`; tái sử dụng `metrics.validate_metrics_df`, `stats.load_all_metrics`, `stats.aggregate_subject_scores`, `splits.load_valid_subjects`, `splits.iter_cross_folds`, `metrics.metrics_csv_path`, `folds.folds_csv_path`, không sửa evaluation của R4/R5.

Trong sandbox, kiểm tra bằng Python 3.11 Conda `eeg-oddball` với `-S`, thêm site-packages/src vào `PYTHONPATH`, `Library/bin` vào PATH và `os.add_dll_directory` để tránh lỗi lock `.pth`/DLL; `MPLCONFIGDIR` nằm dưới `outputs/`. Đây là điều chỉnh tiến trình, không cài/sửa môi trường. Tests cũ có cảnh báo MNE/pandas về `DataFrame.set_index(inplace=True)`; không sửa code thành viên khác để xử lý cảnh báo. Giới hạn TeX, F2, notebook và phạm vi gói được ghi tập trung ở phần bàn giao/kiểm tra cuối.

## Gói output local

- **Phiên bản:** `r3_outputs_v2_20261123_review_v1`.
- **ZIP:** `outputs/r3_handoff/r3_outputs_v2_20261123_review_v1.zip`.
- **Kích thước:** **11.395.205 byte**.
- **SHA256:** `6712db564201bcba03c2e0a8157209342594fbd00bd877c56fd4f1bec1eee7cf`.
- **Checksum sidecar:** cùng đường dẫn, hậu tố `.zip.sha256`.
- **47 file**: 44 payload được liệt kê theo đường dẫn/byte/SHA256, cộng ba file metadata gói. ZIP đã qua CRC và giải nén vào thư mục tạm để đối chiếu toàn bộ hash payload.
- Gói đã tạo và kiểm tra **trên máy**, để chủ nhánh chuyển lên Drive; chưa có URL hoặc xác nhận tiếp nhận. Không giả định đã phát hành.
- HEAD tham chiếu lúc đóng gói: `b873a91ae11011aca481519f1e431a6badcbae83`; thay đổi R3 vẫn chưa commit. Không coi HEAD này là revision chính xác của job ERP/F3 lịch sử.

Danh sách đầy đủ nằm tại **`outputs/r3_handoff/package/FILES.csv`**; trong ZIP có cùng đường dẫn. `MANIFEST.json` ghi phạm vi, provenance, trạng thái duyệt và phần không kèm theo; `README.txt` hướng dẫn giải nén. Ba metadata này nằm dưới `outputs/`, nên khi giải nén không tạo output ngoài vùng gitignore. Bảng hash loại trừ chính ba metadata để tránh tự tham chiếu.

Tám tên hình, mỗi tên có **cả `.png` và `.pdf`** trong `figures/`:

| Tên | Nội dung |
|---|---|
| `F2_erp_grand_average` | ERP Pz/CPz/Cz, mean ± SEM |
| `F2_difference_wave` | Target − standard ghép cặp |
| `F2_erp_topomap` | Hình F2 được LaTeX tham chiếu |
| `F2_topomap_timecourse` | Topomap theo các thời điểm cấu hình |
| `F2_p300_candidate_distribution` | Phân bố biên độ/latency Pz, giữ điểm có cờ |
| `F2_p300_candidate_review` | ERP/ứng viên theo từng subject |
| `R3_behaviour_comparison` | Bảng đối chiếu hành vi dạng hình |
| `F3_balanced_accuracy` | Balanced accuracy theo mô hình/setting/window |

Bảng và manifest:

- `results/erp/`: `grand_average_summary.csv`, `subject_epoch_counts.csv`, `subject_electrode_positions.csv`, `topomap_subject_epoch_counts.csv`, `topomap_window_summary.csv`, `topomap_timecourse_summary.csv`; bốn CSV `p300_candidate_peaks`, `p300_candidate_summary`, `p300_candidate_review`, `p300_primary_paired`; CSV dẫn xuất `p300_pz_report_summary`; bốn CSV `behaviour_subjects`, `behaviour_exclusions`, `behaviour_summary`, `behaviour_vs_paper`; ba JSON `topomap_manifest`, `p300_candidate_manifest`, `behaviour_manifest`.
- `results/aggregation/`: `F3_subject_scores.csv`, `F3_summary.csv`, `F3_manifest.json`.
- `outputs/`: ba JSON kiểm chứng ERP lịch sử; JSON kiểm chứng F3 thật; JSON kiểm chứng output/P300 và cú pháp LaTeX của đợt này; log 38 tests. Đường dẫn cụ thể của cả 44 payload ở `FILES.csv`.

Không kèm EEG/FIF/checkpoint, 32 CSV nguồn R4/R5, hình F1/F4/F5 hoặc hình QC R2. Các file này không cần để chèn hình R3 đã có vào LaTeX; để chạy lại F3 cần nhận CSV nguồn R4/R5 riêng. `F3_manifest.json` là manifest **tổng hợp R3**, không phải manifest CV gốc.

## Giải nén và dùng với main/LaTeX

Giữ ZIP và checksum ngoài Git. Từ **gốc repo** sau khi lấy ZIP:

```powershell
Get-FileHash -LiteralPath 'outputs/r3_handoff/r3_outputs_v2_20261123_review_v1.zip' -Algorithm SHA256
Expand-Archive -LiteralPath 'outputs/r3_handoff/r3_outputs_v2_20261123_review_v1.zip' -DestinationPath .
```

So hash với giá trị trên/sidecar trước khi giải nén. Nếu đã có file khác nội dung, sao lưu/đối chiếu trước; không tự thêm `-Force`. ZIP chỉ ghi vào `figures/`, `results/`, `outputs/`, không chứa code/config/tài liệu tracked.

Sau giải nén, hai file `figures/F2_erp_topomap.pdf` và `figures/F3_balanced_accuracy.pdf` tồn tại. Build từ `reports/latex/`; `main.tex` dùng `\figdir=../../figures`. Trên Overleaf, chép PDF vào `figures/` của project và đổi `\figdir` thành `figures`. Khung `Missing figure` không chứng minh đã có đủ hình. F1/F4/F5 phải nhận từ các vai phụ trách trước khi chốt báo cáo toàn nhóm.

## Kiểm tra và giới hạn trước PR

- Ruff check và format --check trong phạm vi evaluation/tests, script F3, notebook **02–06**: đạt (23 file định dạng đạt). Syntax Python đạt.
- **38/38 tests liên quan R3/R4/R5 đạt** trong đợt hoàn thiện, 45,4 giây; log nằm trong ZIP. Tests chỉ dùng dữ liệu giả lập ở thư mục tạm, không huấn luyện lại nghiên cứu.
- Kiểm tra tĩnh LaTeX: ngoặc, cặp môi trường, dấu toán, label, bảng P300 khớp CSV và 128 số RQ1 khớp summary: đạt. Đường dẫn F2/F3 giải quyết được.
- **Chưa compile TeX đầy đủ:** không có `xelatex`, `latexmk`, `tectonic`; còn build/phân trang và kiểm tra font F2 ở bản in cuối trên môi trường nhóm.
- Lệnh rộng `ruff check src/eegproj/evaluation tests notebooks` còn **3 lỗi I001 cũ ở notebook 01 của R2**, không sửa ngoài phạm vi R3. Nếu CI chạy phạm vi này, R2/PM cần xử lý hoặc ghi nhận trước merge.
- Notebook 02–06 có **45 output, gồm 8 ảnh trong HEAD**. Đã giữ nguyên file; không tự xóa/strip. Cách xử lý thuộc quyết định nhóm, không khẳng định notebook nguồn đã sạch output.
- Giữ nguyên `.gitignore`, không force-add hình/CSV/ZIP. Chỉ code và tài liệu R3 phù hợp nằm trong thay đổi PR; không stage/commit/push/PR tự động.

Đủ chuẩn bị **PR R3 để review code/tài liệu** với gói output local kiểm chứng. Chưa tuyên bố báo cáo cuối đã duyệt hoặc mọi check toàn repo đều xanh. Phần còn phụ thuộc: PM/người phụ trách duyệt F2 và đồng bộ checklist R1; R2 cung cấp PSD–ICA; chủ nhánh upload ZIP và xác nhận người nhận; môi trường nhóm build LaTeX và thống nhất output notebook. Quy tắc P300 không cần xin duyệt lại.
