# 03 — Analysis protocol (bắt buộc tuân theo)

Mục đích: mọi người dùng **cùng một bộ quyết định**, để kết quả ML và DL so sánh được với nhau và report nhất quán.
Tất cả tham số số học nằm trong [`configs/config.yaml`](../configs/config.yaml). **Không hard-code tham số trong code.** Muốn đổi → mở Issue, cả nhóm đồng ý, sửa config + ghi vào mục *Changelog* cuối file này.

---

## 1. Tiền xử lý (Preprocessing)

| Bước | Quyết định |
|---|---|
| Phạm vi | **Toàn bộ 42 subject** trừ các subject bị loại (`dataset.exclude`, xem [`02_dataset.md`](02_dataset.md)) |
| Lọc hành vi | **Chỉ giữ trial đúng hành vi**, tính từ `*_events.tsv`: target có bấm trong 100–1400 ms; standard và distractor không bấm trong cửa sổ đó (config `behaviour`). Trial sai bị bỏ trước khi epoch; RT của target lưu trong metadata |
| Đọc dữ liệu | `mne_bids.read_raw_bids` (giữ thông tin kênh/sự kiện chuẩn BIDS); annotation dựng lại từ cột `event_type`; montage riêng từng subject: ưu tiên **CapTrak `.bvct`**, fallback **BrainVision `.bvef`** khi không có `.bvct`. Với `.bvef`, dùng `mne.channels.read_custom_montage(..., coord_frame="head")` và chuẩn hóa head size mặc định của MNE; ghi nguồn trong summary |
| Kênh | 127 kênh EEG có trong file (FCz là tham chiếu online, không có trong file); bỏ kênh không phải EEG khỏi phân tích |
| Chỗ nối recording | Marker `New Segment` sau t = 0 (sub-37, sub-38): đánh dấu ±0.5 s là `BAD_ACQ_SKIP` → bỏ qua khi lọc, fit ICA và epoch |
| Lọc | Band-pass **0.1–40 Hz** (FIR, zero-phase); notch **50 Hz** nếu còn nhiễu lưới |
| Kênh hỏng | Phát hiện tự động (độ lệch/biên độ bất thường) + kiểm tra mắt; **nội suy** (spherical spline), ghi danh sách vào log |
| Tham chiếu lại | **Average reference** (sau khi nội suy kênh hỏng) |
| ICA | Extended Infomax (99% phương sai), fit trên **bản sao lọc high-pass 1 Hz**, rồi áp dụng lên dữ liệu 0.1 Hz. Phân loại component bằng **ICLabel**; loại component `eye blink` và `muscle` có xác suất ≥ 0.8. Ghi số component bị loại cho từng subject |
| Resample | **250 Hz** (sau lọc, trước khi lưu) |
| Epoch | **−200 → 800 ms** quanh onset kích thích; baseline **−200 → 0 ms** |
| Loại epoch | Peak-to-peak > **150 µV** sau ICA, hoặc chạm vùng BAD → loại. Báo tỉ lệ epoch giữ lại theo lớp |
| Loại subject | (a) Loại trước theo bài báo: sub-02, 14, 17 (hiểu sai nhiệm vụ). (b) < **40** target đúng. (c) Giữ < **50%** target đúng sau khi loại epoch. Ghi lý do vào log. Hit rate < 80% hoặc số kích thích ≠ 660 chỉ **gắn cờ** (cột `flags`) để kiểm tra bằng mắt |
| Đầu ra | `data/derivatives/preproc/sub-XX_task-oddball_epo.fif` (trial đúng của cả 3 loại sự kiện; metadata `subject`, `onset_s`, `block`, `rt`) + `results/preproc_log.csv` (hành vi, kênh hỏng, IC loại, số epoch theo lớp, cờ, lý do loại) |

ICA được fit trên toàn bộ recording của từng subject (không dùng nhãn) — chấp nhận được, nhưng **phải nêu rõ trong report**.

Dữ liệu chính thức dùng cho mọi phân tích hiện là `preproc_v2_20261118.zip` (35 subject, 20.600 epoch), gắn với triển khai v2 lịch sử có entry point `scripts/01_preprocess.py`. Quy trình R2 hiện tại dùng `scripts/02_r2.py`: mặc định audit v2; chỉ tạo dữ liệu mới khi bật `--process`, ghi riêng vào `_candidate/`. Candidate sub-10/sub-13 chưa được duyệt thay v2. Không dùng candidate hoặc dữ liệu từ nguồn xử lý khác cho kết quả report trước khi nhóm duyệt và phát hành phiên bản mới. Xem `02_dataset.md` cho hướng dẫn chạy và bàn giao.

## 2. Bài toán phân loại

- **Bài toán chính:** `target (S6) = 1` vs `standard (S5) = 0`. Chỉ dùng trial đúng hành vi (§1).
- Mất cân bằng lớp ~1:6 (thường 60–81 target đúng và 480–498 standard đúng mỗi người).
- **Bài toán đối chứng** (dùng cho RQ3 và kiểm tra nhiễu vận động):
  - `distractor (S7)` vs `standard (S5)` — sự kiện hiếm nhưng **không bấm nút**
  - `target (S6)` vs `distractor (S7)` — cùng hiếm, khác nhau ở chỗ **có bấm nút / là mục tiêu**

## 3. Cửa sổ thời gian — kiểm soát nhiễu vận động

Sau khi lọc hành vi, **mọi** trial target đều có bấm nút và **không** trial standard/distractor nào có → mô hình có thể học **tín hiệu vận động** thay vì P300. Vì vậy mọi mô hình chạy trên 2 cửa sổ:

| Tên | Cửa sổ | Ý nghĩa |
|---|---|---|
| `full` | 0–800 ms | Kết quả chính (RQ1) |
| `early` | 0–450 ms | Kết thúc trước phần lớn thời điểm bấm (RT trung bình − 1.5 SD ≈ 444 ms) |

Nếu hiệu năng giảm mạnh từ `full` → `early`, hoặc saliency tập trung ở vùng vận động sau ~450 ms, cần bàn kỹ trong report.

## 4. Thiết lập đánh giá

| Thiết lập | Cách chia | Ghi chú |
|---|---|---|
| **Within-subject** | 5-fold **theo khối thời gian liên tục** (chia recording thành 5 đoạn liên tiếp, ví dụ theo các lần nghỉ `S 10`), không xáo trộn trial | Tránh trial liền kề (tương quan) rơi vào cả train và test |
| **Cross-subject** | **GroupKFold 5 fold theo subject** trên toàn bộ subject dùng được (35 người trong bản v2 hiện hành, lấy danh sách từ log) | Không subject nào xuất hiện ở cả train và test. Phân chia fold cố định (seed trong config) và **lưu ra `results/cv_cross_subject_folds.csv`** để mọi mô hình dùng đúng cùng một phân chia |

Quy tắc chống rò rỉ dữ liệu (data leakage):
1. Mọi bước **học từ dữ liệu** (scaler, xDAWN, CSP, PCA, chọn đặc trưng, chọn ngưỡng, early stopping, tuning) chỉ fit trên **train**.
2. Tuning hyperparameter: dùng **nested CV** hoặc tách validation từ phần train. **Không bao giờ** nhìn test fold để chọn tham số.
3. Cân bằng lớp bằng **class weight** (hoặc undersample *chỉ trong train*). Không oversample trước khi chia.
4. Cố định random seed: `seed` trong config.

## 5. Metric

- **Chính:** **balanced accuracy** và **ROC-AUC**.
- Phụ: F1 của lớp target, confusion matrix.
- **Không** báo accuracy thô làm metric chính (đoán toàn "standard" đã được ~86%).
- Chance level: balanced accuracy = 0.5; AUC = 0.5. Có thể kiểm định thêm bằng permutation test cho mô hình tốt nhất.

## 6. Mô hình

### ML truyền thống (bắt buộc ít nhất 2)
| Tên mã | Mô tả |
|---|---|
| `lda_tw` | Trung bình biên độ theo cửa sổ 50 ms × tất cả kênh → **shrinkage LDA** |
| `xdawn_rg` | **xDAWN covariances → tangent space → Logistic Regression** (`pyriemann`) |
| `bp_svm` (tùy chọn) | Band power (delta/theta/alpha/beta) hoặc wavelet → SVM / Random Forest |

### Deep Learning (bắt buộc ít nhất 2)
| Tên mã | Mô tả |
|---|---|
| `eegnet` | EEGNet (braindecode `EEGNetv4`) |
| `shallow` | ShallowFBCSPNet |
| `conformer` (tùy chọn) | EEG-Conformer |

Đầu vào DL: epoch đã chuẩn hóa theo kênh (thống kê tính trên train), resample còn **128 Hz** cho DL nếu cần tiết kiệm tính toán (ghi rõ).

Khối lượng tính toán DL với bản v2 hiện hành: `within` = 35 subject × 5 fold mỗi mô hình × cửa sổ; `cross` = 5 lần train trên khoảng 28 người mỗi lần (số thực tế phụ thuộc phân chia fold). Số trial train phụ thuộc subject và số epoch còn lại. Ưu tiên chạy đủ task chính `tgt_std` cho cả 2 setting × 2 cửa sổ; task đối chứng cho DL chỉ khi còn thời gian (classical chạy đủ cả 3 task).

## 7. Định dạng kết quả (bắt buộc — để ghép bảng/thống kê tự động)

Mỗi lần chạy ghi **một file** `results/metrics/<model>_<setting>_<window>_<task>.csv` với các cột:

```
subject, fold, model, family, setting, window, task, bal_acc, auc, f1_target, n_train, n_test, seed
```

- `family` ∈ {`classical`, `dl`}; `setting` ∈ {`within`, `cross`}; `window` ∈ {`full`, `early`}; `task` ∈ {`tgt_std`, `dis_std`, `tgt_dis`}.
- Với `within`, mỗi (subject, fold) là một dòng. Với `cross`, cột `subject` = subject trong test fold (tính metric **theo từng subject** trong fold đó) → mỗi subject có đúng một dòng cho mỗi mô hình.

## 8. Thống kê cho RQ2

- Đơn vị thống kê là **subject** (một giá trị mỗi người; tối đa 35 cặp với bản v2 hiện hành, kiểm tra đủ kết quả theo subject):
  - `within`: trung bình metric của 5 fold trong từng subject;
  - `cross`: metric trên subject đó khi nó nằm trong test fold.
- So sánh **mô hình classical tốt nhất vs DL tốt nhất** (chọn theo trung vị balanced accuracy, cửa sổ `full`) bằng **Wilcoxon signed-rank** (two-sided), ghép cặp theo subject, riêng cho từng setting (`within`, `cross`).
- Không dùng (subject, fold) làm đơn vị: các fold cùng một người không độc lập.
- Nhiều phép so sánh → hiệu chỉnh **Holm**.
- Báo kèm: trung vị chênh lệch + effect size (rank-biserial).

## 9. Giải thích mô hình (RQ3)

1. **ERP tham chiếu:** grand-average ERP trên toàn bộ subject dùng được (Pz, CPz, Cz) + difference wave target−standard + topomap 250–600 ms. Đo latency đỉnh P300 thực tế trước khi chốt cửa sổ diễn giải (P3b ở task này có thể muộn, ~400–600 ms, vì 3 ký tự rất giống nhau).
2. **EEGNet:** vẽ trọng số spatial filter (depthwise conv) thành topomap; so với topomap P300.
3. **Saliency:** Integrated Gradients (hoặc gradient × input) → bản đồ kênh × thời gian, trung bình trên trial target đúng.
4. **Classical:** patterns của LDA (Haufe et al. 2014 — chuyển weights thành patterns) để so sánh công bằng với DL.
5. Kết luận dựa trên: đặc trưng nằm ở P300 (đỉnh–trung tâm, ~300–600 ms) hay vận động (C3/vùng trung tâm trái, sau ~450 ms)?

Triển khai hình ERP của R3: các giá trị hiển thị nằm trong mục `erp` của config. Topomap gồm trung bình thời gian 250–600 ms và các lát cắt minh họa tại 300/400/500/600 ms. Ghép dữ liệu theo **tên điện cực** trước khi trung bình qua subject; dùng vị trí điện cực trung bình trong hệ tọa độ đầu để vẽ hình nhóm và ghi rõ trong caption. Đây là biểu diễn tín hiệu trên da đầu. Khoảng vẽ topomap chưa phải quyết định về cửa sổ đo đỉnh P300 theo subject.

Đo đỉnh P300: dùng cực đại có dấu trong 300-600 ms, với Pz là kênh chính và CPz/Cz bổ sung. Quy tắc mô tả đã được PM duyệt cho report ngày 2026-11-22: báo median [IQR] tại Pz cho target và distractor, giữ ứng viên có cờ và nêu số ca có cờ. Không tự loại thêm subject hoặc khẳng định mọi cực đại là P300. Các thông số nằm trong `erp.p300`; tên file output giữ nguyên.

## 10. Hình bắt buộc cho report

| # | Hình | Người phụ trách |
|---|---|---|
| F1 | Sơ đồ pipeline | PM (R1) |
| F2 | ERP target/standard/distractor tại Pz + topomap | Visualization |
| F3 | Bảng/boxplot balanced accuracy theo mô hình × setting × window | Visualization |
| F4 | Paired plot classical-best vs DL-best theo subject (within: trung bình fold trong mỗi người; cross: điểm test của người đó), nhất quán với §8 | Classical ML + DL |
| F5 | F5a: topomap spatial pattern EEGNet; F5b: saliency (IG) theo kênh và thời gian; F5c: pattern LDA (mục 9.4). Hình của task đối chứng có hậu tố `_dis_std`, `_tgt_dis` | Explainability (R1) |

Quy ước hình: lưu `figures/F<số>_<mô-tả>.png` (300 dpi) + `.pdf`; font ≥ 10 pt; có đơn vị trục.

---

## Changelog

| Ngày | Thay đổi | Lý do | Người đề xuất |
|---|---|---|---|
| | Phiên bản đầu | | |
| 2026-11-22 | Duyệt quy tắc mô tả đỉnh P300 tại Pz: median [IQR] cho target và distractor; giữ và báo số ứng viên có cờ; không gán danh tính P300 cho mọi cực đại | Chốt cách báo cáo các ứng viên P300 mà không loại người hoặc diễn giải quá mức | PM |
| 2026-11-19 | **Within-subject CV: 5 đoạn liên tiếp đều nhau** (`evaluation.within.split: equal_segments`) thay cho 5 block giữa các lần nghỉ `S 10`. Trial vẫn theo thứ tự ghi, không xáo trộn; mỗi đoạn có số trial gần bằng nhau. Block gốc giữ ở cột `s10_block`. Dùng `eegproj.evaluation.folds.within_cv_metadata` trước `iter_within_folds` | Các block `S 10` lệch nhau nhiều: block cuối ở mọi subject chỉ có ~40–50 trial và 0–4 target (sub-13: 0 target → fold không xác định, code dừng; sub-07/10/11: 1 target → balanced accuracy của fold gần như ngẫu nhiên). Protocol §4 cho phép "5 đoạn liên tiếp"; với cách chia mới, cả 35 subject × 3 task đều có đủ 2 lớp ở mọi fold. **R4 cần dùng cùng cách chia** để RQ2 ghép cặp đúng | R5 — chờ nhóm review |
| 2026-11-19 | **R5 — Deep Learning.** (1) Thêm hyperparameter DL vào `models.dl` (kernel/pooling tính theo giây, `min_delta`); giá trị cũ giữ nguyên. (2) Validation cho early stopping: `within` = 20% trial cuối của mỗi block train; `cross` = 20% subject train (seed + fold). (3) EEGNet/ShallowConvNet viết bằng PyTorch thuần theo đúng kiến trúc gốc thay vì import braindecode `EEGNetv4`. (4) Resample cả epoch 250 → 128 Hz rồi mới cắt cửa sổ. (5) Ghi `results/cv_cross_subject_folds.csv` từ `iter_cross_folds` của R4 và mọi lần chạy sau đọc lại file này. (6) Cột `seed` = seed config; mô hình fold k khởi tạo bằng seed + k | (2) Validation tách theo thời gian/subject giống cấu trúc test, tránh trial liền kề ở cả fit và validation; trên sub-01 cho kết quả tương đương chia ngẫu nhiên phân tầng (0.723 vs 0.734 bal. acc.). (3) Kiểm soát được tên lớp và đọc trọng số cho explainability, không phụ thuộc phiên bản braindecode. (4) Tránh hiệu ứng biên khi resample. (5) Protocol §4 yêu cầu lưu fold | R5 — chờ nhóm review |
| 2026-11-20 | **R1 - Explainability (mục 9.2-9.3, 9.5).** (1) Mục 9.2: vẽ **pattern** của spatial filter EEGNet (Haufe 2014, tính từ hiệp phương sai đầu ra của temporal filter tương ứng, đổi về đơn vị uV) thay cho trọng số thô; trọng số vẫn lưu trong `results/explain/*_patterns.csv`. Dấu của pattern được lật cho cùng chiều với bản đồ target trừ standard trong khoảng 250-600 ms. Mức quan trọng của filter = AUC test giảm bao nhiêu khi thay đầu ra filter bằng giá trị trung bình lúc train. (2) Mục 9.3: Integrated Gradients của logit(target) trừ logit(standard), baseline = trung bình train của từng kênh, 64 bước, trên các trial target đoán đúng ở fold test (cross: mô hình của fold có subject đó trong tập test; within: 5 mô hình của chính subject). Hình dùng trị tuyệt đối của IG; IG có dấu lưu trong file `.npz`. (3) Mục 9.5: vùng P300 = Pz, CPz, POz, CP1, CP2, P1, P2; vùng vận động trái = C3, C1, C5, FC3, CP3; các khoảng thời gian 0-250, 250-450, 450-600, 600-800 ms; so sánh bằng Wilcoxon theo subject (chỉ mô tả, chưa hiệu chỉnh). Các giá trị này là hằng số trong `src/eegproj/explain/dl.py`, **không** thêm vào config. (4) Hình F5 tách thành `F5a_eegnet_spatial_patterns` và `F5b_eegnet_saliency_{cross,within}` | (1) Trọng số filter không đọc được như bản đồ da đầu, còn pattern thì đọc được (Haufe 2014). (2) Giải thích đúng mô hình và đúng trial đã dùng để báo cáo kết quả; IG có dấu của các kênh cạnh nhau hay ngược dấu nên khó đọc. (3) Người tham gia bấm nút bằng tay phải (docs/02_dataset.md) nên vùng vận động nằm bên trái; 450 ms là cuối cửa sổ `early` (mục 3). (4) Gộp vào một hình thì quá nhiều panel | R1, chờ nhóm review |
| 2026-11-21 | **R1 - Explainability (mục 9.4, 9.5, bài toán đối chứng mục 2).** (1) Mục 9.4: pattern LDA (Haufe 2014) tính cho `lda_tw` fit lại trên đúng 5 fold cross-subject, trên đặc trưng bin 50 ms, lấy trung bình qua fold (fit lại khớp đúng bal_acc/AUC của R4); hình F5c. (2) Bài toán đối chứng `dis_std`, `tgt_dis`: chỉ train `eegnet` (DL tốt nhất), cùng code R5, fold và hyperparameter, chạy trên Kaggle T4; giải thích bằng IG như task chính, trên trial lớp dương đoán đúng. (3) Mục 9.5: dùng enrichment vùng vận động của `dis_std` (không lớp nào bấm nút) làm mức nền khi đánh giá nhiễu vận động. Hình và file của task đối chứng có hậu tố `_<task>` | Hoàn thành mục 9 và trả lời câu hỏi nhiễu vận động; không đổi tham số config, CV hay metric | R1 (Minh), chờ nhóm review |
| 2026-11-18 | Review sau `b97a0684`: ghi đúng cohort hiện hành 35/42; làm rõ F4 dùng đơn vị subject theo §8; bổ sung kiểm tra cohort và metadata v2 trong code R3 | Sửa kiểm tra đầu vào và mô tả; giữ nguyên mọi tham số config, CV/metric và trạng thái P300/QC | R3 — đợt review chưa commit |
| 2026-11-18 | Bổ sung số tham chiếu hành vi từ Table 5 vào `erp.behaviour_reference`; bảng R3 tách nhóm trước loại EEG và nhóm ERP, dùng mean ± sample SD qua subject và số đếm hành vi trước loại epoch | Đối chiếu mô tả đúng cỡ mẫu/đơn vị; không đổi quy tắc lọc hành vi, tiền xử lý hay phân loại | R3 — commit `b97a0684`, chờ nhóm review |
| 2026-11-18 | Chuẩn bị đề xuất thăm dò `erp.p300`: ứng viên cực đại có dấu trong 300–600 ms, Pz chính/CPz-Cz bổ sung, target/distractor riêng; cờ biên 20 ms và kiểm tra cực đại. Chưa chốt phương pháp report | Bước tiếp theo sau ERP/topomap, tạo bảng để xem xét phương pháp và các trường hợp có cờ | R3 — commit `b97a0684`, đề xuất chờ nhóm review |
| 2026-11-18 | Cấu hình hóa phần hiển thị ERP đã yêu cầu ở §9–10 (`erp` trong config): Pz/CPz/Cz, topomap 250–600 ms, lát cắt 300/400/500/600 ms, PNG 300 dpi. Ghi rõ ghép theo tên điện cực và hình nhóm dùng vị trí điện cực trung bình | Triển khai hình F2; không thay đổi tiền xử lý, mô hình hay cửa sổ đo đỉnh P300 | R3 — commit `b97a0684`, chờ nhóm review |
| 2026-11-17 | Chỉ dùng **sub-01, sub-03**; cross-subject 2 fold; Wilcoxon theo (subject, fold) | Thu hẹp phạm vi; Wilcoxon theo subject vô nghĩa với n nhỏ | PM — **đã thay bằng dòng dưới** |
| 2026-11-17 | **Pipeline v2.** (1) Dùng **toàn bộ 42 subject**, loại trước sub-02/14/17 theo bài báo, gắn cờ sub-13/23/24/37/38. (2) **Chỉ giữ trial đúng hành vi** (cửa sổ phản hồi 100–1400 ms), lưu `rt` vào metadata. (3) Loại subject nếu < 40 target đúng; hit rate < 80% chỉ gắn cờ. (4) Đánh dấu BAD quanh chỗ nối recording. (5) Cross-subject: GroupKFold 5 fold theo subject, lưu phân chia fold. (6) Thống kê RQ2 theo **subject** (~37 cặp) thay vì (subject, fold). Dữ liệu v1 (sub-01…05, mọi trial) không dùng nữa | 2 subject không đủ để trả lời RQ1/RQ2; đề bài yêu cầu chỉ dùng trial phản hồi đúng; 3 subject hiểu sai nhiệm vụ làm nhiễu nhãn | PM |
