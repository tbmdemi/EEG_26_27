# 03 — Analysis protocol (bắt buộc tuân theo)

Mục đích: mọi người dùng **cùng một bộ quyết định**, để kết quả ML và DL so sánh được với nhau và report nhất quán.
Tất cả tham số số học nằm trong [`configs/config.yaml`](../configs/config.yaml). **Không hard-code tham số trong code.** Muốn đổi → mở Issue, cả nhóm đồng ý, sửa config + ghi vào mục *Changelog* cuối file này.

---

## 1. Tiền xử lý (Preprocessing)

| Bước | Quyết định |
|---|---|
| Phạm vi | **Toàn bộ 42 subject** trừ các subject bị loại (`dataset.exclude`, xem [`02_dataset.md`](02_dataset.md)) |
| Lọc hành vi | **Chỉ giữ trial đúng hành vi**, tính từ `*_events.tsv`: target có bấm trong 100–1400 ms; standard và distractor không bấm trong cửa sổ đó (config `behaviour`). Trial sai bị bỏ trước khi epoch; RT của target lưu trong metadata |
| Đọc dữ liệu | `mne_bids.read_raw_bids` (giữ thông tin kênh/sự kiện chuẩn BIDS); annotation dựng lại từ cột `event_type`; montage từ tọa độ **CapTrak** riêng của từng người |
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

Dữ liệu dùng cho mọi phân tích là bộ do `scripts/01_preprocess.py` tạo ra (pipeline v2, xem `PIPELINE_VERSION` trong `src/eegproj/preprocessing/pipeline.py`). Không dùng bộ dữ liệu tiền xử lý bằng cách khác (ví dụ notebook Colab thử nghiệm) cho kết quả trong report.

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
| **Cross-subject** | **GroupKFold 5 fold theo subject** trên toàn bộ subject dùng được (~37 người) | Không subject nào xuất hiện ở cả train và test. Phân chia fold cố định (seed trong config) và **lưu ra `results/cv_cross_subject_folds.csv`** để mọi mô hình dùng đúng cùng một phân chia |

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

Khối lượng tính toán DL: `within` = ~37 subject × 5 fold mỗi mô hình × cửa sổ (mạng nhỏ, ~400–500 trial train/lần); `cross` = 5 lần train trên ~30 người. Ưu tiên chạy đủ task chính `tgt_std` cho cả 2 setting × 2 cửa sổ; task đối chứng cho DL chỉ khi còn thời gian (classical chạy đủ cả 3 task).

## 7. Định dạng kết quả (bắt buộc — để ghép bảng/thống kê tự động)

Mỗi lần chạy ghi **một file** `results/metrics/<model>_<setting>_<window>_<task>.csv` với các cột:

```
subject, fold, model, family, setting, window, task, bal_acc, auc, f1_target, n_train, n_test, seed
```

- `family` ∈ {`classical`, `dl`}; `setting` ∈ {`within`, `cross`}; `window` ∈ {`full`, `early`}; `task` ∈ {`tgt_std`, `dis_std`, `tgt_dis`}.
- Với `within`, mỗi (subject, fold) là một dòng. Với `cross`, cột `subject` = subject trong test fold (tính metric **theo từng subject** trong fold đó) → mỗi subject có đúng một dòng cho mỗi mô hình.

## 8. Thống kê cho RQ2

- Đơn vị thống kê là **subject** (một giá trị mỗi người, ~37 cặp):
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

## 10. Hình bắt buộc cho report

| # | Hình | Người phụ trách |
|---|---|---|
| F1 | Sơ đồ pipeline | PM |
| F2 | ERP target/standard/distractor tại Pz + topomap | Visualization |
| F3 | Bảng/boxplot balanced accuracy theo mô hình × setting × window | Visualization |
| F4 | Paired plot classical-best vs DL-best theo (subject, fold) | Classical ML + DL |
| F5 | Topomap spatial filter EEGNet + saliency kênh × thời gian | Explainability |

Quy ước hình: lưu `figures/F<số>_<mô-tả>.png` (300 dpi) + `.pdf`; font ≥ 10 pt; có đơn vị trục.

---

## Changelog

| Ngày | Thay đổi | Lý do | Người đề xuất |
|---|---|---|---|
| | Phiên bản đầu | | |
| 2026-11-17 | Chỉ dùng **sub-01, sub-03**; cross-subject 2 fold; Wilcoxon theo (subject, fold) | Thu hẹp phạm vi; Wilcoxon theo subject vô nghĩa với n nhỏ | PM — **đã thay bằng dòng dưới** |
| 2026-11-17 | **Pipeline v2.** (1) Dùng **toàn bộ 42 subject**, loại trước sub-02/14/17 theo bài báo, gắn cờ sub-13/23/24/37/38. (2) **Chỉ giữ trial đúng hành vi** (cửa sổ phản hồi 100–1400 ms), lưu `rt` vào metadata. (3) Loại subject nếu < 40 target đúng; hit rate < 80% chỉ gắn cờ. (4) Đánh dấu BAD quanh chỗ nối recording. (5) Cross-subject: GroupKFold 5 fold theo subject, lưu phân chia fold. (6) Thống kê RQ2 theo **subject** (~37 cặp) thay vì (subject, fold). Dữ liệu v1 (sub-01…05, mọi trial) không dùng nữa | 2 subject không đủ để trả lời RQ1/RQ2; đề bài yêu cầu chỉ dùng trial phản hồi đúng; 3 subject hiểu sai nhiệm vụ làm nhiễu nhãn | PM |
