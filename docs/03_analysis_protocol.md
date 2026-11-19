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
| **Cross-subject** | **2 fold theo subject**: train sub-01 → test sub-03 và ngược lại | Không subject nào xuất hiện ở cả train và test; chỉ là kết quả phụ (train trên 1 người) |

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

## 7. Định dạng kết quả (bắt buộc — để ghép bảng/thống kê tự động)

Mỗi lần chạy ghi **một file** `results/metrics/<model>_<setting>_<window>_<task>.csv` với các cột:

```
subject, fold, model, family, setting, window, task, bal_acc, auc, f1_target, n_train, n_test, seed
```

- `family` ∈ {`classical`, `dl`}; `setting` ∈ {`within`, `cross`}; `window` ∈ {`full`, `early`}; `task` ∈ {`tgt_std`, `dis_std`, `tgt_dis`}.
- Với `within`, mỗi (subject, fold) là một dòng. Với `cross`, cột `subject` = subject trong test fold (tính metric **theo từng subject** trong fold đó) → mỗi subject có đúng một dòng cho mỗi mô hình.

## 8. Thống kê cho RQ2

- So sánh **mô hình classical tốt nhất vs DL tốt nhất**, theo cặp trên từng **(subject, fold)** của within-subject: 2 subject × 5 fold = **10 cặp**, **Wilcoxon signed-rank** (two-sided).
- Lý do: chỉ có 2 subject nên không thể kiểm định theo subject. Các fold cùng subject không hoàn toàn độc lập → **phải nêu rõ là hạn chế trong report**; kết quả chỉ mang tính thăm dò, không khái quát cho quần thể.
- Nhiều phép so sánh → hiệu chỉnh **Holm**.
- Báo kèm: trung vị chênh lệch + effect size (rank-biserial).

## 9. Giải thích mô hình (RQ3)

1. **ERP tham chiếu:** grand-average ERP trên toàn bộ subject dùng được (Pz, CPz, Cz) + difference wave target−standard + topomap 250–600 ms. Đo latency đỉnh P300 thực tế trước khi chốt cửa sổ diễn giải (P3b ở task này có thể muộn, ~400–600 ms, vì 3 ký tự rất giống nhau).
2. **EEGNet:** vẽ trọng số spatial filter (depthwise conv) thành topomap; so với topomap P300.
3. **Saliency:** Integrated Gradients (hoặc gradient × input) → bản đồ kênh × thời gian, trung bình trên trial target đúng.
4. **Classical:** patterns của LDA (Haufe et al. 2014 — chuyển weights thành patterns) để so sánh công bằng với DL.
5. Kết luận dựa trên: đặc trưng nằm ở P300 (đỉnh–trung tâm, ~300–600 ms) hay vận động (C3/vùng trung tâm trái, sau ~450 ms)?

Triển khai hình ERP của R3: các giá trị hiển thị nằm trong mục `erp` của config. Topomap gồm trung bình thời gian 250–600 ms và các lát cắt minh họa tại 300/400/500/600 ms. Ghép dữ liệu theo **tên điện cực** trước khi trung bình qua subject; dùng vị trí điện cực trung bình trong hệ tọa độ đầu để vẽ hình nhóm và ghi rõ trong caption. Đây là biểu diễn tín hiệu trên da đầu. Khoảng vẽ topomap chưa phải quyết định về cửa sổ đo đỉnh P300 theo subject.

Đo đỉnh P300: R3 đã chuẩn bị **đề xuất thăm dò để nhóm review**, chưa chốt phương pháp cho report, ở [bàn giao R3, mục P300](../reports/r3_review_and_handoff.md#p300). Các thông số đề xuất nằm riêng trong `erp.p300`; output mang tên `candidate` và giữ cờ kiểm tra của từng phép đo. Không tự loại thêm subject hoặc xác nhận mọi cực đại là P300.

## 10. Hình bắt buộc cho report

| # | Hình | Người phụ trách |
|---|---|---|
| F1 | Sơ đồ pipeline | PM |
| F2 | ERP target/standard/distractor tại Pz + topomap | Visualization |
| F3 | Bảng/boxplot balanced accuracy theo mô hình × setting × window | Visualization |
| F4 | Paired plot classical-best vs DL-best theo subject (within: trung bình fold trong mỗi người; cross: điểm test của người đó), nhất quán với §8 | Classical ML + DL |
| F5 | Topomap spatial filter EEGNet + saliency kênh × thời gian | Explainability |

Quy ước hình: lưu `figures/F<số>_<mô-tả>.png` (300 dpi) + `.pdf`; font ≥ 10 pt; có đơn vị trục.

---

## Changelog

| Ngày | Thay đổi | Lý do | Người đề xuất |
|---|---|---|---|
| | Phiên bản đầu | | |
| 2026-11-17 | Chỉ dùng **sub-01, sub-03**; cross-subject 2 fold; Wilcoxon theo (subject, fold) | Thu hẹp phạm vi; Wilcoxon theo subject vô nghĩa với n nhỏ | PM |
