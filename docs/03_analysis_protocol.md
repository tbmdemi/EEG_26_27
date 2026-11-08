# 03 — Analysis protocol (bắt buộc tuân theo)

Mục đích: mọi người dùng **cùng một bộ quyết định**, để kết quả ML và DL so sánh được với nhau và report nhất quán.
Tất cả tham số số học nằm trong [`configs/config.yaml`](../configs/config.yaml). **Không hard-code tham số trong code.** Muốn đổi → mở Issue, cả nhóm đồng ý, sửa config + ghi vào mục *Changelog* cuối file này.

---

## 1. Tiền xử lý (Preprocessing)

| Bước | Quyết định |
|---|---|
| Đọc dữ liệu | `mne_bids.read_raw_bids` (giữ thông tin kênh/sự kiện chuẩn BIDS) |
| Kênh | 128 kênh EEG; bỏ kênh không phải EEG khỏi phân tích |
| Lọc | Band-pass **0.1–40 Hz** (FIR, zero-phase); notch **50 Hz** nếu còn nhiễu lưới |
| Kênh hỏng | Phát hiện tự động (độ lệch/biên độ bất thường) + kiểm tra mắt; **nội suy** (spherical spline), ghi danh sách vào log |
| Tham chiếu lại | **Average reference** (sau khi nội suy kênh hỏng) |
| ICA | Fit trên **bản sao lọc high-pass 1 Hz**, rồi áp dụng lên dữ liệu 0.1 Hz. Phân loại component bằng **ICLabel**; loại component `eye blink` và `muscle` có xác suất ≥ 0.8. Ghi số component bị loại cho từng subject |
| Resample | **250 Hz** (sau lọc, trước khi lưu) |
| Epoch | **−200 → 800 ms** quanh onset kích thích; baseline **−200 → 0 ms** |
| Loại epoch | Peak-to-peak > **150 µV** sau ICA → loại. Báo tỉ lệ epoch giữ lại theo lớp |
| Loại subject | Subject giữ < **50%** epoch target → loại khỏi phân tích, ghi lý do |
| Đầu ra | `data/derivatives/preproc/sub-XX_task-oddball_epo.fif` (chứa cả 3 loại sự kiện) + `results/preproc_log.csv` |

ICA được fit trên toàn bộ recording của từng subject (không dùng nhãn) — chấp nhận được, nhưng **phải nêu rõ trong report**.

## 2. Bài toán phân loại

- **Bài toán chính:** `target (S6) = 1` vs `standard (S5) = 0`.
- **Bài toán đối chứng** (dùng cho RQ3 và kiểm tra nhiễu vận động):
  - `distractor (S7)` vs `standard (S5)` — sự kiện hiếm nhưng **không bấm nút**
  - `target (S6)` vs `distractor (S7)` — cùng hiếm, khác nhau ở chỗ **có bấm nút / là mục tiêu**

## 3. Cửa sổ thời gian — kiểm soát nhiễu vận động

Chỉ trial target có bấm nút → mô hình có thể học **tín hiệu vận động** thay vì P300. Vì vậy mọi mô hình chạy trên 2 cửa sổ:

| Tên | Cửa sổ | Ý nghĩa |
|---|---|---|
| `full` | 0–800 ms | Kết quả chính (RQ1) |
| `early` | 0–450 ms | Kết thúc trước phần lớn thời điểm bấm (RT trung bình − 1.5 SD ≈ 444 ms) |

Nếu hiệu năng giảm mạnh từ `full` → `early`, hoặc saliency tập trung ở vùng vận động sau ~450 ms, cần bàn kỹ trong report.

## 4. Thiết lập đánh giá

| Thiết lập | Cách chia | Ghi chú |
|---|---|---|
| **Within-subject** | 5-fold **theo khối thời gian liên tục** (chia recording thành 5 đoạn liên tiếp, ví dụ theo các lần nghỉ `S 10`), không xáo trộn trial | Tránh trial liền kề (tương quan) rơi vào cả train và test |
| **Cross-subject** | **GroupKFold 6 fold theo subject** (~7 subject/fold) | Không subject nào xuất hiện ở cả train và test |

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
- Với `cross`, cột `subject` = subject trong test fold (tính metric **theo từng subject** trong fold đó).

## 8. Thống kê cho RQ2

- So sánh **mô hình classical tốt nhất vs DL tốt nhất**, theo cặp trên từng subject: **Wilcoxon signed-rank** (two-sided).
- Nhiều phép so sánh → hiệu chỉnh **Holm**.
- Báo kèm: trung vị chênh lệch + effect size (rank-biserial).

## 9. Giải thích mô hình (RQ3)

1. **ERP tham chiếu:** grand-average ERP (Pz, CPz, Cz) + difference wave target−standard + topomap 250–600 ms.
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
| F4 | Paired plot classical-best vs DL-best theo subject | Classical ML + DL |
| F5 | Topomap spatial filter EEGNet + saliency kênh × thời gian | Explainability |

Quy ước hình: lưu `figures/F<số>_<mô-tả>.png` (300 dpi) + `.pdf`; font ≥ 10 pt; có đơn vị trục.

---

## Changelog

| Ngày | Thay đổi | Lý do | Người đề xuất |
|---|---|---|---|
| | Phiên bản đầu | | |
