# Báo cáo Thực hiện, Kiểm định & Bàn giao Vai trò R4 — Classical ML & RQ2

**Người thực hiện:** Trần Đức Trung (`24022477`, [@TrungDev2431](https://github.com/TrungDev2431))  
**Vai trò:** **R4 — Classical ML (+ Feature Extraction) & RQ2 Statistics**  
**Ngày cập nhật:** 2026-11-20  
**Nhánh làm việc:** `Trung` (đồng bộ trên `origin/main` tại commit `8fbbd44` sau PR #5 R3, PR #7 R5, PR #8 R1, PR #9 R2)  
**Dữ liệu & Đối chứng:** Bộ **Pipeline v2** (`preproc_v2_20261118.zip`, **35 usable subjects**, 127 kênh EEG, 250 Hz, trial đúng hành vi) + bộ kết quả **R5 Deep Learning** (`r5_dl_results_20261120_fc0cfaf.zip`, 8 file CSV của `eegnet` và `shallow`).

---

## 1. Bảng Theo dõi Tiến độ Cho Nhóm trưởng (Đối chiếu `docs/04_roles_and_tasks.md`)

| Hạng mục R4 (`docs/04_roles_and_tasks.md`) | Trạng thái | Kết quả & Đường dẫn kiểm chứng |
|---|---|---|
| **1. Module chia CV dùng chung (`within` + `cross`)** | **HOÀN THÀNH** | - `src/eegproj/evaluation/splits.py` (tương thích 100% với `evaluation/folds.py` và `tests/test_r5_dl.py` của R5; lọc đúng `35` usable subjects v2, loại `02, 08, 14, 17, 23, 35, 41`).<br>- Đồng bộ `within_cv_metadata` (`equal_segments`: 5 đoạn liên tiếp đều nhau) và `cross_subject_folds` (`results/cv_cross_subject_folds.csv`: 5 folds × 7 test / 28 train subjects). |
| **2. Module Metric dùng chung & CSV chuẩn §7** | **HOÀN THÀNH** | - `src/eegproj/evaluation/metrics.py` (`bal_acc`, `auc`, `f1_target`, `validate_metrics_df`, `save_metrics_csv` chuẩn 13 cột). |
| **3. Mô hình Classical ML (`lda_tw`, `xdawn_rg`) & Haufe patterns** | **HOÀN THÀNH** | - `src/eegproj/features/time_windows.py` (`TimeWindowAverager` bin 50 ms, `µV`).<br>- `src/eegproj/models/classical.py` (`build_lda_tw`, `build_xdawn_rg` với `_RankSafeXdawnEstimator`, `extract_lda_haufe_patterns`).<br>- `src/eegproj/models/runner.py` + `scripts/02_run_classical_ml.py` (đã sửa căn chỉnh kênh theo tên cho `sub-13`, `sub-42`). |
| **4. Chạy đủ lưới thực nghiệm: 2 models × 2 settings × 2 windows × 3 tasks** | **HOÀN THÀNH** | - Đủ **24/24 file CSV** trong `results/metrics/` (`12 within × 175` + `12 cross × 35` = **`2,520` dòng**, `0` NaN, khớp tuyệt đối `cv_cross_subject_folds.csv` của R5). |
| **5. Thống kê RQ2 (Wilcoxon + Holm) & Hình F4** | **HOÀN THÀNH** | - `src/eegproj/evaluation/stats.py` & `scripts/03_stats_rq2.py`.<br>- Chạy thực tế đối chiếu **Classical-best (`xdawn_rg`) vs DL-best (`eegnet`)** trên 35 subjects → xuất `results/rq2_wilcoxon_holm.csv`, `results/rq2_wilcoxon_holm_auc.csv` và **Hình F4** (`figures/F4_paired_classical_vs_dl.{png,pdf}`, 300 dpi). |
| **6. Kiểm thử đơn vị, Ruff & Nháp Report** | **HOÀN THÀNH** | - `tests/test_r4_ml.py` (6 unit/integration tests) + toàn bộ `tests/` (34/34 tests qua).<br>- `ruff format --check` & `ruff check` sạch lỗi trên toàn bộ code R4.<br>- Nháp tiếng Anh sẵn sàng ghép report tại `reports/methods_ml_rq2_draft.md`. |
| **7. Hạng mục tùy chọn / chờ ghép nhóm** | **LƯU Ý** | - `bp_svm` (tùy chọn): chưa triển khai để giữ đúng phạm vi `configs/config.yaml`.<br>- `LDA activation patterns` trên dữ liệu thật (protocol §9.4): hàm `extract_lda_haufe_patterns` đã sẵn sàng trong `eegproj.models` để R1 gọi vẽ đối chiếu F5. |

---

## 2. Các Lỗi Kỹ thuật Đã Phát hiện & Sửa trong Đợt Kiểm toán Đồng bộ v2

1. **Sửa lỗi lệch thứ tự kênh trong cross-subject Classical ML (`sub-13`, `sub-42`):**
   - Kiểm tra trực tiếp 35 file `.fif` cho thấy `sub-13` và `sub-42` lưu 127 kênh EEG theo thứ tự khác với 33 subject còn lại (tuy vẫn cùng tập tên điện cực). Nếu ghép trực tiếp mảng NumPy theo chỉ số vị trí sẽ làm trộn lẫn điện cực giữa các subject khi huấn luyện `cross`.
   - Đã bổ sung `_align_eeg_channels` trong `src/eegproj/models/runner.py` để sắp xếp lại trục kênh theo tên điện cực (`ref_ch_names`) trước khi ghép, kèm hồi quy kiểm thử trong `tests/test_r4_ml.py`.
2. **Đồng bộ 100% chia fold Within (`equal_segments`) & Cross (`cv_cross_subject_folds.csv`) với R5:**
   - Sau khi `origin/main` cập nhật `evaluation.within.split: equal_segments` và `src/eegproj/evaluation/folds.py` (để khắc phục `block 4` S10 chỉ có 0 target ở `sub-13` và 1 target ở `sub-07/10/11`), `runner.py` đã chuyển sang gọi `within_cv_metadata` cho `within` và `cross_subject_folds` (đọc đúng `results/cv_cross_subject_folds.csv` của R5) cho `cross`.
   - Giữ nguyên hành vi chặt chẽ của `splits.iter_within_folds` khi `s10_blocks` được truyền vào, bảo đảm toàn bộ `tests/test_r5_dl.py` tiếp tục pass 100%.
3. **Khắc phục suy biến hạng trong xDAWN (`_RankSafeXdawnEstimator`):**
   - Sau khi re-reference `average` và loại trung bình `7.51` IC (`0–15` IC), ma trận hiệp phương sai `127 × 127` bị suy biến hạng ($r < 127$). Dùng shrinkage đẳng hướng `'lwf'` cho mẫu ERP trung bình $P$ bơm phương sai vào không gian nhân (null space) của $C_X$. `_RankSafeXdawnEstimator` dùng `'scm'` cho $P$ (`x.shape[1] <= 1000`) và `'lwf'` cho tín hiệu ghép $X$, giúp `xdawn_rg` ổn định trên toàn bộ 35 subjects.
4. **Tuân thủ `.gitignore` mới trên `main` (`CONTRIBUTING.md`):**
   - Không commit các file CSV trong `results/metrics/`, `results/cv_cross_subject_folds.csv`, `results/rq2_*.csv` hay hình `figures/F4*` lên Git; các script tái lập (`scripts/02_run_classical_ml.py`, `scripts/03_stats_rq2.py`) tái sinh toàn bộ cục bộ trong vài phút.

---

## 3. Kết quả Benchmark Classical ML (N = 35 Subjects, Đồng bộ `equal_segments` & R5 Folds)

> *Tổng hợp theo protocol §8 (`aggregate_subject_scores`):* `within` lấy trung bình 5 folds cho từng subject trước; `cross` lấy điểm của từng subject khi nằm trong test fold. Báo cáo `Mean ± SD` (`ddof=1`) và `Median ± IQR` qua **35 subjects**.

### 3.1 Bài toán chính: `tgt_std` (Target `S 6` vs Standard `S 5`)

| Setting | Window | Model | Bal. Acc (Mean ± SD) | Bal. Acc (Median ± IQR) | ROC-AUC (Mean ± SD) | ROC-AUC (Median ± IQR) | Target F1 (Mean ± SD) | Target F1 (Median ± IQR) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `within` | `full` (0–800 ms) | `lda_tw` | `0.9162 ± 0.0665` | `0.9390 ± 0.0770` | `0.9715 ± 0.0455` | `0.9839 ± 0.0298` | `0.8426 ± 0.1245` | `0.8887 ± 0.1339` |
| `within` | `full` (0–800 ms) | **`xdawn_rg`** | **`0.9173 ± 0.0863`** | **`0.9488 ± 0.0736`** | **`0.9774 ± 0.0506`** | **`0.9946 ± 0.0176`** | **`0.8622 ± 0.1564`** | **`0.9093 ± 0.1135`** |
| `within` | `early` (0–450 ms) | `lda_tw` | **`0.8533 ± 0.0952`** | `0.8487 ± 0.1219` | `0.9331 ± 0.0747` | `0.9463 ± 0.0693` | `0.7229 ± 0.1705` | `0.7047 ± 0.2322` |
| `within` | `early` (0–450 ms) | `xdawn_rg` | `0.8526 ± 0.1088` | **`0.8731 ± 0.1177`** | **`0.9371 ± 0.0838`** | **`0.9698 ± 0.0712`** | **`0.7473 ± 0.1919`** | **`0.7713 ± 0.2078`** |
| `cross` | `full` (0–800 ms) | `lda_tw` | `0.8605 ± 0.0621` | `0.8551 ± 0.0957` | `0.9391 ± 0.0488` | `0.9436 ± 0.0455` | `0.6944 ± 0.1057` | `0.7081 ± 0.1329` |
| `cross` | `full` (0–800 ms) | **`xdawn_rg`** | **`0.8690 ± 0.0829`** | **`0.8716 ± 0.0835`** | **`0.9543 ± 0.0444`** | **`0.9614 ± 0.0375`** | **`0.6973 ± 0.1356`** | **`0.7215 ± 0.2052`** |
| `cross` | `early` (0–450 ms) | `lda_tw` | `0.7910 ± 0.0881` | `0.7957 ± 0.0975` | `0.8799 ± 0.0861` | `0.8910 ± 0.0831` | **`0.5514 ± 0.1208`** | **`0.5549 ± 0.1706`** |
| `cross` | `early` (0–450 ms) | **`xdawn_rg`** | **`0.7987 ± 0.0928`** | **`0.8133 ± 0.1186`** | **`0.8999 ± 0.0811`** | **`0.9221 ± 0.0783`** | `0.5475 ± 0.1330` | `0.5521 ± 0.1344` |

- **So sánh nội bộ `xdawn_rg` vs `lda_tw` (`tgt_std`, `full`, Wilcoxon ghép cặp N = 35):**
  - Về `bal_acc`: `xdawn_rg` đạt trung vị cao nhất ở cả hai setting (`0.9488` vs `0.9390` trong `within`, thắng `21/35`, `p = 0.190`, `RBC = +0.257`; `0.8716` vs `0.8551` trong `cross`, thắng `22/35`, `p = 0.084`, `RBC = +0.337`) → **`xdawn_rg` là `classical-best` ở cả `within` và `cross`**.
  - Về `ROC-AUC`: `xdawn_rg` vượt trội có ý nghĩa thống kê rất cao trước `lda_tw` ở cả `within` (thắng `29/35`, `p = 7.30 × 10⁻⁴`, `RBC = +0.632`) và `cross` (thắng `30/35`, `p = 3.15 × 10⁻⁵`, `RBC = +0.756`).
- **Subject khó / biên:** `sub-07` (`hit_rate = 0.605`, `31` target sạch) thấp nhất ở `within` (`lda_tw = 0.6692`, `xdawn_rg = 0.5511`, `eegnet = 0.6350`), còn `sub-28` thấp nhất ở `cross` cho classical (`lda_tw = 0.7213`, `xdawn_rg = 0.6271`). Các subject có nối file/lệch kênh/thiếu `block 4` giữ hiệu năng rất tốt (`sub-13`: `0.8819` within / `0.8716` cross; `sub-27`: `0.9917` within / `0.9638` cross; `sub-42`: `0.9778` within / `0.8530` cross cho `xdawn_rg`).

### 3.2 Bài toán đối chứng 1 (`dis_std`: Distractor vs Standard) & 2 (`tgt_dis`: Target vs Distractor)

| Task | Setting | Window | Model | Bal. Acc (Mean ± SD) | Bal. Acc (Median ± IQR) | ROC-AUC (Mean ± SD) | ROC-AUC (Median ± IQR) | Target F1 (Mean ± SD) |
|---|---|---|---|---:|---:|---:|---:|---:|
| `dis_std` | `within` | `full` | `lda_tw` | **`0.8471 ± 0.0879`** | `0.8499 ± 0.1012` | `0.9266 ± 0.0823` | `0.9557 ± 0.0640` | `0.7184 ± 0.1612` |
| `dis_std` | `within` | `full` | `xdawn_rg` | `0.8379 ± 0.1087` | **`0.8571 ± 0.1026`** | **`0.9360 ± 0.0829`** | **`0.9625 ± 0.0456`** | **`0.7221 ± 0.1969`** |
| `dis_std` | `within` | `early` | `lda_tw` | **`0.8075 ± 0.1053`** | **`0.8144 ± 0.1127`** | `0.8917 ± 0.1030` | **`0.9266 ± 0.1023`** | `0.6352 ± 0.1897` |
| `dis_std` | `within` | `early` | `xdawn_rg` | `0.7932 ± 0.1240` | `0.8061 ± 0.1469` | **`0.8959 ± 0.1152`** | `0.9240 ± 0.0833` | **`0.6398 ± 0.2322`** |
| `dis_std` | `cross` | `full` | `lda_tw` | `0.8053 ± 0.0883` | `0.8178 ± 0.1173` | `0.8833 ± 0.0816` | `0.9125 ± 0.0872` | `0.5811 ± 0.1394` |
| `dis_std` | `cross` | `full` | `xdawn_rg` | **`0.8299 ± 0.0786`** | **`0.8303 ± 0.0966`** | **`0.9106 ± 0.0713`** | **`0.9331 ± 0.0872`** | **`0.6108 ± 0.1381`** |
| `dis_std` | `cross` | `early` | `lda_tw` | `0.7326 ± 0.0928` | `0.7347 ± 0.1247` | `0.8146 ± 0.1024` | `0.8195 ± 0.1347` | `0.4562 ± 0.1226` |
| `dis_std` | `cross` | `early` | `xdawn_rg` | **`0.7495 ± 0.1026`** | **`0.7566 ± 0.1342`** | **`0.8430 ± 0.1008`** | **`0.8532 ± 0.0979`** | **`0.4653 ± 0.1389`** |
| `tgt_dis` | `within` | `full` | `lda_tw` | `0.7362 ± 0.0918` | `0.7472 ± 0.1126` | `0.8059 ± 0.0964` | `0.8024 ± 0.1213` | `0.7377 ± 0.0993` |
| `tgt_dis` | `within` | `full` | `xdawn_rg` | **`0.7652 ± 0.1167`** | **`0.7841 ± 0.1577`** | **`0.8435 ± 0.1172`** | **`0.8719 ± 0.1604`** | **`0.7626 ± 0.1241`** |
| `tgt_dis` | `within` | `early` | `lda_tw` | `0.6178 ± 0.0905` | **`0.6272 ± 0.1152`** | `0.6646 ± 0.1097` | **`0.6824 ± 0.1459`** | `0.6207 ± 0.0927` |
| `tgt_dis` | `within` | `early` | `xdawn_rg` | **`0.6361 ± 0.0854`** | `0.6205 ± 0.1231` | **`0.6900 ± 0.1114`** | `0.6707 ± 0.1617` | **`0.6402 ± 0.0955`** |
| `tgt_dis` | `cross` | `full` | `lda_tw` | `0.6863 ± 0.0782` | `0.7135 ± 0.1184` | `0.7691 ± 0.0896` | `0.7882 ± 0.1303` | `0.6795 ± 0.1124` |
| `tgt_dis` | `cross` | `full` | `xdawn_rg` | **`0.7096 ± 0.0814`** | **`0.7149 ± 0.1198`** | **`0.8145 ± 0.0844`** | **`0.8403 ± 0.1122`** | **`0.6797 ± 0.1432`** |
| `tgt_dis` | `cross` | `early` | `lda_tw` | `0.6082 ± 0.0720` | `0.6053 ± 0.0809` | `0.6596 ± 0.0929` | `0.6609 ± 0.1219` | `0.6023 ± 0.1145` |
| `tgt_dis` | `cross` | `early` | `xdawn_rg` | **`0.6231 ± 0.0823`** | **`0.6198 ± 0.0945`** | **`0.6914 ± 0.0924`** | **`0.6832 ± 0.1248`** | **`0.6054 ± 0.1377`** |

---

## 4. Kết quả Chính thức RQ2 (Classical vs DL) & Hình F4 (`N = 35`)

Theo protocol §8, trong cả hai setting `within` và `cross`, mô hình tốt nhất theo trung vị `bal_acc` cửa sổ `full` là **`xdawn_rg`** (`classical-best`) và **`eegnet`** (`dl-best`). Bảng kết quả từ `scripts/03_stats_rq2.py`:

### 4.1 So sánh theo Balanced Accuracy (`results/rq2_wilcoxon_holm.csv`, Hình `figures/F4_paired_classical_vs_dl.{png,pdf}`)

| Setting | Window | `xdawn_rg` (Med ± IQR) | `eegnet` (Med ± IQR) | Δ Med (`DL − Cls`) | Wilcoxon *W* | *p* (raw) | *p* (Holm) | RBC (*r*) | Kết luận (α = 0.05) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `within` | `full` | **0.9488 ± 0.0736** | 0.9412 ± 0.0665 | `-0.0039` | `294.0` | `0.740` | `0.740` | `-0.067` | **Không khác biệt** |
| `within` | `early` | 0.8731 ± 0.1177 | **0.8746 ± 0.1464** | `-0.0112` | `233.0` | `0.184` | `0.369` | `-0.260` | **Không khác biệt** |
| `cross` | `full` | 0.8716 ± 0.0835 | **0.9236 ± 0.0822** | **`+0.0202`** | `80.0` | `4.16e-5` | **`1.66e-4`** | **`+0.746`** | **DL thắng có ý nghĩa** |
| `cross` | `early` | 0.8133 ± 0.1186 | **0.8303 ± 0.0957** | `+0.0158` | `182.0` | `0.0287` | `0.0862` | `+0.422` | Xu hướng DL (không qua Holm) |

### 4.2 So sánh theo ROC-AUC (`results/rq2_wilcoxon_holm_auc.csv`, Hình `figures/F4_paired_classical_vs_dl_auc.{png,pdf}`)

| Setting | Window | `xdawn_rg` (Med ± IQR) | `eegnet` (Med ± IQR) | Δ Med (`DL − Cls`) | Wilcoxon *W* | *p* (raw) | *p* (Holm) | RBC (*r*) | Kết luận (α = 0.05) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `within` | `full` | **0.9946 ± 0.0176** | 0.9812 ± 0.0276 | **`-0.0057`** | `53.0` | `2.58e-6` | **`1.03e-5`** | **`-0.832`** | **Classical thắng có ý nghĩa** |
| `within` | `early` | **0.9698 ± 0.0712** | 0.9447 ± 0.1155 | **`-0.0158`** | `78.0` | `3.46e-5` | **`6.92e-5`** | **`-0.752`** | **Classical thắng có ý nghĩa** |
| `cross` | `full` | 0.9614 ± 0.0375 | **0.9793 ± 0.0308** | **`+0.0108`** | `63.0` | `7.84e-6` | **`2.35e-5`** | **`+0.800`** | **DL thắng có ý nghĩa** |
| `cross` | `early` | **0.9221 ± 0.0783** | 0.9193 ± 0.1065 | `+0.0056` | `245.0` | `0.259` | `0.259` | `+0.222` | **Không khác biệt** |

**Phát hiện trọng tâm trả lời RQ2:** Tồn tại **sự phân tách rõ rệt theo quy mô dữ liệu huấn luyện**:
- Trong **within-subject** (~430 trial train/fold mỗi người), mô hình hình học Riemann trên xDAWN (`xdawn_rg`) đạt `balanced accuracy` ngang ngửa EEGNet (`0.9488` vs `0.9412`, `p_Holm = 0.740`) và **vượt trội có ý nghĩa thống kê ở `ROC-AUC`** trên cả hai cửa sổ `full` (`0.9946` vs `0.9812`, `p_Holm = 1.03e-5`) và `early` (`0.9698` vs `0.9447`, `p_Holm = 6.92e-5`).
- Ngược lại, trong **cross-subject** (gộp 28 người ≈ 14.5k trial train/fold), **`eegnet` vượt trội có ý nghĩa thống kê** ở `full` window cả về `bal_acc` (`0.9236` vs `0.8716`, `+0.0202`, `p_Holm = 1.66e-4`, `RBC = +0.746`) lẫn `ROC-AUC` (`0.9793` vs `0.9614`, `p_Holm = 2.35e-5`, `RBC = +0.800`), cho thấy mạng nơ-ron tích chập trích xuất đặc trưng bất biến liên cá thể tốt hơn phép chiếu covariance tuyến tính khi dữ liệu đủ lớn.

---

## 5. Phân tích Đối chứng Nhiễu Vận động (Hỗ trợ RQ3)

- **Chuyển từ `full` (0–800 ms) xuống `early` (0–450 ms):**
  - `tgt_std`: giảm có ý nghĩa `-0.0629` (`within lda_tw`), `-0.0647` (`within xdawn_rg`), `-0.0695` (`cross lda_tw`), `-0.0703` (`cross xdawn_rg`), đều có $p \le 9.84 \times 10^{-9}$, nhưng vẫn giữ vững mức **`0.7910–0.8533`** (`AUC = 0.8799–0.9371`) trước khi phần lớn phản ứng bấm nút diễn ra (`RT = 543.79 ± 61.33 ms`).
  - `dis_std` (cả hai lớp **không bấm phím**): vẫn đạt `0.8471` (`within lda_tw`, `AUC = 0.9266`) và `0.8299` (`cross xdawn_rg`, `AUC = 0.9106`) ở `full`; khi cắt về `early` vẫn giảm `-0.0396` đến `-0.0804` ($p \le 9.74 \times 10^{-7}$), chứng tỏ phần `450–800 ms` vẫn chứa thông tin **P3b/P3a nhận thức muộn** chứ không thuần túy là vận động.
  - `tgt_dis` (cùng hiếm ~12%, chỉ `target` bấm phím): `early` chỉ đạt `0.6082–0.6361` và tăng vọt **`+0.1184` đến `+0.1291`** ở `within` (`+0.0781` đến `+0.0865` ở `cross`, $p \le 9.74 \times 10^{-7}$) khi mở tới `800 ms`. Kết hợp với phân tích IG của R1 (`reports/explain_dl_draft.md` chỉ ra đỉnh |IG| của `eegnet` nằm ở vùng đỉnh P300 lúc 456–511 ms chứ không tập trung ở vùng vận động trái sau 600 ms), có thể kết luận phân loại trước 450 ms dựa chủ yếu vào ERP nhận thức, còn phần tăng thêm ở `tgt_dis` sau 450 ms là giao thoa giữa đỉnh P3b mục tiêu và tín hiệu vận động bấm phím.
- **Tương quan Spearman với hành vi/chất lượng tiền xử lý (`within full tgt_std`, N = 35):** `bal_acc` tương quan thuận với `hit_rate` (`lda_tw`: $\rho=+0.603, p=1.26\times 10^{-4}$; `xdawn_rg`: $\rho=+0.382, p=0.0237$) và `target_after` ($\rho=+0.398, p=0.0178$; $\rho=+0.351, p=0.0384$), tương quan nghịch với `rt_mean_ms` ($\rho=-0.466, p=0.0048$; $\rho=-0.516, p=0.0015$), và **không** tương quan với số IC bị loại `n_ic_removed` ($p \ge 0.59$).

---

## 6. Hướng dẫn Tái lập & Bàn giao cho R1, R3, R5

```powershell
# 1. Lấy dữ liệu v2 + kết quả R5 DL (tự động cài results/cv_cross_subject_folds.csv & 8 DL CSVs)
python scripts/00_get_data.py
python scripts/04_get_dl_results.py

# 2. Chạy toàn bộ 24 tổ hợp Classical ML (sử dụng đúng equal_segments & cv_cross_subject_folds.csv của R5)
python scripts/02_run_classical_ml.py

# 3. Chạy thống kê RQ2 (Wilcoxon + Holm) & xuất Hình F4
python scripts/03_stats_rq2.py --metric bal_acc
python scripts/03_stats_rq2.py --metric auc --out-csv results/rq2_wilcoxon_holm_auc.csv --fig-stem figures/F4_paired_classical_vs_dl_auc

# 4. Chạy kiểm thử đơn vị toàn nhóm & lint code R4
python -m unittest discover -s tests -v
python -m ruff check src/eegproj/evaluation/splits.py src/eegproj/evaluation/stats.py src/eegproj/evaluation/__init__.py src/eegproj/features src/eegproj/models/classical.py src/eegproj/models/runner.py src/eegproj/models/__init__.py scripts/02_run_classical_ml.py scripts/03_stats_rq2.py tests/test_r4_ml.py
```

- **Bàn giao cho R1 (Explainability — protocol §9.4):** Gọi `from eegproj.models import build_classical_model, extract_lda_haufe_patterns` để tính forward activation pattern `pattern_uv` kích thước `(127, 16)` (µV) của `lda_tw` đối chiếu trực tiếp với spatial filter & IG của EEGNet.
- **Bàn giao cho R3 (Visualization — Hình F3):** Cả `32` file CSV (`24` classical + `8` dl) đều nằm trong `results/metrics/` theo đúng schema 13 cột §7; bảng tổng hợp RQ2 được tách riêng tại `results/rq2_wilcoxon_holm*.csv` nên không ảnh hưởng hàm đọc `results/metrics/*.csv`.
