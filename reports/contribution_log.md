# Contribution log

Ghi **ngay sau khi xong mỗi việc**. Đây là căn cứ để chốt % đóng góp trong report (điểm cá nhân được điều chỉnh theo đóng góp).

| Ngày | Thành viên | Vai | Việc đã làm | Link PR / Issue / commit |
|---|---|---|---|---|
| 2026-11-18 | Bảo | R3 — ERP & Visualization | Triển khai notebook 02–06: ERP/SEM và difference wave, topomap/F2, ứng viên P300, hành vi nhóm 39/35; thêm module, cấu hình hiển thị và hướng dẫn chạy. QC/P300 chờ duyệt. | [b97a0684](https://github.com/tbmdemi/EEG_26_27/commit/b97a0684a14a1d999b5f73025d04643220eb77c0) |
| 2026-11-18 | Bảo | R3 — ERP & Visualization | Sửa kiểm tra cohort/metadata v2, lint và tương thích pandas; đối chiếu NumPy, kiểm tra 35 FIF, thêm 8 test; viết nháp Results – ERP và bàn giao. | Review sau `b97a0684`; chưa commit |
| 2026-11-19 | Bảo | R3 — ERP & Visualization | Chạy lại 02–06: 26/26 ô đạt; Ruff và 8 test đạt. Dọn 20 file tạm/cache; gộp ghi chú phương pháp vào bản bàn giao và rút gọn tài liệu. QC/P300/F3 còn chờ. | Thay đổi sau `b97a0684`; chưa commit/push |
| 2026-11-19 | Long | R2 - Signal Preprocessing | Modular R2 pipeline; source-event/FIF audit; visual summary QC of 35 subjects; raw-data correction candidates for sub-10/sub-13; Methods and 42-subject/epoch tables; 11 regression tests passed. Candidate release decision remains open. | `fbc41f4a` (included in this PR); [Issue #6](https://github.com/tbmdemi/EEG_26_27/issues/6) |
| 2026-11-19 | Giang | R5 — Deep Learning | Triển khai EEGNet + ShallowConvNet (`models/dl.py`), runner theo lưới setting × window × task có resume (`models/dl_runner.py`, `scripts/04_train_dl.py`), file fold cố định `evaluation/folds.py`; dùng nguyên module CV/metric của R4; checkpoint/predictions cho R1; 12 test (`tests/test_r5_dl.py`); tự sắp kênh theo tên (sub-13, sub-42 lưu khác thứ tự); đề xuất + triển khai chia within thành 5 đoạn đều nhau (block S10 cuối quá ngắn, sub-13 không có target); thử trên 1 subject thật; nháp *Methods – DL*. Chờ chạy full 35 subject. | Nhánh `feat/dl-r5`; chưa commit |
| 2026-11-20 | Giang | R5 — Deep Learning | Chạy đủ 8 tổ hợp DL trên 35 subject (eegnet/shallow × within/cross × full/early, `tgt_std`; 4.6 giờ GPU); tổng hợp bảng kết quả + Wilcoxon/Holm EEGNet vs Shallow vào Methods – DL; EEGNet là DL-best. Checkpoint/predictions để ở `results/` (chia sẻ qua Drive cho R1/R3/R4). | Nhánh `feat/dl-r5`; chưa commit |
| 2026-11-20 | Minh | R1 - Explainability | Giải thích EEGNet (protocol mục 9.2-9.3): spatial pattern Haufe và mức giảm AUC khi tắt từng filter (16 filter, 5 fold cross); Integrated Gradients theo kênh và thời gian trên target đoán đúng (cross + within, full + early, 35 subject); kiểm tra lại quyết định từng checkpoint khớp `results/predictions/`; bảng ROI P300 vs vận động trái + Wilcoxon; hình F5a/F5b; 5 test (`tests/test_r1_explain.py`); nháp Methods/Results RQ3 (`reports/explain_dl_draft.md`). Code: `explain/dl.py` (tính toán), `explain/plots.py` (hình), `scripts/05_explain_dl.py`, comment tiếng Việt; ghi các quyết định phương pháp vào changelog protocol (chờ nhóm review). LDA patterns chờ R4. | Nhánh `tbm`; chưa commit |
| 2026-11-20 | Long | R2 - Signal Preprocessing | Align current audit/candidate guidance; integrate latest main while preserving R1/R5 contributions and R2 provenance. Candidate data remain separate from v2. | R2 handoff PR |
| 2026-11-14 | Trung | R4 — Classical ML | Xây dựng module chia CV dùng chung (`evaluation/splits.py`) và bộ ghi/kiểm tra metric 13 cột (`evaluation/metrics.py`) trên nhánh `Trung` (R5 tái sử dụng nguyên bản). | Nhánh `Trung` (`2f1f7f6`, `0c83883`) |
| 2026-11-20 | Trung | R4 — Classical ML | Đồng bộ nhánh `Trung` trên `origin/main` (`8fbbd44`): triển khai `TimeWindowAverager` (`features/time_windows.py`), `lda_tw` + `xdawn_rg` (`_RankSafeXdawnEstimator`) + `extract_lda_haufe_patterns` cho R1 (`models/classical.py`), `models/runner.py` (đồng bộ `equal_segments`, `cv_cross_subject_folds.csv` và sửa căn chỉnh kênh theo tên điện cực cho `sub-13`/`sub-42`), module thống kê RQ2 + Hình F4 (`evaluation/stats.py`, `scripts/02_run_classical_ml.py`, `scripts/03_stats_rq2.py`), `6` test (`tests/test_r4_ml.py`), chạy lại đủ 24 tổ hợp Classical ML trên 35 subject + chạy thống kê RQ2 thật đối chiếu với DL của R5 (`xdawn_rg` vs `eegnet`) + xuất Hình F4, viết báo cáo bàn giao `reports/r4_evaluation_review.md` và nháp report `reports/methods_ml_rq2_draft.md`. | Nhánh `Trung` |
| 2026-11-20 | Minh | R1 - Explainability | LDA patterns (protocol mục 9.4): fit lại `lda_tw` của R4 trên 5 fold cross, pattern Haufe theo bin 50 ms (khớp đúng bal_acc/AUC của R4), so với bản đồ P300 và vùng vận động trái; hình F5c; test so khớp với `extract_lda_haufe_patterns` (`explain/lda.py`, `scripts/06_explain_lda.py`). Hình F1 sơ đồ pipeline (`scripts/07_plot_pipeline.py`). Sửa bố cục colorbar F5b. Cập nhật nháp RQ3. Bắt đầu train EEGNet cho task đối chứng `dis_std`, `tgt_dis`. | Nhánh `tbm`; chưa commit |
| 2026-11-21 | Minh | R1 - Explainability | Đối chứng nhiễu vận động: train EEGNet cho `dis_std`, `tgt_dis` (8 tổ hợp, code R5 commit f865c97, Kaggle T4, 81 phút); IG cho cả 3 task (sửa `05_explain_dl.py` để chạy theo task, hình có hậu tố task); so vùng vận động trái với mức nền của `dis_std`: P300 là nguồn chính cho target vs standard, có thêm phần nhỏ từ chuẩn bị vận động (rõ nhất ở `tgt_dis`). Viết vào nháp RQ3. | Nhánh `tbm`; chưa commit |

## Tổng kết % đóng góp (chốt ở mốc M6)

| Thành viên | Vai trò | Đóng góp chính | % |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| **Tổng** | | | **100%** |

Cả nhóm xác nhận (ghi tên + ngày):
