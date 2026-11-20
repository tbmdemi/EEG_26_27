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
