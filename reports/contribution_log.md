# Contribution log

Ghi **ngay sau khi xong mỗi việc**. Đây là căn cứ để chốt % đóng góp trong report (điểm cá nhân được điều chỉnh theo đóng góp).
Mỗi dòng 1 đến 2 câu: việc đã giao nộp và trạng thái còn chờ (nếu có). Chi tiết kỹ thuật để trong PR hoặc commit, cột cuối là link tới đó.

| Ngày | Thành viên | Vai | Việc đã làm | Link PR / Issue / commit |
|---|---|---|---|---|
| 2026-11-14 | Trung | R4 Classical ML | Module chia CV dùng chung (`evaluation/splits.py`) và ghi/kiểm tra metric 13 cột (`evaluation/metrics.py`); R5 dùng lại nguyên bản. | `2f1f7f6`, `0c83883` (nhánh `Trung`) |
| 2026-11-18 | Bảo | R3 ERP | Notebook 02 đến 06: ERP và SEM, difference wave, topomap F2, ứng viên P300, bảng hành vi nhóm 39/35; kiểm tra cohort v2, 8 test. | [#5](https://github.com/tbmdemi/EEG_26_27/pull/5) |
| 2026-11-19 | Bảo | R3 ERP | Chạy lại notebook 02 đến 06 (26/26 ô đạt), nháp Results ERP và bàn giao. Quy tắc P300 và F3 còn chờ. | [#5](https://github.com/tbmdemi/EEG_26_27/pull/5) |
| 2026-11-19 | Long | R2 Preprocessing | Pipeline R2 dạng module, audit sự kiện và FIF, QC tóm tắt 35 subject, candidate sửa sub-10/sub-13, Methods và bảng subject/epoch, 11 test. Chưa chốt có dùng candidate không. | [#9](https://github.com/tbmdemi/EEG_26_27/pull/9), [Issue #6](https://github.com/tbmdemi/EEG_26_27/issues/6) |
| 2026-11-19 | Giang | R5 Deep Learning | EEGNet và ShallowConvNet, runner có resume, file fold cố định, sắp kênh theo tên, chia within thành 5 đoạn đều nhau, 12 test, nháp Methods DL. | [#7](https://github.com/tbmdemi/EEG_26_27/pull/7) |
| 2026-11-20 | Giang | R5 Deep Learning | Chạy đủ 8 tổ hợp DL trên 35 subject (4.6 giờ GPU), bảng kết quả và so sánh EEGNet với Shallow; EEGNet là DL tốt nhất; chia sẻ checkpoint qua Drive. | [#7](https://github.com/tbmdemi/EEG_26_27/pull/7) |
| 2026-11-20 | Long | R2 Preprocessing | Cập nhật hướng dẫn audit/candidate và ghép main mới; dữ liệu candidate để riêng khỏi v2. | [#9](https://github.com/tbmdemi/EEG_26_27/pull/9) |
| 2026-11-20 | Trung | R4 Classical ML | `lda_tw`, `xdawn_rg`, hàm Haufe pattern cho R1, runner căn kênh theo tên; chạy 24 tổ hợp trên 35 subject; thống kê RQ2 và hình F4; 6 test; báo cáo bàn giao và nháp Methods/Results. | [#10](https://github.com/tbmdemi/EEG_26_27/pull/10) |
| 2026-11-20 | Minh | R1 Explainability | Giải thích EEGNet: spatial pattern, độ quan trọng filter, Integrated Gradients, bảng vùng P300 và vùng vận động; hình F5a/F5b; 5 test; nháp RQ3; ghi quyết định vào changelog protocol. | [#8](https://github.com/tbmdemi/EEG_26_27/pull/8) |
| 2026-11-20 | Minh | R1 Explainability | LDA patterns (fit lại khớp đúng kết quả R4), hình F5c; hình F1 sơ đồ pipeline. | `ac98248` (nhánh `tbm`) |
| 2026-11-21 | Minh | R1 Explainability | Đối chứng nhiễu vận động: train EEGNet cho `dis_std`, `tgt_dis` (Kaggle T4), IG cho 3 task. Kết luận: P300 là nguồn chính, có thêm phần nhỏ từ chuẩn bị vận động. | `ac98248` (nhánh `tbm`) |

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
