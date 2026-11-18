# Contribution log

Ghi **ngay sau khi xong mỗi việc**. Đây là căn cứ để chốt % đóng góp trong report (điểm cá nhân được điều chỉnh theo đóng góp).

| Ngày | Thành viên | Vai | Việc đã làm | Link PR / Issue / commit |
|---|---|---|---|---|
| 2026-11-18 | Bảo | R3 — ERP & Visualization | Cập nhật tên/vai trò trong README và hướng dẫn bỏ qua output notebook/script theo thông báo nhóm trưởng; chưa chạy phân tích ERP | Thay đổi local, chưa tạo PR |
| 2026-11-18 | Bảo | R3 — ERP & Visualization | Tạo notebook `02_erp_exploration.ipynb`; xác nhận đủ file epoch của 35 subject hợp lệ; chạy các ô code kiểm tra sub-01 và vẽ ERP tại Pz (478 standard, 61 target, 52 distractor); lưu PNG/PDF local, notebook nguồn không lưu output | Thay đổi local, chưa tạo PR |
| 2026-11-18 | Bảo | R3 — ERP & Visualization | Kiểm tra lần Run All đã lưu của notebook 02: đủ 4 ô code, không có lỗi. Tạo module `evaluation/erp.py` và notebook `03_erp_grand_average.ipynb`; chạy trên đủ 35 subject, xuất ERP Pz/CPz/Cz và difference wave với SEM qua subject, bảng CSV local; kiểm tra công thức SEM và ruff. Topomap, đo P300 và duyệt QC vẫn chưa hoàn tất | Thay đổi local, chưa tạo PR |
| 2026-11-18 | Bảo | R3 — ERP & Visualization | Tạo `evaluation/topography.py` và notebook `04_erp_topomaps.ipynb`; chạy đủ 35 subject/127 kênh, ghép theo tên điện cực (sub-13/sub-42 khác thứ tự), xuất F2 ERP + topomap 250–600 ms và lát cắt 300/400/500/600 ms. Dùng vị trí điện cực trung bình trong hệ tọa độ đầu; lưu bảng/provenance local, đối chiếu khớp notebook 03, kiểm tra SEM và ruff. Bổ sung riêng cấu hình hiển thị ERP để review, giữ nguyên tham số tiền xử lý/phân loại. Đo P300 và duyệt QC chưa hoàn tất | Thay đổi local, chưa tạo PR |
| 2026-11-18 | Bảo | R3 — ERP & Visualization | Xác nhận notebook 04 đã chạy đủ 6 ô, không có lỗi. Tạo `evaluation/p300.py`, notebook `05_erp_p300_candidates.ipynb` và ghi chú phương pháp; chạy đo thăm dò 300–600 ms, xuất 210 ứng viên và hình phân bố/lưới ERP. Tại Pz có 5 target và 1 distractor cần xem lại theo quy tắc đỉnh; giữ mọi subject. Kiểm tra số đo/đơn vị, đỉnh âm, đỉnh sát biên, đỉnh bằng nhau, ruff; cửa sổ/quy tắc đo vẫn là đề xuất để nhóm review và QC của R2 còn pending | Thay đổi local, chưa tạo PR |
| 2026-11-18 | Bảo | R3 — ERP & Visualization | Tạo `evaluation/behaviour_summary.py`, notebook `06_erp_behaviour_summary.ipynb` và ghi chú công thức; tổng hợp hành vi trước loại epoch cho 39 người và nhóm ERP 35, đối chiếu mốc mean/SD đã kiểm tra ở Table 5 của bài báo. Xuất bảng/hình/provenance local, kiểm tra mẫu số thực tế, trọng số bằng nhau qua subject và SD của RT qua người; ruff đạt. Giữ nguyên phần lọc hành vi/EEG | Thay đổi local, chưa tạo PR |

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
