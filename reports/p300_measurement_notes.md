# R3 — Đề xuất đo ứng viên đỉnh P300

**Trạng thái:** thăm dò, để nhóm review; chưa chốt phương pháp cho report. Dữ liệu tiền xử lý và các tham số phân loại được giữ nguyên.

## Cơ sở và cửa sổ đề xuất

Bài báo của chính dataset mô tả P3 trong khoảng 300–600 ms và đáp ứng target nổi bật ở vùng đỉnh, gồm Pz ([Dzianok et al., 2022, mục Dataset ERP results](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac015/6543635)). R3 dùng khoảng **300–600 ms** làm cửa sổ thăm dò cố định, **Pz** là điện cực chính; CPz/Cz là các phép đo bổ sung. Đây là đề xuất triển khai để nhóm review sau khi xem ERP và topomap, không phải phép đo đã đăng ký trước.

Thông số nằm ở `erp.p300` trong `configs/config.yaml`. Cửa sổ này được khai báo riêng với cửa sổ hiển thị topomap 250–600 ms. Không tự đổi cửa sổ cho từng người để lấy một đỉnh đẹp hơn.

## Định nghĩa phép đo

- Lấy ERP đã trung bình trial trong từng subject, riêng cho `target` và `distractor`.
- Trong cửa sổ cố định, lấy mẫu có **điện thế lớn nhất có dấu**; không lấy trị tuyệt đối. Biên độ ứng viên tính bằng µV, latency tính bằng ms sau kích thích. Nếu nhiều mẫu bằng cực đại, lấy mẫu xuất hiện đầu tiên và gắn cờ.
- Ghi thêm biên độ trung bình toàn cửa sổ, số trial và trạng thái QC từ log.
- Không lọc thêm, nội suy latency hay chọn đỉnh từ grand-average để thay cho đỉnh từng subject. Với 250 Hz, độ phân giải mẫu của latency là 4 ms.
- Không tự gán mọi ứng viên thành P300, P3a hoặc P3b; cần xem hình và duyệt phương pháp.

## Cờ kiểm tra

Gắn cờ nếu cực đại không dương, nằm ngay biên, cách biên cửa sổ không quá 20 ms, không phải cực đại cục bộ trong waveform đầy đủ, hoặc có nhiều mẫu cùng cực đại. Khoảng 20 ms là quy tắc sàng lọc thăm dò, để nhóm review.

`peak_needs_review` chỉ mô tả các cờ của phép đo đỉnh. `visual_review` là trạng thái duyệt chất lượng tín hiệu của R2. Không có cờ đỉnh không đồng nghĩa dữ liệu đã được R2 duyệt.

Giữ mọi subject hợp lệ trong bảng và hình, kể cả ứng viên có cờ; không tự loại thêm người. Bảng tổng hợp mô tả toàn bộ ứng viên và ghi rõ số có cờ. So sánh target/distractor phải ghép cặp theo subject; ba điện cực của cùng một người không phải ba quan sát độc lập.

## Đầu ra

Notebook `05_erp_p300_candidates.ipynb` tạo bảng ứng viên theo subject × điều kiện × kênh, bảng tổng hợp mô tả, danh sách cần xem lại và manifest trong `results/erp/`. Hình phân bố tại Pz và lưới ERP từng người nằm trong `figures/`. Các output được gitignore.

Trước khi dùng số liệu trong report: nhóm review cửa sổ/thuật toán, xem các ứng viên có cờ và phối hợp R2 duyệt QC. Nếu đổi phương pháp, cập nhật config, ghi lại quyết định và chạy lại toàn bộ subject bằng cùng một quy tắc.
