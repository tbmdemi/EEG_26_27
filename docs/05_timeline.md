# 05 — Timeline

`D` = ngày thuyết trình (chờ TA xác nhận). **Hạn gửi report = D − 7.** Nhóm đặt hạn nội bộ sớm hơn để có thời gian sửa.

| Mốc | Thời điểm | Việc phải xong | Owner |
|---|---|---|---|
| M0 | Tuần 1 | Repo, môi trường, mọi người tải được dữ liệu; Issues/board; chốt protocol | R1, tất cả |
| M1 | Tuần 1–2 | Pipeline tiền xử lý chạy trên 3 subject; module CV + metric dùng chung | R2, R4 |
| M2 | Tuần 2 | Tiền xử lý toàn bộ subject; ERP grand-average | R2, R3 |
| M3 | Tuần 3 | Đủ kết quả classical + DL cho task chính (`tgt_std`, 2 setting × 2 window) | R4, R5 |
| M4 | Tuần 3–4 | Task đối chứng, thống kê RQ2, explainability | R1, R4, R5 |
| M5 | D − 10 | **Freeze kết quả** (không chạy lại nữa); bản nháp report đầy đủ | Tất cả |
| M6 | D − 8 | Review chéo report, chốt % đóng góp | R1 |
| M7 | **D − 7** | **Gửi report** | R1 |
| M8 | D − 7 → D − 1 | Slide + tập dượt ≥ 2 lần, bấm giờ; mỗi người tự luyện trả lời câu hỏi (xem dưới) | Tất cả |
| M9 | D | Thuyết trình | Tất cả |

Khi biết ngày D, PM điền ngày cụ thể vào cột *Thời điểm* và tạo Milestone tương ứng trên GitHub.

## Câu hỏi mỗi người phải trả lời được (tự luyện)

1. P300 là gì, xuất hiện ở đâu/khi nào, vì sao target gây ra P300 lớn hơn standard?
2. Vì sao dùng balanced accuracy thay vì accuracy?
3. Vì sao chia CV theo khối thời gian / theo subject? Data leakage là gì, nhóm tránh thế nào?
4. Nhiễu vận động (motor confound) là gì và nhóm kiểm soát ra sao?
5. Vì sao DL thắng/thua ML truyền thống trong kết quả của nhóm?
6. Mô hình DL dựa vào kênh nào, thời điểm nào? Điều đó có hợp lý về sinh lý không?
7. Phần mình làm: đã quyết định gì, vì sao, nếu làm lại sẽ đổi gì?
