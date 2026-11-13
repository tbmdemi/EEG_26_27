# 01 — Project brief

Tóm tắt từ slide "More information about the mini project" (bài giảng *Neuroimaging: EEG*, Guy Nagels).

## Nội dung

- Xử lý dữ liệu EEG
- Hiểu chức năng não qua việc ánh xạ hoạt động
- Bài toán phân loại: **hai loại kích thích** (frequent vs target)
- So sánh **ML truyền thống trên phân tích evoked response** với **Deep Learning**

## Công cụ được gợi ý

- Python
- MNE-Python — https://mne.tools/stable/
- osl-ephys — https://osl-ephys.readthedocs.io/en/latest/
- osl-dynamics — https://osl-dynamics.readthedocs.io/en/latest/

## Câu hỏi nghiên cứu

- **RQ1.** Kết quả phân loại single-trial tốt nhất cho frequent vs target?
- **RQ2.** Deep Learning có tốt hơn kỹ thuật ML tiêu chuẩn cho single-trial classification?
- **RQ3.** Đặc trưng mà mô hình DL khám phá có giải thích được không?

## Deliverables

### Report (tiếng Anh)
- Mô tả **đóng góp của từng thành viên** (ví dụ vai trò: Signal Preprocessing, Feature Extraction, Machine Learning, Visualization, Documentation, Project Manager).
- Ghi **phần trăm đóng góp** của từng người; **tổng = 100%**.
- Gửi **ít nhất 1 tuần trước** buổi thuyết trình tới:
  - guy.nagels@vub.be
  - Mr Tran Quoc Sang
  - Mr Nguyen Tien Dung

### Presentation (tiếng Anh)
- **Tối đa 10 phút.**
- Tập trung vào **phát hiện chính**: chúng là gì và giải thích chúng thế nào.
- **Tất cả thành viên phải có mặt; tất cả đều bị hỏi** (TA có thể dịch câu hỏi sang tiếng Việt).

## Chấm điểm

- Nhóm nhận một điểm khoa học chung làm cơ sở.
- Điểm cuối của từng người **được điều chỉnh theo đóng góp cá nhân và mức độ hiểu bài thể hiện ra**.
- Tham gia project là điều kiện để qua môn.
- Ai **không chứng minh được đóng góp thực chất** có thể trượt phần này, bất kể bài nhóm tốt đến đâu.

👉 Hệ quả cho nhóm: mỗi người phải (1) có commit/PR mang tên mình, (2) ghi `reports/contribution_log.md` đều đặn, (3) hiểu và trả lời được câu hỏi về **toàn bộ** pipeline, không chỉ phần của mình.