# Vai trò và các bước công việc của tôi — Tiền xử lý EEG

## Vai trò trong dự án

Phần công việc đang thực hiện tương ứng **R2 — Signal Preprocessing** trong
`docs/04_roles_and_tasks.md`. Đây là mô tả phần việc cá nhân đang chuẩn bị,
không tự thay đổi phân công chính thức hoặc phần trăm đóng góp của nhóm.

Trách nhiệm: kiểm tra dữ liệu, làm sạch tín hiệu cho sub-01 đến sub-05,
kiểm tra chất lượng và bàn giao epoch cùng log cho người làm ERP, ML và DL.
Notebook thực hiện: `notebooks/01_preproc_five_subjects.ipynb`.

## Phạm vi và các thay đổi cấu hình riêng cho lần chạy

- Chỉ dùng 5 subject theo yêu cầu, không tải toàn bộ 42 subject.
- Đọc số kênh thực tế là 127; không tạo thêm kênh giả để đủ 128.
- Bổ sung cấu hình ngưỡng kênh hỏng, thông số hội tụ/decimation ICA và QC.
- Các thông số đã chốt về band-pass, epoch, baseline, ICLabel và ngưỡng
  loại epoch vẫn theo protocol.
- Tài liệu nhóm được giữ nguyên. Các quyết định triển khai và hạn chế
  của lần chạy này tập trung trong file cá nhân này để nhóm xem xét.

## Nguồn yêu cầu

- `docs/01_project_brief.md`: mục tiêu bài tập, so sánh ML/DL và giải thích.
- `docs/02_dataset.md`: dataset, task oddball, mã sự kiện.
- `docs/03_analysis_protocol.md`, mục 1: các bước tiền xử lý bắt buộc.
- `configs/config.yaml`: tham số chạy thực tế.
- `docs/04_roles_and_tasks.md`, R2: các đầu ra phải bàn giao.

## Chạy

Mở `notebooks/01_preproc_five_subjects.ipynb`, chọn kernel của môi trường
đã cài `requirements.txt`, rồi chạy các ô từ trên xuống.
Ô thực hiện dùng `run(ROOT, subjects=CONFIG['dataset']['subjects'], overwrite=False)`.
Để chạy thử một người, truyền `subjects=['01']`; để chạy lại, đặt `overwrite=True`.
Code xử lý nằm trong `src/eegproj/preprocessing/`; notebook chỉ gọi hàm và hiển thị QC.
Chạy không cần Jupyter: `python scripts/01_preprocess.py` (thêm `-s 01` để chạy một người, `--overwrite` để chạy lại).

Đầu vào: `data/raw/ds004621`. Bổ sung từ nguồn OpenNeuro của cùng
dataset: `task-oddball_eeg.json` và 5 tệp
`sourcedata/sub-XX/coords/sub-XX_captrak.bvct`.
Không tải thêm bản ghi EEG hay subject ngoài tập con được yêu cầu.

## Xử lý và quyết định triển khai

1. Đọc bằng MNE-BIDS, chỉ giữ EEG thực có (127 kênh).
2. Dùng tọa độ CapTrak riêng từng subject; đổi nhãn tai TP9/TP10 để khớp.
3. Tạo lại annotations từ cột `event_type` của events.tsv. Cột `trial_type`
   chỉ có stimulus/response, không đủ để phân biệt ba kích thích.
4. Phát hiện kênh phẳng hoặc độ lệch chuẩn bất thường qua robust z của log(std).
   Ngưỡng nằm trong config; xuất bảng điểm và hình để kiểm tra bằng mắt.
   Có thể thêm kênh xác nhận thủ công trong `bad_channels.manual` rồi chạy lại.
5. FIR zero-phase 0.1–40 Hz; nội suy spline; average reference.
   Không notch 50 Hz mặc định vì đã nằm ngoài dải low-pass 40 Hz.
6. Extended Infomax trên bản sao high-pass 1 Hz (1–40 Hz), 99% phương sai.
   Fit trên toàn recording không dùng nhãn; lấy mỗi mẫu thứ 4 sau lọc để
   giảm chi phí tính toán. Dừng với lỗi nếu chạm giới hạn số vòng lặp.
   Infomax dùng tối đa 512 vòng, ngưỡng thay đổi trọng số 1e-7 trong config;
   kiểm tra hội tụ từ log bước lặp thực tế. MNE Infomax có thể trả `n_iter_`
   bằng `max_iter` ngay khi hội tụ, nên không dùng riêng trường đó để kết luận.
7. ICLabel backend ONNX; loại thành phần có xác suất eye blink hoặc muscle
   artifact >=0.8. Áp dụng ICA lên dữ liệu 0.1–40 Hz.
8. Resample 250 Hz, đồng thời chuyển vị trí events; epoch -0.2 đến 0.8 s,
   baseline -0.2 đến 0 s; loại peak-to-peak >150 uV.
9. Giữ cả standard/target/distractor; metadata giữ onset gốc và block xác
   định từ các lần nghỉ S10. Subject giữ dưới 50% target được gắn cờ loại.

ICLabel được huấn luyện cho 1–100 Hz; chạy 1–40 Hz là lựa chọn theo protocol
hiện tại và cần ghi hạn chế này trong report.

## Đầu ra và nghiệm thu

- `data/derivatives/preproc/sub-XX_task-oddball_epo.fif`: epoch sau xử lý.
- Cùng thư mục: ICA để tái kiểm tra, summary JSON, provenance của lần chạy.
- `results/preproc_log.csv`: số epoch trước/sau, tỷ lệ giữ từng lớp,
  kênh nội suy, IC loại, subject loại và lý do.
- `figures/preprocessing/`: PSD, ví dụ tín hiệu trước/sau, bảng kênh,
  xác suất ICLabel và hình các IC bị loại.

Không coi một lần chạy là hoàn tất nghiệm thu nếu chưa kiểm tra hình QC.
`visual_review=pending` thể hiện trạng thái cần kiểm tra, không phải chứng
nhận dữ liệu đã sạch. Không dùng subject được gắn `excluded_subject=True`
cho phân loại chính.

Phạm vi chỉ gồm tiền xử lý. Thiết lập cross-subject 6 fold trong config
chưa được đổi; phải dùng tối đa 5 fold khi triển khai phân loại tập con.

## Kết quả đã thực hiện ngày 2026-11-13

| Subject | Standard còn lại / 498 | Target còn lại / 81 | Distractor còn lại / 81 | IC loại |
|---|---:|---:|---:|---:|
| sub-01 | 491 | 80 | 80 | 8 |
| sub-02 | 474 | 77 | 76 | 6 |
| sub-03 | 471 | 78 | 80 | 8 |
| sub-04 | 496 | 80 | 81 | 2 |
| sub-05 | 480 | 75 | 76 | 10 |

Tổng giữ **3.195/3.300 epoch (96,82%)**. Không có subject bị loại.
Không kênh nào vượt ngưỡng phát hiện tự động, nên không có nội suy trong
lần chạy này; nhánh nội suy vẫn có trong notebook cho trường hợp phát hiện kênh hỏng.

Cả 5 tệp FIF đã được đọc lại và vượt kiểm tra tự động: 127 kênh, 250 Hz,
251 mẫu/epoch, baseline, biên độ peak-to-peak, ba lớp kích thích, thứ tự
thời gian và đủ 5 block. Notebook đã chạy 11 ô code từ đầu đến cuối bằng
kernel Jupyter thật, theo chế độ tiếp tục từ các kết quả xử lý đã lưu,
không có lỗi. Code được giữ trong notebook; không còn script `.py` riêng.

Codex đã xem PSD, tín hiệu mẫu trước/sau, phân bố độ lệch chuẩn theo kênh,
topomap IC bị loại, bản đồ peak-to-peak của các epoch giữ lại và ERP Pz
của cả 5 người. Log ghi `summary_plots_reviewed`; đây là kiểm tra hình
tổng hợp, chưa phải duyệt tương tác mọi kênh trên toàn recording.
Sub-02 có nhiễu lớn ở đoạn đầu trước task; các epoch giữ lại vẫn vượt
kiểm tra biên độ. Không diễn giải toàn bộ continuous recording là sạch hoàn toàn.

Các kết quả này phục vụ bước tiền xử lý và bàn giao; chưa phải kết quả
huấn luyện ML/DL hoặc kiểm định thống kê của dự án.
