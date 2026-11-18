# Single-trial EEG Classification: Frequent vs Target (Visual Oddball)

Mini project môn học — Neuroimaging: EEG (Guy Nagels).
Dữ liệu: **Nencki-Symfonia EEG/ERP dataset** (GigaDB 10.5524/100990, OpenNeuro `ds004621`), task **3-stimulus visual oddball**, **chỉ các trial có phản hồi hành vi đúng**. Bản v2 hiện hành có **35 subject ERP hợp lệ từ 42 người**: loại trước sub-02/14/17, rồi loại thêm sub-08/23/35/41 theo tiêu chí pipeline; danh sách nằm trong `results/preproc_log.csv`.

## Câu hỏi nghiên cứu

| ID | Câu hỏi |
|---|---|
| RQ1 | Kết quả phân loại single-trial tốt nhất cho *frequent* vs *target* là bao nhiêu? |
| RQ2 | Deep Learning có tốt hơn ML truyền thống (dựa trên evoked response) cho single-trial classification không? |
| RQ3 | Các đặc trưng mà mô hình DL học được có giải thích được không? |

## Đọc theo thứ tự

1. [`docs/01_project_brief.md`](docs/01_project_brief.md) — đề bài, deliverables, cách chấm điểm
2. [`docs/02_dataset.md`](docs/02_dataset.md) — mô tả dữ liệu và cách tải
3. [`docs/03_analysis_protocol.md`](docs/03_analysis_protocol.md) — **các quyết định phân tích đã chốt, mọi người PHẢI theo**
4. [`docs/04_roles_and_tasks.md`](docs/04_roles_and_tasks.md) — phân vai, checklist, đầu vào/đầu ra mỗi giai đoạn
5. [`docs/05_timeline.md`](docs/05_timeline.md) — mốc thời gian
6. [`CONTRIBUTING.md`](CONTRIBUTING.md) — quy trình Git, quy ước code, cách ghi nhận đóng góp

## Cài đặt nhanh

```bash
git clone <URL-repo>
cd <tên-repo>
# Dùng môi trường Python >= 3.10 của riêng bạn (venv, conda, ...), ví dụ:
#   python -m venv .venv && source .venv/bin/activate
#   hoặc: conda create -n eeg python=3.11 && conda activate eeg
pip install -r requirements.txt   # gồm cả `pip install -e .` cho package eegproj
nbstripout --install          # tự xóa output notebook khi commit
```

### Dữ liệu

- **Thành viên:** không cần dữ liệu thô. Chạy `python scripts/00_get_data.py`: script tải bản v2 đã tiền xử lý (`preproc_v2_20261118.zip`, 2.6 GB) từ Drive, kiểm tra rồi giải nén vào `data/derivatives/preproc/`; chỉ dùng subject có `excluded_subject = False` và `pipeline_version = 2` trong `results/preproc_log.csv` (hoặc `eegproj.preprocessing.usable_subjects`).
- **PM / R2:** tạo dữ liệu bằng `python scripts/01_preprocess.py --download --delete-raw`.

Chi tiết: [`docs/02_dataset.md`](docs/02_dataset.md). **Không bao giờ commit dữ liệu lên repo.**

## Cấu trúc thư mục

```
.
├── configs/config.yaml        # MỌI tham số dùng chung (filter, epoch, CV...) — nguồn sự thật duy nhất
├── data/                      # (gitignored) raw/ds004621/ và derivatives/preproc/*_epo.fif
├── docs/                      # tài liệu dự án
├── notebooks/                 # khám phá, trực quan hóa (đặt tên NN_<tên>_<mô-tả>.ipynb)
├── src/eegproj/
│   ├── preprocessing/         # lọc hành vi, lọc tín hiệu, ICA, epoching, QC
│   ├── features/              # trích xuất đặc trưng cho ML truyền thống
│   ├── models/                # ML truyền thống + Deep Learning
│   ├── evaluation/            # chia CV, metric, thống kê
│   └── explain/               # giải thích mô hình
├── results/                   # preproc_log.csv được giữ trong Git; kết quả chạy được gitignore
├── figures/                   # hình sinh ra cho report/slide (PNG/PDF, gitignored)
├── reports/                   # report_template.md, contribution_log.md, slide
└── scripts/                   # script chạy end-to-end từng bước
```

## Thành viên

| Họ tên | MSSV | Vai trò chính | GitHub |
|---|---|---|---|
| Trịnh Bình Minh | 24022413 | Project Manager + Explainability | [@tbmdemi](https://github.com/tbmdemi)|
| | | Signal Preprocessing | |
| Bảo | | ERP & Visualization (R3) | |
| | | Classical ML | |
| | | Deep Learning | |

## Lưu output khi chạy

Theo thông báo cập nhật của nhóm trưởng, dùng code và dữ liệu v2 hiện hành; bước cleaning đã được thực hiện cho cả nhóm. Thành viên R3 tải dữ liệu bằng `python scripts/00_get_data.py`, rồi thực hiện phân tích ERP.

Commit code `.py` và notebook `.ipynb`; chạy `nbstripout --install` một lần trên mỗi máy để tự động loại output nhúng trong notebook khi commit. Các file sinh ra khi chạy lưu trong `figures/`, `results/`, `outputs/`, `notebooks/outputs/` hoặc `scripts/outputs/` và được gitignore. Riêng `results/preproc_log.csv` là log bàn giao dữ liệu được giữ trong Git. Chia sẻ hình và bảng kết quả qua thư mục chung của nhóm.

## R3 — Chạy lại và trạng thái bàn giao

Chạy notebook `02_erp_exploration` → `03_erp_grand_average` → `04_erp_topomaps` → `05_erp_p300_candidates` → `06_erp_behaviour_summary` bằng kernel của môi trường đã cài requirements; Restart Kernel rồi Run All cho từng notebook. Bộ nạp R3 kiểm tra log v2 đủ subject cấu hình và cohort khớp log bàn giao trong Git; bản hiện tại phải có 35 người ERP.

Đã triển khai ERP/SEM qua subject, difference wave ghép cặp, topomap theo tên điện cực, đo ứng viên P300 thăm dò và bảng hành vi tách nhóm 39/35. **F2 và Results – ERP còn là bản nháp**; visual QC của R2 và quy tắc P300 vẫn chờ nhóm duyệt. F3 và `evaluation/aggregate.py` chưa triển khai; cần metrics của R4/R5 cho phần đó.

Kiểm tra code R3: `python -m ruff check src/eegproj/evaluation tests notebooks`, `python -m ruff format --check src/eegproj/evaluation tests notebooks`, `python -m unittest discover -s tests -v`.

Xem [bàn giao R3 và ghi chú phương pháp](reports/r3_review_and_handoff.md), [bản nháp Results – ERP](reports/results_erp_draft.md). Nhật ký đóng góp nằm ở `reports/contribution_log.md`.

## Liên hệ nộp bài

Report (tiếng Anh) gửi **ít nhất 1 tuần trước buổi thuyết trình** tới: guy.nagels@vub.be, thầy Trần Quốc Sang, thầy Nguyễn Tiến Dũng.
