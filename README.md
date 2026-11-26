# Single-trial EEG Classification: Frequent vs Target (Visual Oddball)

Mini project môn học: Neuroimaging EEG (Guy Nagels).
Dữ liệu: **Nencki-Symfonia EEG/ERP dataset** (GigaDB 10.5524/100990, OpenNeuro `ds004621`), task **3-stimulus visual oddball**, **chỉ các trial có phản hồi hành vi đúng**. Bản v2 hiện hành có **35 subject ERP hợp lệ từ 42 người**: loại trước sub-02/14/17, rồi loại thêm sub-08/23/35/41 theo tiêu chí pipeline; danh sách nằm trong `results/preproc_log.csv`.

## Câu hỏi nghiên cứu

| ID | Câu hỏi |
|---|---|
| RQ1 | Kết quả phân loại single-trial tốt nhất cho *frequent* vs *target* là bao nhiêu? |
| RQ2 | Deep Learning có tốt hơn ML truyền thống (dựa trên evoked response) cho single-trial classification không? |
| RQ3 | Các đặc trưng mà mô hình DL học được có giải thích được không? |

## Đọc theo thứ tự

1. [`docs/01_project_brief.md`](docs/01_project_brief.md): đề bài, deliverables, cách chấm điểm
2. [`docs/02_dataset.md`](docs/02_dataset.md): mô tả dữ liệu và cách tải
3. [`docs/03_analysis_protocol.md`](docs/03_analysis_protocol.md): **các quyết định phân tích đã chốt, mọi người PHẢI theo**
4. [`docs/04_roles_and_tasks.md`](docs/04_roles_and_tasks.md): phân vai, checklist, đầu vào/đầu ra mỗi giai đoạn
5. [`docs/05_timeline.md`](docs/05_timeline.md): mốc thời gian
6. [`CONTRIBUTING.md`](CONTRIBUTING.md): quy trình Git, quy ước code, cách ghi nhận đóng góp

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

- Thành viên tải đúng release `preproc_v2_20261118.zip` (35 subject, 20.600 epoch) bằng `python scripts/00_get_data.py`; không cần tải raw. Dùng subject v2 được đánh dấu hợp lệ trong `results/preproc_log.csv` hoặc `eegproj.preprocessing.usable_subjects`.
- R2 mặc định audit release; `--process --download` chỉ tạo candidate riêng, không thay release v2. Xem [R2 handoff](reports/r2_qc_and_handoff.md) trước khi xử lý candidate.
- Chi tiết dữ liệu ở [dataset guide](docs/02_dataset.md). Colab preprocessing là [tài liệu lịch sử](https://colab.research.google.com/drive/1nqyGmEw7IiSgC1svfK4Oud3w8JP8NLi3?usp=sharing), không thuộc luồng phân tích hiện tại.

## Cấu trúc

| Thư mục | Vai trò |
|---|---|
| `configs/` | Tham số dùng chung |
| `data/` | Dữ liệu cục bộ, không commit |
| `docs/` | Dataset, protocol, timeline và phân vai |
| `notebooks/` | Khám phá dữ liệu và phân tích ERP |
| `src/eegproj/` | Code tái sử dụng: preprocessing, features, models, evaluation, explain |
| `scripts/` | Lệnh chạy pipeline theo bước |
| `results/`, `figures/`, `outputs/` | Kết quả sinh ra, không commit |
| `reports/` | Bản nháp, bàn giao và report LaTeX |
| `tests/` | Test theo từng vai |

## Thành viên

| Họ tên | MSSV | Vai trò chính | GitHub |
|---|---|---|---|
| Trịnh Bình Minh | 24022413 | Project Manager + Explainability | [@tbmdemi](https://github.com/tbmdemi) |
| | | Signal Preprocessing | |
| Bảo | | ERP & Visualization (R3) | |
| Trần Đức Trung | 24022477 | Classical ML (R4) | [@TrungDev2431](https://github.com/TrungDev2431) |
| Trần Bùi Hà Giang | 24022314 | Deep Learning (R5) | |

## Chạy pipeline

| Vai | Lệnh chính |
|---|---|
| Tải dữ liệu v2 | `python scripts/00_get_data.py` |
| R2, audit release | `python scripts/02_r2.py` |
| R3, ERP | Chạy notebook `02` đến `06` theo thứ tự |
| R3, tổng hợp F3 | `python scripts/08_aggregate_results.py` |
| R4, classical ML | `python scripts/02_run_classical_ml.py` |
| R4, thống kê RQ2 | `python scripts/03_stats_rq2.py --metric bal_acc` |
| R5, deep learning | `python scripts/04_train_dl.py` hoặc `python scripts/04_get_dl_results.py` |
| R1, giải thích | `python scripts/05_explain_dl.py` và `python scripts/06_explain_lda.py` |
| Hình pipeline F1 | `python scripts/07_plot_pipeline.py` |

Xem [phân vai và checklist](docs/04_roles_and_tasks.md) để biết đầu vào, đầu ra và tình trạng từng vai. Tham số phân tích chỉ lấy từ [config.yaml](configs/config.yaml); quy tắc đã chốt nằm trong [analysis protocol](docs/03_analysis_protocol.md).

## Output

Không commit dữ liệu, checkpoint hay output sinh ra. Kết quả và hình được chia sẻ qua thư mục chung; ngoại lệ tracked là `results/preproc_log.csv`. Notebook source được commit sau khi `nbstripout` loại output nhúng.

R3 chi tiết: [bàn giao](reports/r3_review_and_handoff.md) và [report template](reports/report_template.md).
