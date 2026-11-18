# Single-trial EEG Classification: Frequent vs Target (Visual Oddball)

Mini project môn học — Neuroimaging: EEG (Guy Nagels).
Dữ liệu: **Nencki-Symfonia EEG/ERP dataset** (GigaDB 10.5524/100990, OpenNeuro `ds004621`), task **3-stimulus visual oddball**, **toàn bộ subject dùng được** (42 người trừ sub-02/14/17), **chỉ các trial có phản hồi hành vi đúng**.

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
├── results/                   # preproc_log.csv, metrics/*.csv (commit được, nhỏ) ; checkpoints/ (gitignored)
├── figures/                   # hình cho report/slide (PNG/PDF)
├── reports/                   # report_template.md, contribution_log.md, slide
└── scripts/                   # script chạy end-to-end từng bước
```

## Thành viên

| Họ tên | MSSV | Vai trò chính | GitHub |
|---|---|---|---|
| Trịnh Bình Minh | 24022413 | Project Manager + Explainability | [@tbmdemi](https://github.com/tbmdemi)|
| | | Signal Preprocessing | |
| | | ERP & Visualization | |
| | | Classical ML | |
| | | Deep Learning | |

## Liên hệ nộp bài

Report (tiếng Anh) gửi **ít nhất 1 tuần trước buổi thuyết trình** tới: guy.nagels@vub.be, thầy Trần Quốc Sang, thầy Nguyễn Tiến Dũng.
