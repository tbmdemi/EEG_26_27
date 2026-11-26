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

## Lưu output khi chạy

Theo thông báo cập nhật của nhóm trưởng, dùng code và dữ liệu v2 hiện hành; bước cleaning đã được thực hiện cho cả nhóm. Thành viên R3 tải dữ liệu bằng `python scripts/00_get_data.py`, rồi thực hiện phân tích ERP.

Commit code `.py` và notebook `.ipynb`; chạy `nbstripout --install` một lần trên mỗi máy để tự động loại output nhúng trong notebook khi commit. Các file sinh ra khi chạy lưu trong `figures/`, `results/`, `outputs/`, `notebooks/outputs/` hoặc `scripts/outputs/` và được gitignore. Riêng `results/preproc_log.csv` là log bàn giao dữ liệu được giữ trong Git. Chia sẻ hình và bảng kết quả qua thư mục chung của nhóm.

## R3 — Chạy lại và trạng thái bàn giao

ERP/P300/hành vi và F3 đã triển khai, kiểm chứng trên release v2; F2 còn chờ duyệt hình cuối và gói output mới chuẩn bị local, chưa xác nhận upload Drive. [Bàn giao R3](reports/r3_review_and_handoff.md) tập trung phương pháp, lệnh tái lập, validation, bằng chứng, ZIP/SHA256 và các phụ thuộc; xem riêng [mục F3](reports/r3_review_and_handoff.md#f3) cho tổng hợp ML/DL.

Báo cáo khoa học nằm trong [LaTeX](reports/latex/README.md). [Đường dẫn ERP draft cũ](reports/results_erp_draft.md) được giữ để tương thích; nhật ký chung ở `reports/contribution_log.md`.

## R4 — Classical ML & RQ2: chạy và bàn giao

Cần dữ liệu v2 (`python scripts/00_get_data.py`). Đã triển khai `lda_tw` (50 ms bins + shrinkage LDA, kèm `extract_lda_haufe_patterns` cho R1) và `xdawn_rg` (xDAWN rank-safe + Riemannian Tangent Space + balanced Logistic Regression), căn chỉnh thứ tự 127 kênh theo tên điện cực (`sub-13`, `sub-42`), đồng bộ chia CV `equal_segments` và `results/cv_cross_subject_folds.csv` cùng R5, chạy đủ 24 tổ hợp trên 35 subject và thực hiện kiểm định RQ2 (Wilcoxon + Holm) + Hình F4 đối chiếu với DL:

```bash
python scripts/02_run_classical_ml.py                # 2 models × 2 settings × 2 windows × 3 tasks (24 CSVs)
python scripts/03_stats_rq2.py --metric bal_acc      # thống kê RQ2 + hình F4 (sau khi có CSV của R5)
python scripts/03_stats_rq2.py --metric auc --out-csv results/rq2_wilcoxon_holm_auc.csv --fig-stem figures/F4_paired_classical_vs_dl_auc
```

- Kiểm tra code R4: `python -m unittest tests.test_r4_ml -v`, `python -m ruff check src/eegproj/evaluation/splits.py src/eegproj/evaluation/stats.py src/eegproj/evaluation/__init__.py src/eegproj/features src/eegproj/models/classical.py src/eegproj/models/runner.py src/eegproj/models/__init__.py scripts/02_run_classical_ml.py scripts/03_stats_rq2.py tests/test_r4_ml.py`.

Báo cáo bàn giao: [`reports/r4_evaluation_review.md`](reports/r4_evaluation_review.md); bản nháp report: [`reports/methods_ml_rq2_draft.md`](reports/methods_ml_rq2_draft.md).

## R5 — Deep Learning: chạy và bàn giao

Cần dữ liệu v2 (`python scripts/00_get_data.py`). Huấn luyện EEGNet và ShallowConvNet trên **đúng** phân chia CV của R4, ghi CSV theo protocol §7:

```bash
python scripts/04_train_dl.py                                   # eegnet + shallow × within + cross × full + early, task tgt_std
python scripts/04_train_dl.py --models eegnet --settings cross   # chạy từng phần
python scripts/04_train_dl.py --tasks dis_std tgt_dis            # task đối chứng (nếu còn thời gian)
python scripts/04_train_dl.py --subjects 01 03 --settings within # chạy thử nhanh → results/dev/ (không dùng cho report)
```

- Tự dùng GPU nếu có (`--device cuda|cpu`). Ước tính: CPU 4–8 nhân khoảng 4–9 giờ cho cả lưới chính; GPU (ví dụ Colab T4) dưới 1 giờ. Bị ngắt giữa chừng thì chạy lại đúng lệnh: kết quả đã xong được bỏ qua, phần dở dang chạy tiếp từ fold cuối.
- Đầu ra (gitignored): `results/metrics/{eegnet,shallow}_*.csv`, `results/cv_cross_subject_folds.csv` (phân chia subject cố định dùng chung với R4), `results/predictions/*.csv.gz` (điểm từng trial), `results/dl_runs/*.json` (hyperparameter, số epoch, thời gian từng fold), `results/checkpoints/` (mô hình cho R1, đọc bằng `eegproj.models.dl.load_checkpoint`).
- Không cần train lại: `python scripts/04_get_dl_results.py` tải bản outputs đã chạy (285 MB, Google Drive) vào `results/`, kiểm tra sha256, không ghi đè file bạn tự train (trừ khi có `--force`).
- Kiểm tra code R5: `python -m unittest tests.test_r5_dl -v`, `python -m ruff check src/eegproj/models src/eegproj/evaluation/folds.py scripts/04_train_dl.py tests/test_r5_dl.py`.

Phương pháp: [`reports/methods_dl_draft.md`](reports/methods_dl_draft.md).

## R1: Explainability (RQ3), chạy và bàn giao

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
