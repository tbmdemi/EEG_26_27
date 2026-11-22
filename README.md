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
- **PM / R2:** v2 đang dùng là bản phát hành sẵn có. Quy trình R2 hiện tại dùng `python scripts/02_r2.py --process --download` để tạo candidate riêng; xem bàn giao trước khi thay phiên bản dữ liệu chung. `scripts/01_preprocess.py` là entry point của triển khai v2 lịch sử, chưa nhận các sửa đổi candidate mới.

Notebook Colab tiền xử lý dữ liệu: [`preprocess_oddball_colab.ipynb`](https://colab.research.google.com/drive/1nqyGmEw7IiSgC1svfK4Oud3w8JP8NLi3?usp=sharing).

R2 review/candidate workflow: [`reports/r2_qc_and_handoff.md`](reports/r2_qc_and_handoff.md). Notebook 01 dùng script R2 mới; `python scripts/02_r2.py` mặc định chỉ audit, `--process --download` tạo candidate riêng.

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
│   └── explain/               # giải thích mô hình: dl.py (IG, spatial pattern, bảng ROI), plots.py (hình F5)
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

Cần dữ liệu v2 và checkpoint/predictions của R5 trong `results/`. Giải thích EEGNet theo protocol mục 9.2-9.3: spatial pattern (Haufe 2014) của từng spatial filter cùng mức quan trọng (AUC giảm bao nhiêu khi tắt filter), và Integrated Gradients (theo kênh và thời gian) trên các trial target được đoán đúng ở fold test:

```bash
python scripts/05_explain_dl.py                    # cross + within, full + early, tgt_std (~5 phút trên GPU)
python scripts/05_explain_dl.py --settings cross   # chỉ mô hình cross-subject
```

- Trước khi giải thích, script tính lại quyết định của từng checkpoint trên đúng trial test và so với `results/predictions/`; lệch quá 0.01 thì dừng (sai dữ liệu/thứ tự kênh). Sai số completeness của IG được ghi vào `results/explain/manifest.json`.
- Đầu ra (gitignored): `results/explain/` (`*_saliency.npz`, `*_patterns.csv`, `saliency_roi_tgt_std.csv`, `saliency_summary_tgt_std.csv`), hình `figures/F5a_eegnet_spatial_patterns`, `figures/F5b_eegnet_saliency_{cross,within}` (PNG 300 dpi + PDF).
- LDA patterns (mục 9.4): `python scripts/06_explain_lda.py` fit lại `lda_tw` của R4 trên đúng 5 fold cross-subject, đổi trọng số thành pattern (Haufe 2014). Script in bal_acc/AUC trung bình để so với CSV của R4 (phải khớp). Đầu ra: `results/explain/lda_*`, hình `figures/F5c_lda_patterns`. Chỉ cần CPU, khoảng 5 phút.
- Hình F1 (sơ đồ pipeline): `python scripts/07_plot_pipeline.py`. Chữ trong hình lấy từ protocol, đổi protocol thì sửa script.
- Kiểm tra code R1: `python -m unittest tests.test_r1_explain -v`, `python -m ruff check src/eegproj/explain scripts/05_explain_dl.py scripts/06_explain_lda.py scripts/07_plot_pipeline.py tests/test_r1_explain.py`.
- Task đối chứng (nhiễu vận động): train `python scripts/04_train_dl.py --models eegnet --tasks dis_std tgt_dis` (khoảng 80 phút trên GPU T4), rồi `python scripts/05_explain_dl.py --task dis_std` và `--task tgt_dis`. Hình và manifest của task đối chứng có hậu tố `_<task>`, không ghi đè hình của task chính.

Phương pháp và kết quả: [`reports/explain_dl_draft.md`](reports/explain_dl_draft.md).

## Liên hệ nộp bài

Report (tiếng Anh) gửi **ít nhất 1 tuần trước buổi thuyết trình** tới: guy.nagels@vub.be, thầy Trần Quốc Sang, thầy Nguyễn Tiến Dũng.
