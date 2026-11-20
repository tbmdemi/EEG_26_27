# 04 — Roles & tasks

Mỗi vai có **owner** chịu trách nhiệm chính, nhưng mọi người phải hiểu toàn bộ pipeline (ai cũng bị hỏi khi thuyết trình).
Nếu nhóm < 5 người: gộp *ERP & Visualization* vào *Preprocessing*, gộp *Explainability* vào *Deep Learning*.

Theo dõi tiến độ bằng **GitHub Issues** (mỗi checkbox bên dưới = 1 issue, gán đúng người, gắn label vai trò) và một **GitHub Project board** (To do / In progress / Review / Done).

---

## R1 — Project Manager + Explainability
**Owner:** Minh

- [X] Tạo Issues/Project board từ file này; bật branch protection cho `main`
- [ ] Review PR liên quan đến protocol/config
- [ ] Explainability: spatial filter EEGNet, Integrated Gradients, LDA patterns (protocol §9). Đã xong EEGNet spatial pattern và IG (2026-11-20, `scripts/05_explain_dl.py`, `reports/explain_dl_draft.md`); LDA patterns chờ `lda_tw` của R4
- [ ] Phân tích đối chứng nhiễu vận động (`early` vs `full`, task `dis_std`, `tgt_dis`)
- [ ] Hình F1, F5. F5 bản nháp: `figures/F5a_eegnet_spatial_patterns`, `figures/F5b_eegnet_saliency_{cross,within}`; F1 chưa làm
- [ ] Ghép report, kiểm tra tổng % đóng góp = 100%, gửi email đúng hạn

**Đầu ra:** `src/eegproj/explain/`, `figures/F1*`, `figures/F5*`, report hoàn chỉnh.

## R2 — Signal Preprocessing
**Owner:** ______

- [x] Pipeline tiền xử lý theo protocol §1, đọc tham số từ config (v1: sub-01…05, 2026-11-13)
- [x] Pipeline v2: lọc trial đúng hành vi, loại/gắn cờ subject, đánh dấu chỗ nối recording, chế độ `--download --delete-raw` (2026-11-17)
- [x] Chạy `python scripts/01_preprocess.py --download --delete-raw` cho **toàn bộ subject** (42 subject, 35 dùng được; phát hành `preproc_v2_20261118.zip`)
- [ ] Kiểm tra bằng mắt hình QC (`figures/preprocessing/`), ưu tiên subject có cờ trong cột `flags`; cập nhật `visual_review` / `bad_channels.manual` nếu cần rồi chạy lại subject đó
- [x] Upload `preproc_v2_20261118.zip` lên Drive `EEG_26_27/release/` (**không** qua Git), cập nhật `RELEASE` trong `scripts/00_get_data.py` (2026-11-18)
- [ ] Commit code v2 + `results/preproc_log.csv`, mở Issue **"Data v2 ready"** kèm commit hash
- [ ] Viết mục *Methods – Preprocessing* + bảng subject/epoch bị loại trong report

**Đầu ra:** `src/eegproj/preprocessing/`, `scripts/01_preprocess.py`, epoch `.fif`, log.
**Data v2 đã phát hành (2026-11-18).** Mọi người lấy dữ liệu bằng `python scripts/00_get_data.py` và chỉ làm việc trên v2; dữ liệu v1 đã bỏ.

## R3 — ERP & Visualization
**Owner:** Bảo

- [x] Grand-average ERP 3 loại kích thích tại Pz/CPz/Cz (toàn bộ subject dùng được, mean ± SEM qua subject); difference wave; topomap theo thời gian — đã tạo bản nháp trên 35 subject bằng notebook 03/04; chờ R2 duyệt QC trước khi chốt hình cho report
- [ ] Đo biên độ và latency đỉnh P300 (target, distractor) theo từng subject → bảng + phân bố — đã có bản thăm dò notebook 05 (210 ứng viên, 35 subject × 2 điều kiện × 3 kênh); chờ review cửa sổ/quy tắc đo và các trường hợp có cờ trước khi chốt
- [x] Bảng hành vi (hit rate, false alarm, RT) từ log tiền xử lý v2, so với bài báo — notebook 06 đã xuất riêng nhóm 39 trước loại EEG và nhóm ERP 35; mean ± sample SD qua subject, đối chiếu Table 5 và giữ chênh lệch thực tế
- [ ] Hình kiểm tra chất lượng: phổ công suất trước/sau lọc, ví dụ IC bị loại
- [ ] Script tổng hợp `results/metrics/*.csv` → bảng kết quả + hình F3
- [x] Hình F2 bản nháp (ERP + topomap, PNG 300 dpi/PDF); chờ R2 duyệt QC trước khi chốt
- [ ] Hình F3 theo quy ước hình — chờ metrics R4/R5; chưa có kết quả F3
- [x] Viết bản nháp *Results – ERP* tại `reports/results_erp_draft.md`; chờ duyệt QC và quy tắc P300 trước khi chốt report

**Đầu ra đã triển khai:** notebook 02–06, `src/eegproj/evaluation/{erp,topography,p300,behaviour_summary}.py`, hình F2 và bảng hành vi local, bản nháp Results – ERP. `evaluation/aggregate.py` và F3 chưa triển khai. Xem `reports/r3_review_and_handoff.md` để biết kiểm tra đã chạy và các phụ thuộc còn lại.

## R4 — Classical ML (+ Feature Extraction)
**Owner:** ______

- [ ] Module chia CV dùng chung (`within` theo khối dùng cột `block`, `cross` GroupKFold 5 fold theo subject, lưu `results/cv_cross_subject_folds.csv`) — **làm trước, DL dùng lại**
- [ ] Module metric dùng chung + ghi CSV đúng định dạng protocol §7
- [ ] `lda_tw`, `xdawn_rg` (bắt buộc), `bp_svm` (tùy chọn)
- [ ] Chạy đủ: 2 setting × 2 window × 3 task
- [ ] Thống kê RQ2 (Wilcoxon theo subject + Holm, protocol §8) — phối hợp R5; hình F4
- [ ] Viết mục *Methods – Classical ML* và phần RQ2 trong report

**Đầu ra:** `src/eegproj/features/`, `src/eegproj/models/classical.py`, `src/eegproj/evaluation/`, CSV kết quả.

## R5 — Deep Learning
**Owner:** Giang (Trần Bùi Hà Giang, 24022314)

- [x] Dataloader từ epoch `.fif` v2 (`models/dl_runner.load_dl_subject`): resample cả epoch 250 → 128 Hz rồi mới cắt cửa sổ; chuẩn hóa z-score theo kênh với thống kê **chỉ từ phần fit của train** (2026-11-19)
- [x] `eegnet` (EEGNet-8,2), `shallow` (ShallowFBCSPNet) viết bằng PyTorch thuần, cùng kiến trúc braindecode, các lớp đặt tên rõ để R1 đọc spatial filter / chạy IG — dùng **nguyên** `evaluation/splits.py` + `metrics.py` của R4 (lấy từ nhánh `Trung`, không sửa) và file fold cố định `results/cv_cross_subject_folds.csv` (`evaluation/folds.py`). Within-subject chia 5 đoạn liên tiếp đều nhau (`evaluation.within.split`, xem changelog protocol 2026-11-19) — **R4 cần dùng cùng cách chia**. `conformer` (tùy chọn): chưa làm
- [x] Early stopping trên validation tách từ train (within: 20% cuối của mỗi đoạn train; cross: 20% subject train), khôi phục trọng số tốt nhất; class weight cân bằng; hyperparameter ghi trong `models.dl` của config
- [x] Chạy đủ trên 35 subject: 2 setting × 2 window × `tgt_std` cho `eegnet` và `shallow` (2026-11-20, RTX 4060 Laptop, 4.6 giờ). EEGNet tốt nhất: trung vị bal. acc. within/full 0.941, cross/full 0.924, cross/early 0.830 — bảng đầy đủ trong `reports/methods_dl_draft.md`. Task đối chứng DL (`--tasks dis_std tgt_dis`): chưa chạy
- [x] Lưu checkpoint (gitignored) cho R1: `results/checkpoints/<model>/<setting>_<window>_<task>/*.pt`, đọc lại bằng `eegproj.models.dl.load_checkpoint`; kèm dự đoán từng trial `results/predictions/*.csv.gz` và log huấn luyện `results/dl_runs/*.json`
- [x] Chia sẻ outputs qua Drive (`r5_dl_results_20261120_fc0cfaf.zip`), tải bằng `python scripts/04_get_dl_results.py` (2026-11-20)
- [x] Viết bản nháp *Methods – Deep Learning* + bảng kết quả DL và so sánh EEGNet vs Shallow (`reports/methods_dl_draft.md`)

**Đầu ra:** `src/eegproj/models/dl.py`, `src/eegproj/models/dl_runner.py`, `src/eegproj/evaluation/folds.py`, `scripts/04_train_dl.py`, `tests/test_r5_dl.py`, CSV kết quả, checkpoints, `reports/methods_dl_draft.md`.

---

## Ma trận phụ thuộc

```
R2 Preprocessing ──► R3 ERP ───────────────► F2
        │
        ├──► R4 Classical ML ─┐
        │      (CV + metric)  ├──► R3 tổng hợp ──► F3, F4 ──► Report
        └──► R5 Deep Learning ┘          ▲
                    │                    │
                    └──► R1 Explainability ──► F5
```

## Phân bổ thuyết trình (10 phút)

| Phần | Thời lượng | Người trình bày |
|---|---|---|
| Bối cảnh, dữ liệu, câu hỏi | 1.5' | R1 |
| Tiền xử lý + ERP | 2' | R2, R3 |
| Kết quả RQ1/RQ2 | 3' | R4, R5 |
| Giải thích mô hình (RQ3) + nhiễu vận động | 2.5' | R1 |
| Kết luận | 1' | R1 |
