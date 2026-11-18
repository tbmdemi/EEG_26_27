# 04 — Roles & tasks

Mỗi vai có **owner** chịu trách nhiệm chính, nhưng mọi người phải hiểu toàn bộ pipeline (ai cũng bị hỏi khi thuyết trình).
Nếu nhóm < 5 người: gộp *ERP & Visualization* vào *Preprocessing*, gộp *Explainability* vào *Deep Learning*.

Theo dõi tiến độ bằng **GitHub Issues** (mỗi checkbox bên dưới = 1 issue, gán đúng người, gắn label vai trò) và một **GitHub Project board** (To do / In progress / Review / Done).

---

## R1 — Project Manager + Explainability
**Owner:** Minh

- [ ] Tạo Issues/Project board từ file này; bật branch protection cho `main`
- [ ] Họp tiến độ hằng tuần (15–20 phút), ghi biên bản vào `reports/meeting_notes.md`
- [ ] Review PR liên quan đến protocol/config
- [ ] Explainability: spatial filter EEGNet, Integrated Gradients, LDA patterns (protocol §9)
- [ ] Phân tích đối chứng nhiễu vận động (`early` vs `full`, task `dis_std`, `tgt_dis`)
- [ ] Hình F1, F5
- [ ] Ghép report, kiểm tra tổng % đóng góp = 100%, gửi email đúng hạn

**Đầu ra:** `src/eegproj/explain/`, `figures/F1*`, `figures/F5*`, report hoàn chỉnh.

## R2 — Signal Preprocessing
**Owner:** ______

- [x] Pipeline tiền xử lý theo protocol §1, đọc tham số từ config (v1: sub-01…05, 2026-11-13)
- [x] Pipeline v2: lọc trial đúng hành vi, loại/gắn cờ subject, đánh dấu chỗ nối recording, chế độ `--download --delete-raw` (2026-11-17)
- [x] Chạy `python scripts/01_preprocess.py --download --delete-raw` cho **toàn bộ subject** (42 subject, 35 dùng được; phát hành `preproc_v2_20261118.zip`)
- [ ] Kiểm tra bằng mắt hình QC (`figures/preprocessing/`), ưu tiên subject có cờ trong cột `flags`; cập nhật `visual_review` / `bad_channels.manual` nếu cần rồi chạy lại subject đó
- [x] Upload `preproc_v2_20261118.zip` lên Drive `EEG_26_27/release/` (**không** qua Git), cập nhật `RELEASE` trong `scripts/00_get_data.py` (2026-11-18)
- [ ] Commit code v2 + `results/preproc_log.csv`, ghi commit hash vào `reports/meeting_notes.md` → mở Issue **"Data v2 ready"**
- [ ] Viết mục *Methods – Preprocessing* + bảng subject/epoch bị loại trong report

**Đầu ra:** `src/eegproj/preprocessing/`, `scripts/01_preprocess.py`, epoch `.fif`, log.
**Data v2 đã phát hành (2026-11-18).** Mọi người lấy dữ liệu bằng `python scripts/00_get_data.py` và chỉ làm việc trên v2; dữ liệu v1 đã bỏ.

## R3 — ERP & Visualization
**Owner:** ______

- [ ] Grand-average ERP 3 loại kích thích tại Pz/CPz/Cz (toàn bộ subject dùng được, mean ± SEM qua subject); difference wave; topomap theo thời gian
- [ ] Đo biên độ và latency đỉnh P300 (target, distractor) theo từng subject → bảng + phân bố
- [ ] Bảng hành vi (hit rate, false alarm, RT) từ `results/preproc_log.csv`, so với bài báo
- [ ] Hình kiểm tra chất lượng: phổ công suất trước/sau lọc, ví dụ IC bị loại
- [ ] Script tổng hợp `results/metrics/*.csv` → bảng kết quả + hình F3
- [ ] Hình F2, F3 theo quy ước hình
- [ ] Viết mục *Results – ERP* trong report

**Đầu ra:** `notebooks/`, `src/eegproj/evaluation/aggregate.py`, `figures/F2*`, `figures/F3*`.

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
**Owner:** ______

- [ ] Dataloader từ epoch `.fif`, chuẩn hóa theo kênh với thống kê từ train
- [ ] `eegnet`, `shallow` (bắt buộc), `conformer` (tùy chọn) — dùng **cùng** module chia CV của R4
- [ ] Early stopping trên validation tách từ train; ghi hyperparameter vào `configs/config.yaml`
- [ ] Chạy đủ: 2 setting × 2 window × task chính (`tgt_std`); task đối chứng nếu còn thời gian
- [ ] Lưu checkpoint (gitignored) để R1 làm explainability
- [ ] Viết mục *Methods – Deep Learning* trong report

**Đầu ra:** `src/eegproj/models/dl.py`, `scripts/04_train_dl.py`, CSV kết quả, checkpoints.

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
