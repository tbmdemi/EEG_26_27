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
- [x] Visual summary QC đủ 35 subject dùng được; đã ghi reviewer/scope/findings vào log, cập nhật manual OI1h (sub-10) và AFF1h (sub-13), chạy lại hai candidate từ raw thật. Đây không phải duyệt tương tác toàn bộ raw; candidate chưa thay v2. Xem `reports/r2_qc_and_handoff.md`.
- [x] Upload `preproc_v2_20261118.zip` lên Drive `EEG_26_27/release/` (**không** qua Git), cập nhật `RELEASE` trong `scripts/00_get_data.py` (2026-11-18)
- [x] Commit code R2 và log QC: `fbc41f4`; ghi hash vào `reports/meeting_notes.md`. Bản v2 lịch sử tham chiếu `6fccfc2`; không khẳng định đây là revision chính xác của job sinh ZIP.
- [x] Mở Issue **[Data v2 ready #6](https://github.com/tbmdemi/EEG_26_27/issues/6)**; nội dung lưu tại `reports/data_v2_ready_issue.md`. Commit mới hiện chỉ local, chưa push; hai candidate vẫn chờ quyết định phát hành.
- [x] Viết *Methods – Preprocessing* v2 vào report §3.1; bảng đủ 42 subject và accounting epoch tại `reports/r2_subject_epoch_table.md`. Report vẫn dùng v2 (20.600 epoch), tách khỏi candidate.

**Đầu ra:** v2 hiện hành từ pipeline nhóm; notebook `01_preprocess.ipynb`, code cải tiến `scripts/r2_*.py`, CLI `scripts/02_r2.py`; candidate `.fif`/log/QC riêng. Bản phát hành v2 chưa bị ghi đè.
**Data v2 đã phát hành (2026-11-18).** Mọi người lấy dữ liệu bằng `python scripts/00_get_data.py` và chỉ làm việc trên v2; dữ liệu v1 đã bỏ.

## R3 — ERP & Visualization
**Owner:** Bảo

- [x] Grand-average ERP 3 loại kích thích tại Pz/CPz/Cz (toàn bộ subject dùng được, mean ± SEM qua subject); difference wave; topomap theo thời gian — đã tạo bản nháp trên 35 subject bằng notebook 03/04; QC tổng hợp R2 đã hoàn thành, chờ nhóm chốt phiên bản dữ liệu và duyệt hình cuối cho report
- [ ] Đo biên độ và latency đỉnh P300 (target, distractor) theo từng subject → bảng + phân bố — đã có bản thăm dò notebook 05 (210 ứng viên, 35 subject × 2 điều kiện × 3 kênh); chờ review cửa sổ/quy tắc đo và các trường hợp có cờ trước khi chốt
- [x] Bảng hành vi (hit rate, false alarm, RT) từ log tiền xử lý v2, so với bài báo — notebook 06 đã xuất riêng nhóm 39 trước loại EEG và nhóm ERP 35; mean ± sample SD qua subject, đối chiếu Table 5 và giữ chênh lệch thực tế
- [ ] Hình kiểm tra chất lượng: phổ công suất trước/sau lọc, ví dụ IC bị loại
- [ ] Script tổng hợp `results/metrics/*.csv` → bảng kết quả + hình F3
- [x] Hình F2 bản nháp (ERP + topomap, PNG 300 dpi/PDF); QC tổng hợp R2 đã hoàn thành, chờ nhóm chốt phiên bản dữ liệu và duyệt hình cuối
- [ ] Hình F3 theo quy ước hình — chờ metrics R4/R5; chưa có kết quả F3
- [x] Viết bản nháp *Results – ERP* tại `reports/results_erp_draft.md`; QC tổng hợp R2 đã hoàn thành, chờ nhóm chốt phiên bản dữ liệu, duyệt quy tắc P300 và hình cuối trước khi chốt report

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
