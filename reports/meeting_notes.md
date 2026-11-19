# Meeting notes

## Link chia sẻ
- Dữ liệu đã tiền xử lý (pipeline v2): thư mục `EEG_26_27/release/` https://drive.google.com/drive/folders/1sqrYetZH3xr_eFbH2Cce8V_iwulI44KR
  - File `preproc_v2_20261118.zip`, FILE_ID cho `gdown`: `1hXdIGzBk8p1QziEy3DARKdihhgpi95S9` (tải bằng `python scripts/00_get_data.py`)
  - Cập nhật: 2026-11-18 · tracked v2 code: `6fccfc2502f3ee64a919de3ac97e86f52a5e3361` (PR #4; exact generating-job revision was not recorded in the archive) · số subject dùng được: 35 · sha256: c53f0407
- Checkpoints: <link>
- Slide: <link>

---

## Họp #1 — <ngày>
**Có mặt:**
**Đã xong từ lần trước:**
**Vướng mắc:**
**Quyết định:**
**Việc tuần tới (ai — việc — hạn):**
-

## R2 completion - 2026-11-19

- R2 implementation/QC commit: `fbc41f4a0f58b23f3c8cd2af73e713d6e83c3af9` (local; not yet pushed). The normalized core-code fingerprint matches both real-data candidate summaries.
- 11/11 tests passed with network disabled; Ruff passed; modular notebook passed real-kernel audit of 35 FIFs and 42 event tables.
- Published v2 FIFs/archive files remain unchanged. Repository CSV contains documented human QC annotations only.
- Candidate sub-10/sub-13: 591/648 epochs; local review ZIP: `data/r2_candidate_review_20261119.zip`. A versioned release decision is still needed before using these instead of v2.
- Data v2 ready Issue body: `reports/data_v2_ready_issue.md`; not posted because GitHub is not connected.
