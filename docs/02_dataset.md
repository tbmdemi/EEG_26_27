# 02 — Dataset

**The Nencki-Symfonia EEG/ERP dataset** — Dzianok et al., *GigaScience* 2022.
- GigaDB DOI: 10.5524/100990
- OpenNeuro: `ds004621` — https://openneuro.org/datasets/ds004621
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8900497/

## Thông số ghi

| Mục | Giá trị |
|---|---|
| Người tham gia | 42 người khỏe mạnh, 20–34 tuổi, thuận tay phải (22 nữ) |
| Máy | Brain Products actiCHamp, actiCAP 128 kênh active; vị trí điện cực số hóa bằng CapTrak |
| Tần số lấy mẫu | 1000 Hz (low-pass 280 Hz khi ghi; **không** có high-pass/notch) |
| Tham chiếu online | FCz (có thêm A1/A2 = TP9/TP10) |
| Định dạng | BIDS / EEG-BIDS; BrainVision (`.vhdr`, `.eeg`, `.vmrk`); events trong `*_events.tsv` |
| Tình trạng | **Dữ liệu thô** — có drift, chưa làm sạch nhiễu |

Dataset có 4 task (MSIT+, oddball, SRT, rest). **Project chỉ dùng task oddball.**

## Task oddball (3 kích thích, thị giác)

| Loại | Mã sự kiện | Tỉ lệ | Hành động |
|---|---|---|---|
| Standard (frequent) | `S  5` | ~76% | Không bấm |
| Target | `S  6` | ~12% | **Bấm phím "2"** (ngón giữa tay phải) |
| Distractor (deviant) | `S  7` | ~12% | Không bấm |
| Response | `S  1` | — | Thời điểm bấm |
| Break | `S 10` | — | 4 lần nghỉ 15 s |
| Begin run | `S 12` | — | Bắt đầu |

- Khoảng **660 kích thích**/người (5 người có 659–775 do lỗi ghi).
- Mỗi kích thích hiện 200 ms; ISI 1200–1600 ms. Hai kích thích hiếm không xuất hiện liền nhau.
- Hành vi (N=39): RT target ≈ **542 ± 66 ms**; bỏ sót target 5.6%; false alarm với distractor 9.5%.
- ⇒ mỗi người ≈ 500 standard / 80 target / 80 distractor (**mất cân bằng ~6:1**).

> Mã sự kiện trong `.vmrk`/`events.tsv` có thể có khoảng trắng (`"S  5"`). Kiểm tra bằng `mne.events_from_annotations` trước khi map.

## Tải dữ liệu

Chỉ tải task oddball (tiết kiệm dung lượng). Dùng `openneuro-py` (đã có trong `requirements.txt`):

```bash
openneuro-py download --dataset ds004621 --target-dir data/raw/ds004621 \
    --include "sub-*/eeg/*oddball*" \
    --include "participants.tsv" --include "dataset_description.json" \
    --include "task-oddball_events.json"
```

**Cách khác — bản sao của nhóm trên Google Drive:**
https://drive.google.com/drive/folders/1Lkgc9Rn_nBzV1dR6im9uFgZM5fPsNAcw

Thư mục Drive có cấu trúc `data/raw/ds004621/...` giống hệt trong repo. Tải về rồi đặt sao cho đường dẫn là `<repo>/data/raw/ds004621/sub-*/...` (khớp `paths.raw` trong `configs/config.yaml`).

Sau khi tải, **kiểm tra**: đủ 42 thư mục `sub-*`, mỗi thư mục có bộ `.vhdr/.eeg/.vmrk` + `events.tsv` của task oddball. Nếu tên task trong file khác `oddball`, sửa pattern `--include` và cập nhật mục này.

Lưu ý:
- `data/` đã nằm trong `.gitignore`. **Không commit dữ liệu.**
- Mỗi người tự tải về máy mình; nếu máy yếu, chỉ cần tải vài subject để phát triển code, chạy full trên máy mạnh nhất nhóm.
