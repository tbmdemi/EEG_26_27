# 02 — Dataset

**The Nencki-Symfonia EEG/ERP dataset** — Dzianok et al., *GigaScience* 2022.
- GigaDB DOI: 10.5524/100990
- OpenNeuro: `ds004621` — https://openneuro.org/datasets/ds004621
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8900497/

## Thông số ghi

| Mục | Giá trị |
|---|---|
| Người tham gia | 42 người khỏe mạnh, 20–34 tuổi, thuận tay phải (22 nữ) — `sub-01` … `sub-42` |
| Máy | Brain Products actiCHamp, actiCAP 128 kênh active; vị trí điện cực số hóa bằng CapTrak |
| Tần số lấy mẫu | 1000 Hz (low-pass 280 Hz khi ghi; **không** có high-pass/notch) |
| Tham chiếu online | FCz (có thêm A1/A2 = TP9/TP10) |
| Định dạng | BIDS / EEG-BIDS; BrainVision (`.vhdr`, `.eeg`, `.vmrk`); events trong `*_events.tsv`; file chứa **127 kênh** EEG (FCz là tham chiếu, không có trong file) |
| Tình trạng | **Dữ liệu thô** — có drift, chưa làm sạch nhiễu |

Dataset có 4 task (MSIT+, oddball, SRT, rest). **Project chỉ dùng task oddball.**

## Task oddball (3 kích thích, thị giác)

| Loại | Mã sự kiện (`event_type`) | Tỉ lệ | Hành động đúng |
|---|---|---|---|
| Standard (frequent) | `S  5` | ~76% (498) | Không bấm |
| Target | `S  6` | ~12% (81) | **Bấm phím "2"** (ngón giữa tay phải) |
| Distractor (deviant) | `S  7` | ~12% (81) | Không bấm |
| Response | `S  1` | — | Thời điểm bấm |
| Break | `S 10` | — | 4 lần nghỉ 15 s → chia recording thành 5 block |
| Begin run | `S 12` | — | Bắt đầu |

- **660 kích thích**/người (5 người có số khác: xem bảng dưới).
- Mỗi kích thích hiện 200 ms; ISI 1200–1600 ms; hai kích thích hiếm không xuất hiện liền nhau. Kích thước: standard/distractor ≈ 1.07°, target ≈ 1.22° góc nhìn (target hơi lớn hơn → có thể khác biệt thị giác sớm P1/N1, cần lưu ý khi giải thích mô hình).
- Hành vi (bài báo, N = 39): RT target ≈ **542 ± 66 ms**; bỏ sót target 5.6%; false alarm với distractor 9.5%, với standard 0.24%.
- Cột `trial_type` chỉ có `stimulus`/`response`; **phải dùng cột `event_type`** để phân biệt 3 loại kích thích.

> Trong annotation của MNE, mã có dạng `"Stimulus/S  5"` (2 khoảng trắng). Xem `dataset.events` trong config.

## Lọc trial theo hành vi (bắt buộc)

Chỉ giữ các trial có **phản hồi hành vi đúng**, xác định từ `*_events.tsv` (code: `src/eegproj/preprocessing/behaviour.py`, tham số: `behaviour` trong config). Một lần bấm `S  1` được tính cho kích thích nếu nó nằm trong **[100 ms, 1400 ms]** sau onset (= 200 ms kích thích + 1200 ms ISI ngắn nhất; theo bài báo, mục *Reliability – Methods*):

| Loại | Trial đúng |
|---|---|
| Target | Có ≥ 1 lần bấm trong cửa sổ (RT = lần bấm đầu tiên) |
| Standard | Không có lần bấm nào trong cửa sổ |
| Distractor | Không có lần bấm nào trong cửa sổ |

Kiểm chứng trên dữ liệu thật: với định nghĩa này, sub-02 có false alarm với distractor = **98.8%**, trùng con số bài báo báo cáo.

## Danh sách subject

| Subject | Xử lý | Lý do (nguồn) |
|---|---|---|
| sub-02, sub-14, sub-17 | **Loại trước** (`dataset.exclude`) | Hiểu sai hướng dẫn, bấm cả với distractor: false alarm 98.8%, 97.5%, 76.5% (bài báo, mục *Behavioral data*) |
| sub-13, sub-23, sub-24 | Giữ, **gắn cờ** (`dataset.flag`) | Số kích thích ≠ 660 do lỗi ghi |
| sub-37, sub-38 | Giữ, **gắn cờ** | Recording nối từ nhiều file (marker `New Segment`); đoạn ±0.5 s quanh chỗ nối bị đánh dấu BAD |
| 34 người còn lại | Giữ | — |

Ngoài ra pipeline **tự loại** subject nếu: có < 40 target đúng, hoặc giữ < 50% target đúng sau khi loại epoch nhiễu. Hit rate < 80% chỉ **gắn cờ** để xem lại (bài báo không loại ai vì hit rate; ví dụ sub-01 có hit rate 76.5% nhưng vẫn hợp lệ). Danh sách cuối cùng và lý do nằm trong `results/preproc_log.csv`.

## Lấy dữ liệu

### Thành viên (ERP, ML, DL, Explainability) — chỉ cần dữ liệu đã tiền xử lý

**Không cần tải dữ liệu thô.** PM/R2 chạy tiền xử lý một lần cho cả nhóm và chia sẻ:

Dữ liệu được chia sẻ dưới dạng **một file zip có phiên bản** trong thư mục `EEG_26_27/release/` trên Drive (https://drive.google.com/drive/folders/1sqrYetZH3xr_eFbH2Cce8V_iwulI44KR).

| Bản hiện hành | |
|---|---|
| File | `preproc_v2_20261118.zip` (2.6 GB) |
| Nội dung | `data/derivatives/preproc/` **trừ** `_excluded/`: `*_epo.fif`, `*_ica.fif`, `*_summary.json`, `preproc_log.csv`, `config.json` |
| Subject dùng được | 35 / 42 |
| sha256 | `c53f040786b6c23a9b5d135a31f0bef966088018434409a1ca44a2e944c7a6ef` |

Thông tin bản phát hành (FILE_ID, sha256, số subject) nằm trong `RELEASE` ở đầu `scripts/00_get_data.py`. `results/preproc_log.csv` đi kèm trong repo (commit được).

**Máy cá nhân hoặc Colab:** chạy ở gốc repo:

```bash
python scripts/00_get_data.py                 # tải (gdown), kiểm tra sha256, giải nén vào data/derivatives/preproc/
python scripts/00_get_data.py --zip FILE.zip  # nếu đã tải zip bằng trình duyệt (khi Drive giới hạn lượt tải)
python scripts/00_get_data.py --check         # chỉ kiểm tra dữ liệu đang có
```

Trên Colab: `!git clone https://github.com/tbmdemi/EEG_26_27.git && cd EEG_26_27 && pip install -q -r requirements.txt`, rồi `!python scripts/00_get_data.py`. Script giải nén vào ổ local của Colab, nhanh hơn nhiều so với đọc `.fif` trực tiếp từ Drive.

File cũ trong `data/derivatives/preproc/` không thuộc bản phát hành sẽ được chuyển sang `_old/`, script không xoá chúng.

**Kiểm tra sau khi tải:**

```python
from eegproj.config import find_root, load_config
from eegproj.preprocessing import usable_subjects, verify_epochs
root = find_root(); config = load_config(root)
subjects = usable_subjects(root, config)     # chỉ subject pipeline v2, không bị loại
print(len(subjects), subjects)               # phải khớp n_usable trong RELEASE (35)
verify_epochs(root, config, subjects)        # kiểm tra từng file .fif
```

Quy tắc: **không sửa, không tự tạo lại** dữ liệu trong `data/derivatives/preproc/`. Nếu cần thay đổi tiền xử lý → mở Issue (protocol change); PM chạy lại và phát hành zip phiên bản mới (`preproc_v3_...`). Trong report ghi rõ tên file zip đã dùng.

Mỗi file `sub-XX_task-oddball_epo.fif` chứa:
- **Chỉ các trial đúng hành vi** của 3 loại `standard`, `target`, `distractor` (event_id 1, 2, 3).
- 127 kênh, 250 Hz, −0.2 → 0.8 s (251 mẫu), đã baseline −0.2 → 0 s, đơn vị V.
- `epochs.metadata`: `subject`, `onset_s` (onset gốc, giây), `block` (0–4, theo các lần nghỉ — dùng cho CV theo khối), `rt` (giây; chỉ target có giá trị).

Chỉ dùng các subject có `excluded_subject = False` **và** `pipeline_version = 2` trong `results/preproc_log.csv` (hoặc gọi `eegproj.preprocessing.usable_subjects(root, config)`, hàm này kiểm tra cả hai). File của subject bị loại nằm riêng trong `data/derivatives/preproc/_excluded/`, **không** dùng cho phân tích chính.

### PM / R2 — tạo dữ liệu đã tiền xử lý

Raw oddball ≈ 0.7 GB/người (~26 GB cho 39 người). Nếu ổ đĩa không đủ, dùng chế độ tải → xử lý → xóa từng người:

```bash
python scripts/01_preprocess.py --download --delete-raw      # toàn bộ subject trong config
python scripts/01_preprocess.py --download -s 01 03          # chỉ vài subject, giữ lại raw
```

- `--download`: tải từ OpenNeuro (`openneuro-py`) **chỉ** các file `sub-XX/eeg/*task-oddball*` + metadata nhỏ; tọa độ CapTrak (`sourcedata/sub-XX/coords/*.bvct`) và `task-oddball_eeg.json` được tải tự động.
- `--delete-raw`: sau khi xử lý xong một người, xóa `.eeg/.vhdr/.vmrk` của người đó (giữ `.tsv/.json`).
- Chạy lại lệnh sẽ **bỏ qua** subject đã xử lý bằng phiên bản pipeline hiện tại; subject xử lý bằng phiên bản cũ được xử lý lại tự động.
- Xong: kiểm tra `results/preproc_log.csv` + hình trong `figures/preprocessing/`, rồi phát hành dữ liệu:

```bash
# ở gốc repo, sau khi đã commit code + results/preproc_log.csv
D=$(date +%Y%m%d)
cd data/derivatives/preproc && zip -r ../../../preproc_v2_$D.zip . -x "_excluded/*" && cd -
sha256sum preproc_v2_$D.zip; stat -c %s preproc_v2_$D.zip
git rev-parse --short HEAD          # commit hash ghi vào Issue "Data vN ready"
```

Upload zip vào `EEG_26_27/release/` trên Drive (quyền thành viên: **Viewer**, chia sẻ "Anyone with the link" để `gdown` tải được). Sau đó cập nhật `RELEASE` trong `scripts/00_get_data.py` (tên, FILE_ID, sha256, `n_usable`, `n_bytes`), rồi mở Issue "Data vN ready" ghi link, ngày, commit hash, số subject dùng được. Không để zip cũ lẫn trong thư mục `release/`.

Tải thủ công (nếu cần), ví dụ cho sub-01:

```bash
openneuro-py download --dataset ds004621 --target-dir data/raw/ds004621 \
    --include "sub-01/eeg/*task-oddball*" \
    --include "participants.tsv" --include "dataset_description.json" \
    --include "task-oddball_events.json"
```

Lưu ý:
- `data/` đã nằm trong `.gitignore`. **Không commit dữ liệu.**
- Một số `*_channels.tsv` lưu `µV` bằng mã Latin-1; `prepare_raw` / `ensure_utf8_tsv` tự chuyển sang UTF-8.
