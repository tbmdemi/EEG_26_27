# Contributing - quy trình làm việc chung

## 1. Nguyên tắc vàng

1. **Không commit thẳng lên `main`.** Mọi thay đổi đi qua Pull Request (PR).
2. **Không commit dữ liệu, checkpoint, file lớn** (> 5 MB) **hoặc output sinh ra khi chạy notebook/script**, theo thông báo cập nhật của nhóm trưởng. Giữ code trong Git; chia sẻ output qua thư mục chung. Riêng `results/preproc_log.csv` là log bàn giao dữ liệu được giữ trong Git.
3. **Không hard-code tham số.** Đọc từ `configs/config.yaml`.
4. **Làm đúng protocol** (`docs/03_analysis_protocol.md`). Muốn thay đổi thì mở Issue trước.
5. **Ghi đóng góp** vào `reports/contribution_log.md` mỗi khi xong một việc.

## 2. Vòng làm việc

```bash
git checkout main
git pull                                   # luôn cập nhật trước khi bắt đầu
git checkout -b <loại>/<vai>-<mô-tả>       # ví dụ: feat/preproc-ica
# ... code ...
git add <file>
git commit -m "feat(preproc): add ICLabel-based component rejection"
git push -u origin feat/preproc-ica
```

Sau đó lên GitHub, chọn **Compare & pull request**, điền template, gán **1 reviewer**, chờ reviewer approve rồi **Squash and merge** và xóa branch.

Nếu `main` đã thay đổi trong lúc bạn làm:

```bash
git fetch origin
git rebase origin/main          # sửa conflict nếu có, rồi: git add <file> && git rebase --continue
git push --force-with-lease
```

## 3. Đặt tên branch

`<loại>/<vai>-<mô-tả-ngắn>`

| Loại | Dùng khi |
|---|---|
| `feat` | Thêm chức năng mới |
| `fix` | Sửa lỗi |
| `exp` | Thử nghiệm / chạy mô hình |
| `docs` | Tài liệu, report |
| `refactor` | Tổ chức lại code, không đổi hành vi |

Vai: `preproc`, `erp`, `ml`, `dl`, `explain`, `pm`.


## 4. Quy ước code

- Python >= 3.10 (team reference: 3.11); định dạng bằng `ruff format`, kiểm tra bằng `ruff check`.
- Code dùng lại được đặt trong `src/eegproj/`; script chạy end-to-end đặt trong `scripts/NN_<tên>.py`; notebook khám phá đặt trong `notebooks/`.
- Notebook: đặt tên `NN_<vai>_<mô-tả>.ipynb`; output được `nbstripout` tự xóa khi commit. Logic quan trọng phải chuyển sang `src/`.
- Sau khi cài thư viện, chạy `nbstripout --install` một lần trên mỗi máy để bật xóa output nhúng trong notebook khi commit; `.gitignore` không xóa được output nằm bên trong `.ipynb`. Vẫn commit file notebook chứa code.
- Hình và bảng sinh ra khi chạy lưu vào `figures/`, `results/` hoặc các thư mục `outputs/`, `notebooks/outputs/`, `scripts/outputs/` đã được gitignore. Giữ đúng tên/định dạng đầu ra theo protocol để các role đọc kết quả local hoặc từ thư mục chia sẻ.
- Hàm công khai có docstring ngắn (đầu vào, đầu ra, đơn vị; ví dụ µV, giây).
- Đặt seed từ config cho numpy / torch / sklearn.
- Đường dẫn: dùng `pathlib` và đường dẫn tương đối từ gốc repo, lấy từ config.

## 5. Chia sẻ file lớn

Dữ liệu đã tiền xử lý, checkpoint, hình và bảng kết quả sinh ra khi chạy: để trên thư mục chia sẻ chung của nhóm (Drive/OneDrive). Link và sha256 ghi trong `RELEASE` ở đầu script tải tương ứng (`scripts/00_get_data.py` cho dữ liệu, `scripts/04_get_dl_results.py` cho outputs DL). Khi cập nhật, ghi ngày + commit hash của code tạo ra file đó vào tên file zip.

Dữ liệu tiền xử lý hiện hành: **pipeline v2** (`PIPELINE_VERSION = 2`, trial đúng hành vi, toàn bộ subject dùng được). Nếu `epochs.metadata` của bạn **không có cột `rt`**, đó là file v1 cũ; xóa file đó và tải lại. Không trộn dữ liệu từ nguồn tiền xử lý khác vào kết quả.