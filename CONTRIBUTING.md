# Contributing — quy trình làm việc chung

## 1. Nguyên tắc vàng

1. **Không commit thẳng lên `main`.** Mọi thay đổi đi qua Pull Request (PR).
2. **Không commit dữ liệu, checkpoint, file lớn** (> 5 MB). `data/` và `results/checkpoints/` đã được gitignore.
3. **Không hard-code tham số.** Đọc từ `configs/config.yaml`.
4. **Làm đúng protocol** (`docs/03_analysis_protocol.md`). Muốn thay đổi → mở Issue trước.
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

Sau đó lên GitHub → **Compare & pull request** → điền template → gán **1 reviewer** → reviewer approve → **Squash and merge** → xóa branch.

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

## 4. Commit message (Conventional Commits)

```
<loại>(<phạm vi>): <mô tả ngắn, tiếng Anh, động từ nguyên mẫu>
```

Ví dụ: `feat(dl): add EEGNet training loop with early stopping`, `fix(ml): fit xDAWN on train fold only`, `docs(report): draft preprocessing methods`.

Commit nhỏ, mỗi commit một ý. Commit **dưới tên GitHub của chính mình** — đây là bằng chứng đóng góp khi chấm điểm. Kiểm tra: `git config user.email` phải trùng email trong tài khoản GitHub.

## 5. Review PR

Reviewer kiểm tra:
- [ ] Đúng protocol (đặc biệt: không rò rỉ dữ liệu, fit chỉ trên train)
- [ ] Tham số lấy từ config
- [ ] Chạy được theo hướng dẫn trong PR
- [ ] Kết quả ghi đúng định dạng CSV (protocol §7)
- [ ] Không có file dữ liệu / output notebook / file lớn

## 6. Quy ước code

- Python ≥ 3.11; định dạng bằng `ruff format`, kiểm tra bằng `ruff check`.
- Code dùng lại được → `src/eegproj/`; script chạy end-to-end → `scripts/NN_<tên>.py`; khám phá → `notebooks/`.
- Notebook: đặt tên `NN_<vai>_<mô-tả>.ipynb`; output được `nbstripout` tự xóa khi commit. Logic quan trọng phải chuyển sang `src/`.
- Hàm công khai có docstring ngắn (đầu vào, đầu ra, đơn vị — ví dụ µV, giây).
- Đặt seed từ config cho numpy / torch / sklearn.
- Đường dẫn: dùng `pathlib` và đường dẫn tương đối từ gốc repo, lấy từ config.

## 7. Chia sẻ file lớn

Dữ liệu đã tiền xử lý, checkpoint: để trên thư mục chia sẻ chung của nhóm (Drive/OneDrive). Link ghi trong `reports/meeting_notes.md`. Khi cập nhật, ghi ngày + commit hash của code tạo ra file đó.

## 8. Ghi nhận đóng góp

Sau mỗi việc hoàn thành, thêm một dòng vào `reports/contribution_log.md` (ngày, người, việc, link PR/Issue). Cuối dự án, nhóm dựa vào log này + lịch sử commit để thống nhất % đóng góp.
