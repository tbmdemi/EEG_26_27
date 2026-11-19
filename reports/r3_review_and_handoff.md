# R3 — Bàn giao ERP & Visualization

**Sẵn sàng review code; kết quả chưa chốt.** Nhánh `nxb`, commit nền [b97a0684](https://github.com/tbmdemi/EEG_26_27/commit/b97a0684a14a1d999b5f73025d04643220eb77c0). Các sửa đổi sau review chưa commit/push. Kiểm tra lại ngày 2026-11-19.

## Đã làm và kiểm tra

- Notebook **02–06**: ERP/SEM, difference wave, topomap/F2, ứng viên P300 và bảng hành vi. Dùng dữ liệu **v2, 35/42 subject ERP**; không tải lại dữ liệu.
- Chạy từng notebook trong kernel mới: **26/26 ô code đạt**. Kiểm tra cú pháp, Ruff check/format và **8 test hồi quy đạt**. Đã đối chiếu độc lập ERP/SEM, difference, topomap và đỉnh với NumPy; kiểm tra tự động đủ 35 FIF. Đây chưa phải duyệt visual QC.
- Sửa bộ nạp để từ chối log thiếu subject, trộn v1/v2 hoặc lệch cohort bàn giao; kiểm tra metadata subject/RT/block/onset. Sửa lint, so sánh thời gian với dung sai và cảnh báo pandas. Không đổi công thức, tham số config hay code preprocessing/classification.
- Trung bình trial trong từng subject rồi trung bình qua người; SEM dùng SD mẫu (`ddof=1`). Difference-wave SEM dùng hiệu ghép cặp. Ghép kênh theo tên; đổi V → µV và s → ms khi xuất số đo. Topomap dùng vị trí điện cực trung bình trong hệ tọa độ đầu.
- Đầu ra đã kiểm tra: **20.600 epoch** (15.744 standard, 2.477 target, 2.379 distractor), 3.012 dòng ERP, 508/2.032 dòng topomap cửa sổ/lát cắt và 210 ứng viên đỉnh. Bản nháp số liệu: [Results – ERP](results_erp_draft.md).

## P300

**Đề xuất thăm dò, chưa được nhóm duyệt:** `erp.p300` trong config; cửa sổ cố định **300–600 ms**, Pz chính, CPz/Cz bổ sung, target/distractor riêng. Cơ sở tham khảo: [Dzianok et al., 2022](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac015/6543635). Cửa sổ topomap 250–600 ms chỉ dùng hiển thị.

Lấy mẫu ERP từng subject có điện thế lớn nhất **có dấu**; nếu bằng nhau lấy mẫu đầu và gắn cờ. Biên độ µV, latency ms, độ phân giải 4 ms ở 250 Hz; xuất thêm trung bình toàn cửa sổ. Không lọc thêm, nội suy hay đổi cửa sổ theo người.

Gắn cờ cực đại không dương, tại biên/cách biên ≤20 ms, không phải cực đại cục bộ hoặc có nhiều mẫu bằng cực đại. `peak_needs_review` khác `visual_review` của R2. Giữ mọi subject và ứng viên có cờ; so sánh ghép cặp theo người, không coi ba kênh là ba người độc lập. Ở Pz có **5 target và 1 distractor** cần xem lại; không tự xác nhận ứng viên là P300/P3a/P3b.

## Hành vi

Đọc log v2 **trước khi loại epoch nhiễu**, dùng số kích thích thực tế của từng người. Nhóm trước loại EEG có **39 người** (loại trước sub-02/14/17); nhóm ERP còn **35** (loại thêm sub-08/23/35/41). Hai nhóm chồng lấp, chỉ đối chiếu mô tả.

- Hit rate = target đúng / tổng target × 100%; bỏ sót = 100% − hit rate.
- False alarm = (tổng kích thích − số trial đúng) / tổng kích thích × 100%, riêng standard/distractor.
- Mean/SD nhóm tính qua subject, trọng số bằng nhau, `ddof=1`; SD của RT tính từ `rt_mean_ms`, không trung bình `rt_sd_ms`. Thanh SD không phải CI.

Mốc Table 5 của bài báo nằm trong `erp.behaviour_reference`: RT 542,13 ± 65,50 ms; bỏ sót 5,60 ± 8,33%; false alarm distractor 9,52 ± 8,90%, standard 0,24 ± 0,46% (N=39). Hit rate 94,40 ± 8,33% suy từ bỏ sót. Giữ chênh lệch quan sát được, không ép khớp bài báo; chênh lệch tỷ lệ dùng điểm phần trăm, RT dùng ms.

## Còn chờ

| Người phụ trách | Việc cần xác nhận/cung cấp |
|---|---|
| R2 | Visual QC cả 35 subject; nguồn tọa độ `.bvef` của sub-12/sub-24 (33 người còn lại là CapTrak); commit tạo release còn `<hash>` trong biên bản; hình PSD trước/sau và IC bị loại. |
| R1 / nhóm | Duyệt cửa sổ/quy tắc P300, các ứng viên có cờ và F2/Results sau QC. |
| R4/R5 → R3 | Gửi metrics cùng cohort và đúng protocol; sau đó R3 triển khai `evaluation/aggregate.py` và F3. Hiện chưa có metrics CSV, chưa có F3. |

## Chạy lại và bàn giao

Chọn kernel `.venv`, Restart Kernel và Run All lần lượt **02 → 06**. Kiểm tra code:

```powershell
python -m ruff check src/eegproj/evaluation tests notebooks
python -m ruff format --check src/eegproj/evaluation tests notebooks
python -m unittest discover -s tests -v
```

CSV/manifest nằm ở `results/erp/`, hình ở `figures/`; output được gitignore và notebook nguồn không chứa output. Bằng chứng chạy local ở `outputs/r3_review/` (notebook đã chạy và JSON kiểm chứng). Đã dọn 20 file tạm/cache/placeholder, giữ code, dữ liệu và bằng chứng kiểm tra.

**Commit gợi ý:** `fix(erp): validate v2 inputs and document verified R3 handoff`.

**PR `nxb` → `main`:** mô tả ERP/topomap, P300 thăm dò và hành vi; ghi 26/26 ô + 8 test/Ruff đạt, đồng thời nêu rõ các mục còn chờ ở trên. Chưa tự commit, push hoặc tạo PR.
