# R3 — Bàn giao ERP & Visualization

**Cập nhật 2026-11-26:** phân tích ERP/P300 và F3 đã được tổng hợp trên release v2. Code và kết quả có thể review; chưa commit/push.

## Đã làm và kiểm tra

- Notebook **02–06**: ERP/SEM, difference wave, topomap/F2, ứng viên P300 và bảng hành vi. Dùng dữ liệu **v2, 35/42 subject ERP**; không tải lại dữ liệu.
- Chạy từng notebook trong kernel mới: **26/26 ô code đạt** theo handoff trước. Đối chiếu độc lập ERP/SEM, difference, topomap và đỉnh với NumPy; kiểm tra tự động đủ 35 FIF.
- Sửa bộ nạp để từ chối log thiếu subject, trộn v1/v2 hoặc lệch cohort bàn giao; kiểm tra metadata subject/RT/block/onset. Không đổi công thức, tham số số học, tiền xử lý hay phân loại.
- Trung bình trial trong từng subject rồi trung bình qua người; SEM dùng SD mẫu (`ddof=1`). Difference-wave SEM dùng hiệu ghép cặp. Ghép kênh theo tên; đổi V → µV và s → ms khi xuất số đo. Topomap dùng vị trí điện cực trung bình trong hệ tọa độ đầu.
- Đầu ra đã kiểm tra: **20.600 epoch** (15.744 standard, 2.477 target, 2.379 distractor), 3.012 dòng ERP, 508/2.032 dòng topomap cửa sổ/lát cắt và 210 ứng viên đỉnh. Bản nháp số liệu: [Results – ERP](results_erp_draft.md).
- Ngày 2026-11-26: thêm pipeline F3; toàn bộ **38 test** đạt, **11 test R3** đạt, Ruff check/format đạt.

## P300

Quy tắc mô tả trong `erp.p300` đã được PM duyệt cho report ngày 2026-11-22. Dùng cửa sổ cố định **300–600 ms**, Pz chính, CPz/Cz bổ sung, target/distractor riêng. Cơ sở tham khảo: [Dzianok et al., 2022](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac015/6543635). Cửa sổ topomap 250–600 ms chỉ dùng hiển thị.

Lấy mẫu ERP từng subject có điện thế lớn nhất **có dấu**; nếu bằng nhau lấy mẫu đầu và gắn cờ. Biên độ µV, latency ms, độ phân giải 4 ms ở 250 Hz; xuất thêm trung bình toàn cửa sổ. Không lọc thêm, nội suy hay đổi cửa sổ theo người.

Gắn cờ cực đại không dương, tại biên/cách biên ≤20 ms, không phải cực đại cục bộ hoặc có nhiều mẫu bằng cực đại. `peak_needs_review` khác `visual_review` của R2. Giữ mọi subject và ứng viên có cờ; so sánh ghép cặp theo người, không coi ba kênh là ba người độc lập. Ở Pz có **5 target và 1 distractor** cần xem lại; không tự xác nhận ứng viên là P300/P3a/P3b.

## F3

Đã thêm `src/eegproj/evaluation/aggregate.py` và `scripts/08_aggregate_results.py`. Lệnh tổng hợp đọc 16 CSV task `tgt_std` từ các mô hình R4/R5, xác minh đủ 35 subject, đủ 5 fold trong-subject, một điểm test/subject cross-subject và nhất quán fold cross-subject giữa các điều kiện. Trong-subject giảm bằng trung bình fold; cross-subject giữ điểm test subject. Đầu ra là 560 subject-score rows, 48 summary rows và F3 PNG/PDF. Manifest ghi SHA-256 các file metric đầu vào.

## Hành vi

Đọc log v2 **trước khi loại epoch nhiễu**, dùng số kích thích thực tế của từng người. Nhóm trước loại EEG có **39 người** (loại trước sub-02/14/17); nhóm ERP còn **35** (loại thêm sub-08/23/35/41). Hai nhóm chồng lấp, chỉ đối chiếu mô tả.

- Hit rate = target đúng / tổng target × 100%; bỏ sót = 100% − hit rate.
- False alarm = (tổng kích thích − số trial đúng) / tổng kích thích × 100%, riêng standard/distractor.
- Mean/SD nhóm tính qua subject, trọng số bằng nhau, `ddof=1`; SD của RT tính từ `rt_mean_ms`, không trung bình `rt_sd_ms`. Thanh SD không phải CI.

Mốc Table 5 của bài báo nằm trong `erp.behaviour_reference`: RT 542,13 ± 65,50 ms; bỏ sót 5,60 ± 8,33%; false alarm distractor 9,52 ± 8,90%, standard 0,24 ± 0,46% (N=39). Hit rate 94,40 ± 8,33% suy từ bỏ sót. Giữ chênh lệch quan sát được, không ép khớp bài báo; chênh lệch tỷ lệ dùng điểm phần trăm, RT dùng ms.

## Còn chờ

| Người phụ trách | Việc cần xác nhận/cung cấp |
|---|---|
| R2 | Handoff ghi đã review summary QC và đã tạo ví dụ PSD/IC. Ảnh PSD raw/trước-sau không có trong checkout và `data/raw` vắng mặt nên không thể tái tạo trung thực từ epoch v2. Đã tạo lại hình IC bị loại từ ICA v2 có sẵn: [sub-13 excluded ICs](../figures/preprocessing/v2/review/sub-13_task-oddball_excluded_ics_0.png), components 0, 1, 7, 11, 16, 17, 18. Chi tiết review/provenance vẫn ở `r2_qc_and_handoff.md`. |
| R1 / nhóm | Review trực quan bản cuối F2/F3 và chốt vị trí/định dạng ảnh trong report. Quy tắc P300 đã được duyệt; các ứng viên có cờ vẫn được giữ và nêu rõ. |
## Chạy lại và bàn giao

Chọn kernel `.venv`, Restart Kernel và Run All lần lượt **02 → 06** để tái tạo các kết quả ERP. Chạy tiếp script F3:

```bash
python scripts/08_aggregate_results.py
python -m ruff check src/eegproj/evaluation/aggregate.py scripts/08_aggregate_results.py tests/test_r3.py
python -m ruff format --check src/eegproj/evaluation/aggregate.py scripts/08_aggregate_results.py tests/test_r3.py
python -m unittest discover -s tests -p 'test_r3.py' -v
```

CSV/manifest nằm ở `results/aggregation/`, hình ở `figures/`; các kết quả sinh được gitignore. Hình QC R2 vẫn là phụ thuộc ngoài checkout được mô tả ở trên. Chưa tự commit, push hoặc tạo PR.
