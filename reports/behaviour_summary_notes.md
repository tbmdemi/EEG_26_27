# R3 — Cách tổng hợp hành vi

Notebook `06_erp_behaviour_summary.ipynb` đọc `preproc_log.csv` đi kèm bản dữ liệu v2 đã tải, không xử lý lại EEG hay chấm lại marker hành vi.

## Hai nhóm cần phân biệt

- **Trước khi loại theo chất lượng EEG:** 39 người, loại trước sub-02/14/17 theo cấu hình. Đây là nhóm có danh sách loại trước tương ứng nhóm oddball của bài báo.
- **Nhóm ERP hiện hành:** 35 người có `excluded_subject=False`. Sub-08/23/35/41 đã bị pipeline loại thêm vì tỷ lệ giữ target; hành vi của họ vẫn có trong nhóm 39.

Hai nhóm chồng lấp. Không coi chúng là hai mẫu độc lập, không kiểm định chênh lệch giữa hai nhóm này trong notebook.

## Công thức và đơn vị

Tính từ số kích thích gốc và phản hồi đúng **trước khi loại epoch nhiễu**:

- Hit rate = target đúng / tổng target × 100%.
- Bỏ sót target = 100% − hit rate.
- False alarm = (tổng kích thích − số trial đúng) / tổng kích thích × 100%, riêng cho standard/distractor.
- RT dùng `rt_mean_ms`: trung bình RT của target đúng trong từng người.

Dùng mẫu số thực tế theo từng người, không giả định mọi người đều có 81 target hay 498 standard. Trung bình và SD nhóm tính **qua subject**, mỗi người có trọng số bằng nhau; SD dùng `ddof=1`. SD nhóm của RT được tính từ các RT trung bình từng người, không lấy trung bình cột `rt_sd_ms` (SD trong từng người).

## Số đối chiếu

Các mốc mean ± SD lấy từ [Dzianok et al., 2022, Table 5](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac015/6543635): RT target 542,13 ± 65,50 ms; bỏ sót 5,60 ± 8,33%; false alarm distractor 9,52 ± 8,90%; false alarm standard 0,24 ± 0,46%, N = 39. Hit rate tham chiếu 94,40% được suy ra từ 100% − 5,60%; SD giữ 8,33 điểm phần trăm. Số tham chiếu được lưu ở `erp.behaviour_reference` trong config.

Đối chiếu mang tính mô tả. Nhóm 35 người có cỡ mẫu khác bài báo. Ngay cả nhóm 39 cũng không cần khớp mọi số công bố: bảng giữ chênh lệch thực tế và không tự quy nguyên nhân hay tuyên bố tái lập hoàn toàn. Chênh lệch tỷ lệ báo bằng **điểm phần trăm**, RT bằng ms.

## Đầu ra

`results/erp/behaviour_*` chứa bảng subject, danh sách bị loại, bảng tổng hợp/đối chiếu và manifest. `figures/R3_behaviour_comparison.png/.pdf` chứa các điểm từng người và mean ± SD. Output được gitignore. Các thanh SD mô tả độ phân tán, không phải CI hoặc khoảng giá trị có thể xảy ra của tỷ lệ.
