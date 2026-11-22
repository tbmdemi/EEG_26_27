# Báo cáo môn học (LaTeX)

Khung báo cáo tiếng Anh cho môn Neuroimaging: EEG. Mọi chỗ cần viết là ô đỏ `TODO` hoặc chữ đỏ `[...]`, kèm người phụ trách (R1 đến R5) và file nháp để lấy nội dung.

## Compile

Cần XeLaTeX (dùng `fontspec` để hiện tên tiếng Việt và µV), không dùng pdfLaTeX.

- Máy local: `tectonic main.tex` hoặc `latexmk main.tex` (file `latexmkrc` đã chọn XeLaTeX).
- Overleaf: tải cả folder lên, Menu > Compiler chọn XeLaTeX. Tải các hình cần dùng vào thư mục `figures/` rồi sửa `\figdir` trong `main.tex` thành `figures`.

Hình được lấy từ `figures/` của repo, dùng bản `.pdf` (vector, phóng to vẫn nét); riêng hình QC tiền xử lý dùng `.png` trong `figures/preprocessing/v2/`. Hình chưa có sẽ hiện khung đỏ `Missing figure`.

### Output R3 ngoài Git

F2/F3 và bảng ERP/P300/hành vi được chuẩn bị trong gói ZIP R3 riêng; hình và bảng không đi theo PR vì đã gitignore. Xem [bàn giao R3](../r3_review_and_handoff.md) để lấy tên phiên bản, SHA256, inventory và lệnh giải nén **từ gốc repo**. Sau khi giải nén, `figures/F2_erp_topomap.pdf` và `figures/F3_balanced_accuracy.pdf` phải tồn tại; `main.tex` tìm đúng hai file này qua `\figdir=../../figures` khi build trong `reports/latex/`. Trên Overleaf, chép các hình vào thư mục `figures/` của project và đổi `\figdir` thành `figures`.

F2 trong gói là bản gốc v2 chờ duyệt hình cuối; chú thích QC-pending trong ảnh là metadata lịch sử trước khi R2 hoàn tất summary QC. Gói R3 chưa được coi là đã phát hành trên Drive, không có hình QC/PSD–ICA còn phụ thuộc R2 và không thay thế F1/F4/F5 của các vai khác. `Missing figure` là khung giữ chỗ, không phải bằng chứng báo cáo đã có đủ hình. Máy chuẩn bị gói chưa có TeX engine: đã kiểm tra tĩnh cú pháp/số liệu, còn kiểm tra build và phân trang toàn báo cáo trên môi trường của nhóm.

## Cấu trúc

| File | Nội dung | Phụ trách |
|---|---|---|
| `frontmatter/titlepage.tex` | Trang bìa, thành viên | R1 |
| `frontmatter/abstract.tex` | Tóm tắt, từ khoá | R1 |
| `chapters/01_introduction.tex` | Bối cảnh, RQ | R1 |
| `chapters/02_data.tex` | Dữ liệu, loại subject, hành vi | R2, R3 |
| `chapters/03_methods.tex` | Tiền xử lý, ERP, mô hình, đánh giá, giải thích | Cả nhóm |
| `chapters/04_results.tex` | ERP, RQ1, RQ2, RQ3 | Cả nhóm |
| `chapters/05_discussion.tex` | Thảo luận, hạn chế | R1 và cả nhóm |
| `chapters/06_conclusion.tex` | Trả lời RQ1 đến RQ3 | R1 |
| `chapters/07_contributions.tex` | Vai trò và % đóng góp (tổng 100%) | R1 |
| `chapters/08_availability.tex` | Code, dữ liệu, commit | R1 |
| `references.bib` | Tài liệu tham khảo | Cả nhóm |

## Cách điền

- Viết đè lên `\fillpara{...}` hoặc `\fillin{...}` bằng nội dung thật.
- Kiểm tra còn gì chưa điền: `grep -rn "fillin\|fillpara" reports/latex`.
- Bảng RQ1, RQ2 và bảng enrichment lấy từ `results/metrics/`, `results/rq2_wilcoxon_holm*.csv` và `results/explain/`. Chạy lại phân tích thì cập nhật lại các bảng này.
- Report giữ ngắn (khoảng 15 trang, class `article`). Chi tiết đã bỏ ra và câu trả lời cho phần hỏi đáp nằm ở `reports/qa_backup.md`.
- Số liệu lấy từ `results/` và các file nháp trong `reports/`, không gõ từ trí nhớ.
- Trích dẫn bằng `\citep{key}` hoặc `\citet{key}`; bibtex chỉ in tài liệu được trích.
