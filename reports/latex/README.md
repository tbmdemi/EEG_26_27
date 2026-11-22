# Báo cáo môn học (LaTeX)

Khung báo cáo tiếng Anh cho môn Neuroimaging: EEG. Mọi chỗ cần viết là ô đỏ `TODO` hoặc chữ đỏ `[...]`, kèm người phụ trách (R1 đến R5) và file nháp để lấy nội dung.

## Compile

Cần XeLaTeX (dùng `fontspec` để hiện tên tiếng Việt và µV), không dùng pdfLaTeX.

- Máy local: `tectonic main.tex` hoặc `latexmk main.tex` (file `latexmkrc` đã chọn XeLaTeX).
- Overleaf: tải cả folder lên, Menu > Compiler chọn XeLaTeX. Tải các hình cần dùng vào thư mục `figures/` rồi sửa `\figdir` trong `main.tex` thành `figures`.

Hình được lấy từ `figures/` của repo. Hình chưa có sẽ hiện khung đỏ `Missing figure`.

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
| `appendices/` | Bảng 42 subject, hình bổ sung | R2, R3, R4 |
| `references.bib` | Tài liệu tham khảo | Cả nhóm |

## Cách điền

- Viết đè lên `\fillpara{...}` hoặc `\fillin{...}` bằng nội dung thật.
- Kiểm tra còn gì chưa điền: `grep -rn "fillin\|fillpara" reports/latex`.
- Số liệu lấy từ `results/` và các file nháp trong `reports/`, không gõ từ trí nhớ.
- Trích dẫn bằng `\citep{key}` hoặc `\citet{key}`. Khi đã trích đủ thì bỏ `\nocite{*}` trong `main.tex`.
