# Q&A dự phòng cho buổi thuyết trình

Report đã rút gọn còn khoảng 15 trang. File này giữ các chi tiết đã bỏ khỏi report và câu trả lời ngắn cho các câu hỏi dễ gặp. Mọi số liệu lấy từ `results/` (data v2, 35 subject) và đã đối chiếu ngày 2026-11-22. Cột "Ai trả lời" chỉ là gợi ý, ai cũng nên đọc hết vì cả nhóm đều bị hỏi.

## 1. Phần đã bỏ khỏi report và nơi xem lại

| Nội dung | Xem ở đâu |
|---|---|
| Bảng 42 subject (số trial đúng, số epoch giữ, lý do loại, cờ) | `reports/r2_subject_epoch_table.md`, `results/preproc_log.csv` |
| Chi tiết tiền xử lý (ICA, kênh hỏng, chỗ nối recording, QC) | `reports/report_template.md` mục 3.1, `reports/r2_qc_and_handoff.md` |
| Bảng hành vi so với bài báo, hình hành vi | `results/erp/behaviour_vs_paper.csv`, `figures/R3_behaviour_comparison.png` |
| Difference wave, topomap 300/400/500/600 ms, ERP 3 kênh | `figures/F2_difference_wave.png`, `F2_topomap_timecourse.png`, `F2_erp_grand_average.png` |
| Hình pattern LDA (F5c), IG within-subject, IG của task đối chứng | `figures/F5c_lda_patterns.png`, `F5b_eegnet_saliency_within*.png`, `F5b_eegnet_saliency_cross_{dis_std,tgt_dis}.png` |
| Kết quả classical trên task đối chứng, EEGNet so với Shallow | mục 3 và 4 bên dưới; `reports/r4_evaluation_review.md`, `reports/methods_dl_draft.md` |
| Chi tiết phương pháp giải thích mô hình | `reports/explain_dl_draft.md` |

## 2. Dữ liệu và tiền xử lý (R2, R3)

**Vì sao chỉ giữ trial đúng?** Đề bài yêu cầu. Target đúng là có bấm trong 100 đến 1400 ms (200 ms kích thích + ISI ngắn nhất 1200 ms, theo bài báo); standard, distractor đúng là không bấm. Kiểm chứng: sub-02 có false alarm với distractor 98.8%, đúng số bài báo.

**Loại ai, vì sao?** 42 người. Loại trước 3 người hiểu sai nhiệm vụ (sub-02, 14, 17, theo bài báo). Loại thêm 4 người giữ dưới 50% target sau khi loại epoch nhiễu (sub-08, 23, 35, 41). Không ai bị loại vì dưới 40 target đúng. Còn 35 người, 20.600 epoch (15.744 standard, 2.477 target, 2.379 distractor).

**Hành vi có khớp bài báo không?** Khớp gần như tuyệt đối với Table 5 (cùng 39 người): hit rate 94.37% so với 94.40%, false alarm distractor 9.54% so với 9.52%, standard 0.24% so với 0.24%, RT 542.13 ms so với 542.13 ms. Nhóm 35 người: RT 543.8 (SD 61.3) ms.

**Giữ được bao nhiêu epoch?** 90.5% trong 22.751 trial đúng (standard 90.1%, target 92.0%, distractor 91.9%). Trung vị giữ target 97.4%, thấp nhất 53.7% (sub-10).

**Kênh hỏng, ICA?** 5 người có kênh hỏng (tổng 10 kênh, nội suy spherical spline). ICA giữ 33.1 (SD 14.0) component, loại trung bình 7.5 (SD 4.0), từ 0 đến 15 component mỗi người.

**Vì sao không lọc notch 50 Hz?** 50 Hz nằm ngoài dải 0.1 đến 40 Hz, bộ lọc thông dải đã loại.

**ICA fit trên cả recording có rò rỉ dữ liệu không?** ICA không dùng nhãn nên không rò nhãn, nhưng có nhìn thấy dữ liệu của trial test. Nhóm ghi đây là hạn chế (protocol mục 1).

**Có subject nào còn nhiễu?** QC bằng mắt thấy nhiễu kênh còn lại ở sub-10 (OI1h) và sub-13 (AFF1h). R2 đã tạo bản candidate nội suy hai kênh này nhưng nhóm chưa duyệt, nên mọi kết quả dùng bản v2 đã phát hành.

**Cờ khác?** Hit rate dưới 80%: sub-01 (0.77), sub-07 (0.60), sub-25 (0.78). Số kích thích khác 660: sub-24 (659), sub-37 (659), sub-38 (775); sub-13 có lỗi ghi được báo cáo nhưng file vẫn có 660. Recording nối nhiều file: sub-37, 38 (đánh dấu BAD 0.5 s quanh chỗ nối). Vị trí điện cực từ file .bvef thay cho CapTrak: sub-12, 24. Các cờ chỉ để xem lại, không loại.

## 3. ERP (R3)

- Đỉnh grand-average target tại Pz: 484 ms, 9.46 µV (SEM 0.50). Distractor đỉnh muộn hơn, 504 ms.
- Tại 500 ms: target 9.38, standard 0.76, distractor 7.11 µV; hiệu target trừ standard 8.61 (SEM 0.48) µV.
- Đỉnh P300 theo từng người (Pz, cực đại dương trong 300 đến 600 ms; PM duyệt 2026-11-22): target 10.34 [9.03, 12.60] µV, 484 [448, 562] ms; distractor 7.91 [7.02, 9.27] µV, 504 [440, 540] ms. Có 5 target và 1 distractor bị gắn cờ (gần biên cửa sổ hoặc không phải cực đại rõ); sub-38 chạm 600 ms. Không loại thêm ai.
- **P300 muộn vì sao?** Ba kích thích trông rất giống nhau nên phân biệt khó, mà latency P3b tăng theo độ khó (Polich 2007).

## 4. Phân loại (R4, R5)

**Vì sao balanced accuracy, không dùng accuracy?** Khoảng 6 standard mỗi target, đoán toàn standard đã được 86% accuracy.

**Cửa sổ early vì sao 450 ms?** RT trung bình bài báo 542 (SD 66) ms, trừ 1.5 SD là khoảng 443 ms. Trong nhóm 35 người: RT trung vị 536 ms, 10% dưới 412 ms, 25% dưới 461 ms. Vậy early kết thúc trước phần lớn lần bấm, nhưng không phải tất cả.

**Within-subject chia thế nào?** 5 đoạn liên tiếp bằng nhau theo thời gian, không xáo trộn, để trial cạnh nhau không rơi vào cả train và test. Không dùng 5 block theo lần nghỉ S10 vì block cuối chỉ có khoảng 40 đến 50 trial và 0 đến 4 target (sub-13 không có target), fold sẽ không tính được.

**Cross-subject chia thế nào?** GroupKFold 5 fold theo người (7 test, 28 train), lưu cố định ở `results/cv_cross_subject_folds.csv`, mọi mô hình dùng chung. Kênh được sắp theo tên vì sub-13 và sub-42 lưu kênh theo thứ tự khác.

**Mất cân bằng lớp xử lý sao?** Prior bằng nhau (LDA) hoặc class weight (logistic regression, mạng DL). Không oversample.

**Có tune hyperparameter không?** Không. Dùng giá trị trong bài báo gốc cho mọi mô hình, để so sánh công bằng. Có thể làm DL thiệt hơn một chút.

**Vì sao xdawn_rg AUC cao hơn EEGNet nhưng balanced accuracy bằng nhau (within)?** AUC đo khả năng xếp hạng trial, không phụ thuộc ngưỡng; balanced accuracy phụ thuộc ngưỡng 0.5. xdawn_rg xếp hạng tốt hơn chút, nhưng sau khi cắt ngưỡng thì như nhau.

**Vì sao EEGNet hơn ở cross-subject?** Gộp 28 người (khoảng 14.500 trial train) giúp EEGNet học đặc trưng ổn định giữa người. Tangent space của xdawn_rg đặt tâm ở trung bình các người train, không thích nghi với hiệp phương sai và latency P300 của người mới (RT trung bình từng người từ 418 đến 686 ms).

**Vì sao ShallowConvNet kém?** Nó được thiết kế cho đặc trưng công suất dải tần (bình phương rồi log), làm mất dấu của sóng ERP chậm như P300, và có 204.402 tham số (gấp khoảng 60 lần EEGNet), nên cần nhiều dữ liệu. Within/early gần ngẫu nhiên ở 10 người (sub-01, 06, 07, 10, 21, 24, 25, 28, 32, 38).

**EEGNet so với Shallow (Wilcoxon, Holm):** within full +0.045, EEGNet hơn ở 34/35 người, p 7e-10; within early +0.116, 32/35, p 4.4e-8; cross full +0.008, 21/35, p 0.081 (không khác); cross early +0.039, 28/35, p 1.9e-5.

**Người khó nhất?** Within: sub-07 (chỉ còn 31 target), thấp nhất ở mọi mô hình cửa sổ full. Cross: sub-28 với classical, sub-13 với EEGNet.

**Vì sao F1 target thấp hơn balanced accuracy?** Với 6 standard mỗi target, chỉ một tỉ lệ nhỏ false positive cũng làm precision của target giảm.

**Thống kê RQ2:** đơn vị là người (within lấy trung bình 5 fold), không dùng fold vì các fold của cùng một người không độc lập. Wilcoxon hai phía, Holm cho 4 phép so mỗi metric, effect size rank-biserial.

**Classical trên task đối chứng (bal. acc. trung vị, within full / early, cross full / early):**

| Task | Mô hình | Within full | Within early | Cross full | Cross early |
|---|---|---|---|---|---|
| dis_std | lda_tw | 0.850 | 0.814 | 0.818 | 0.735 |
| dis_std | xdawn_rg | 0.857 | 0.806 | 0.830 | 0.757 |
| tgt_dis | lda_tw | 0.747 | 0.627 | 0.714 | 0.605 |
| tgt_dis | xdawn_rg | 0.784 | 0.621 | 0.715 | 0.620 |

EEGNet: dis_std 0.866 / 0.824 / 0.839 / 0.783; tgt_dis 0.684 / 0.549 / 0.723 / 0.620. Within, xdawn_rg phân biệt target với distractor tốt hơn EEGNet (0.784 so với 0.684).

**Kết quả có tái lập được không?** Ngày 2026-11-22 chạy lại toàn bộ 24 tổ hợp classical: khớp số của R4 đến 4 chữ số. 6 tổ hợp xdawn_rg cross chạy trên Kaggle CPU vì cần hơn 9 GB RAM. DL: RTX 4060 Laptop, 4.6 giờ; task đối chứng của EEGNet: Kaggle T4.

## 5. Giải thích mô hình, RQ3 (R1)

**Vì sao chỉ giải thích EEGNet?** Là DL tốt nhất ở mọi setting và window (DL-best của RQ2). Lớp depthwise của EEGNet cho spatial filter đọc được như bản đồ da đầu. Shallow không được giải thích; Conformer (tuỳ chọn) không làm.

**Vì sao dùng pattern thay cho weight?** Weight của mô hình tuyến tính có thể lớn ở kênh chỉ dùng để khử nhiễu, nên không đọc như bản đồ da đầu được. Pattern a = C w / (w^T C w) cho biết tín hiệu thật sự nằm ở đâu (Haufe 2014).

**Integrated Gradients làm gì?** Tích phân gradient của hiệu logit (target trừ standard) theo đường thẳng từ baseline (trung bình train của từng kênh) tới trial, 64 bước. Tổng attribution khớp thay đổi đầu ra với sai số dưới 0.1%. Tính trên trial target được đoán đúng (cross full: 2.193 trên 2.477). Dùng trị tuyệt đối vì attribution có dấu của các kênh cạnh nhau hay ngược dấu nhau.

**Có chắc giải thích đúng mô hình đã báo cáo?** Có. Trước khi giải thích, tính lại quyết định trên tập test và so với dự đoán đã lưu: sai lệch lớn nhất của hiệu logit là 0.00011.

**Spatial pattern (cross):** filter quan trọng nhất của mỗi fold ở cửa sổ full giống bản đồ P300, |r| từ 0.61 đến 0.91, trung vị 0.83, đỉnh ở CCP2h, PPO1h, PPO2h. 9/10 filter hạng 1 và 2 có |r| từ 0.5 trở lên. Bỏ một filter làm AUC giảm tối đa 0.018, tức thông tin trải trên nhiều filter dư thừa. Độ quan trọng không tương quan với độ giống P300 trên toàn bộ 80 filter (Spearman rho 0.19, p 0.09). Cửa sổ early: 3 fold dùng pattern ở Pz hoặc CP1 (|r| từ 0.90 trở lên), fold 0 dùng pattern chẩm phải (|r| 0.03), fold 3 dùng trán trung tâm trái (|r| 0.53).

**IG theo thời gian:** đỉnh 495 ms (cross), 511 ms (within), trước RT trung vị. IG có dấu của vùng P300 đỉnh 472 và 456 ms, gần đỉnh P3b. Khoảng nửa attribution nằm sau 450 ms (0.53 cross, 0.51 within).

**Enrichment, vùng P300 / vùng vận động (full, trung vị 35 người):**

| Task | Setting | 0-250 ms | 250-450 ms | 450-600 ms | 600-800 ms |
|---|---|---|---|---|---|
| tgt_std | cross | 1.07 / 1.25 | 1.38 / 1.30 | 1.27 / 1.15 | 1.26 / 0.99 |
| dis_std | cross | 1.10 / 0.76 | 1.16 / 1.01 | 1.30 / 0.96 | 1.15 / 1.01 |
| tgt_dis | cross | 0.87 / 1.62 | 1.08 / 1.27 | 1.27 / 1.15 | 1.09 / 1.14 |
| tgt_std | within | 0.93 / 1.05 | 1.13 / 1.02 | 1.23 / 1.07 | 1.12 / 1.00 |
| dis_std | within | 0.95 / 0.97 | 1.03 / 0.96 | 1.13 / 1.01 | 1.01 / 0.96 |
| tgt_dis | within | 0.98 / 1.03 | 0.96 / 1.03 | 1.14 / 1.05 | 0.98 / 1.02 |

Sau 450 ms (cross): 1.20 so với 1.05, p 0.002; within 1.19 so với 1.00, p 0.010. Cửa sổ early, 250 đến 450 ms: 1.35 so với 1.23, p 0.003 (cross).

**Vì sao vùng vận động cao ở 0 đến 250 ms?** Chủ yếu do C1 và FCC3h. Attribution này xuất hiện quá sớm so với lần bấm nên khó là vận động thật. Có thể là thành phần ERP trung tâm sớm, hoặc chuẩn bị vận động ở các trial bấm nhanh. Lưu ý thêm: target hơi to hơn (1.22 so với 1.07 độ góc nhìn) nên có thể có khác biệt thị giác sớm.

**Pattern LDA:** gần như giống hệt giữa các fold (tương quan từ 0.98 trở lên). 71% độ lớn pattern nằm sau 450 ms. Tương quan với bản đồ P300: 0.92 (250 đến 450 ms), 0.99 (450 đến 600 ms), 0.95 (600 đến 800 ms). Vùng P300 đạt 0.58 µV ở 500 đến 550 ms. Vùng vận động thấp hơn kênh trung bình ở mọi bin (0.54 đến 0.70). Sau 450 ms: C3 0.14 µV, Pz 0.55 µV. Fit lại khớp đúng kết quả R4 (bal. acc. 0.8605 full, 0.7910 early).

**Task đối chứng nói gì?** dis_std không có bấm nút mà vẫn đạt 0.84 đến 0.87, nên tín hiệu nhận thức (P300/P3a) đủ để phân loại. Khi có bấm (tgt_std, tgt_dis), vùng vận động vượt mức nền của dis_std trong 250 đến 600 ms. Ở tgt_dis (hai lớp đều hiếm, P300 khác ít), vùng vận động thành vùng mạnh nhất: đỉnh 456 ms, 1.71 lần kênh trung bình, kênh mạnh nhất FCC3h, FC1, FCC1h, C1 (bán cầu trái, đối bên tay phải). Ở tgt_std, vùng vận động đỉnh 464 ms (1.31) nhưng vùng P300 vẫn bằng hoặc cao hơn (472 ms, 1.37) và còn cao sau lần bấm (600 đến 800 ms: 1.26 so với 0.99).

**Vậy kết luận RQ3?** Quyết định của EEGNet chủ yếu dựa vào P3b; có thêm phần nhỏ từ chuẩn bị vận động (lateralised readiness potential) bắt đầu trước lần bấm. Vì nó bắt đầu trước lần bấm nên cửa sổ early chỉ giảm chứ không loại hết được.

**Hạn chế của phần giải thích:** ROI và bin thời gian chọn trước, kiểm định chỉ mô tả, không hiệu chỉnh. IG không phải activation pattern (kênh khử nhiễu cũng nhận attribution). Bỏ một filter đánh giá thấp vai trò của filter dư thừa. Chỉ EEGNet được train trên task đối chứng.
