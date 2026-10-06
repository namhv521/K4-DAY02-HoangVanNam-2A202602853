# Báo cáo Lab Day 2 — Hoàng Văn Nam

Mã học viên: **2A202602853**



## 1. Tóm tắt

Đã ghi nhận 5 backbone, 4 công thức huấn luyện, 0 phương pháp suy luận và 0 seed chung kết.

Các bảng dưới đây chỉ lấy từ dữ liệu đã đo; phần chưa chạy không có số liệu.



## 2. Dữ liệu và thiết lập

# EDA — DeepWeeds fold 0

| Class | Train | Val | Test | Total | Paper | Difference |
|---|---:|---:|---:|---:|---:|---:|
| Chinee apple | 675 | 225 | 226 | 1126 | 1125 | 1 |
| Lantana | 637 | 213 | 213 | 1063 | 1064 | -1 |
| Parkinsonia | 618 | 206 | 207 | 1031 | 1031 | 0 |
| Parthenium | 613 | 204 | 205 | 1022 | 1022 | 0 |
| Prickly acacia | 637 | 212 | 213 | 1062 | 1062 | 0 |
| Rubber vine | 605 | 202 | 202 | 1009 | 1009 | 0 |
| Siam weed | 644 | 215 | 215 | 1074 | 1074 | 0 |
| Snake weed | 609 | 203 | 204 | 1016 | 1016 | 0 |
| Negative | 5463 | 1821 | 1822 | 9106 | 9106 | 0 |

Split sizes: {'train': 10501, 'val': 3501, 'test': 3507}. Union: 17509; overlaps: {'train_val': 0, 'train_test': 0, 'val_test': 0}; missing: 0.
All 17509 images passed PIL verification. Sizes: {'256x256': 17509}; modes: {'RGB': 17509}; channels: {'3': 17509}.
Largest/smallest class: 9.0248 overall; 9.0298 on train.
Counts match the supplied Table 1 reference: False.

Original split vs labels.csv label differences: [{'Filename': '20170714-110407-3.jpg', 'Label': 0, 'manifest_label': 1}]. Keep the original split labels unchanged (S1).

![Distribution](artifacts/preparation/class_distribution.png)

Three original TRAIN images per class; filenames and class labels are shown:

![Train samples](artifacts/preparation/class_samples.png)

Input recipe: RGB, train RandomResizedCrop(224) + horizontal flip; val/test Resize(256) + CenterCrop(224).
Use the pretrained weight mean/std (ImageNet defaults for the baseline). Never estimate normalization from test.




Các thí nghiệm chính dùng fold 0 cố định, chọn checkpoint bằng macro-F1 validation. T00 là công thức nền trên backbone được chọn từ Bước 1; B01_resnet50 giữ mốc kiến trúc ResNet-50. Batch mặc định 32 để phù hợp GPU T4 (giảm từ 64 gợi ý trong GUIDE); 12 epoch; AdamW, warmup 1 epoch, cosine; AMP train trên CUDA. Nhãn train không sửa theo labels.csv khi có sai lệch upstream đã ghi trong EDA.



## 3. So sánh backbone

| backbone | macro_f1_val | top1_val | params_m | gmacs | batch1_ms |
| --- | --- | --- | --- | --- | --- |
| resnet50 | 0.8244 | 0.8706 | 23.5265 | None | 6.4625 |
| resnext50 | 0.7651 | 0.8003 | 22.9983 | None | 8.3594 |
| convnext_tiny | 0.9676 | 0.9757 | 27.8270 | None | 8.0487 |
| deit_small | 0.9452 | 0.9617 | 21.6691 | None | 5.3065 |
| mobilenetv3 | 0.7565 | 0.8172 | 4.2136 | None | 6.4380 |


Backbone có macro-F1 val cao nhất được chọn. Đây là sàng lọc một seed; trọng số tiền huấn luyện có thể có công thức khác nhau nên không quy mọi chênh lệch cho riêng kiến trúc. GMAC bị để trống nếu fvcore gặp phép toán chưa hỗ trợ.



## 4. Ablation công thức

| exp_id | axis | changes | macro_f1_val | delta_vs_T00 |
| --- | --- | --- | --- | --- |
| T00 |  | {} | 0.9676 | 0.0000 |
| T01_scratch | A | {"init": "scratch"} | 0.3899 | -0.5777 |
| T02_frozen | A | {"init": "frozen"} | 0.8564 | -0.1112 |
| T03_color | B | {"aug": "color"} | 0.9684 | 0.0008 |


Mỗi dòng A–G thay một yếu tố; T99 kết hợp các thay đổi có delta dương trên từng trục. Kết hợp không mặc nhiên tốt hơn: vẫn được đo lại và so với tất cả công thức. Không kết luận ý nghĩa thống kê từ một seed sàng lọc. Label smoothing giảm độ tự tin; focal ưu tiên mẫu khó; loss có trọng số và sampler tác động khác nhau lên lớp hiếm. Đây là cơ chế lý thuyết, hiệu quả thực tế phải đọc từ bảng.



## 5. Phương pháp suy luận

Chưa có kết quả đo cho phần này.


Chưa có biểu đồ suy luận.

Latency đo 10 warmup + 50 lượt, đồng bộ CUDA trước/sau, batch 1 và 32. Không tính decode ảnh/tiền xử lý PIL hoặc I/O. Temperature khớp bằng NLL trên validation, bảo toàn argmax; giảm NLL không bảo đảm ECE cũng giảm. ViT/Swin có kích thước cố định không chạy resize tùy ý. BN fusion chỉ áp dụng kiến trúc có BatchNorm và kiểm tra tương đương.



## 6. Chung kết và test

Chưa có kết quả đo cho phần này.


Ba seed 0/1/2 cho cả F01 và T00; mean ± sample std (ddof=1). Mọi lựa chọn được khóa trước lần test đầu tiên. Test cache cho phép xuất lại báo cáo mà không chạy lại model trên test. Không dùng test để chọn hoặc sửa công thức.



## 7. Phân tích lỗi và khuyến nghị

Chưa chạy chung kết; chưa có kết luận về chất lượng test.



## 8. Hạn chế

Một fold, ba seed vòng cuối và một seed sàng lọc. Split ngẫu nhiên theo ảnh có thể lạc quan khi chuyển địa điểm/mùa. Chưa đo end-to-end trên robot; độ trễ Colab không thay thế latency thiết bị triển khai. Calibration dùng cùng val chọn công thức, có thể lạc quan; miền mới cần đánh giá riêng. Không đối chiếu trực tiếp số của bài báo 100 epoch với thiết lập 12 epoch này.



## 9. Phụ lục tái lập

Cấu hình, tag trọng số, phiên bản thư viện và phần cứng: `runs/*/seed*/config.json`. Danh sách công thức và lựa chọn khóa: `experiment_state.json`. Notebook tự chứa toàn bộ code: `code/lab_day2.ipynb`. Kết quả chấm gốc: `eval_out/`; các file dự đoán có p0…p8, argmax và nhãn thật.
