# Lab Day 2 — Hoàng Văn Nam (2A202602853)

Mở `code/lab_day2.ipynb` trên Google Colab, chọn GPU rồi Run all.
Notebook tự chứa code, tải DeepWeeds kiểm MD5, lưu Drive và resume theo epoch.
Nếu dừng phiên, mở lại notebook, giữ nguyên cấu hình và chạy lại; không đổi thư mục output sau khi đã mở test.

Các bước: chuẩn bị → backbone → training → inference → final/test → export.
Seed sàng lọc 0; vòng cuối 0,1,2. T00 dùng công thức nền trên backbone được chọn; B01 giữ mốc ResNet-50.
Cấu hình chính: {'epochs': 12, 'batch_size': 32, 'workers': 2}. Chọn hoàn toàn bằng validation.

`results.xlsx` có 7 sheet yêu cầu. `report.md` được sinh từ số liệu đo.
`runs/` chứa log/cấu hình/checkpoint; không đưa checkpoint hoặc dataset vào ZIP bài nộp.
`eval_out/` chứa kết quả `eval.py score/grade`. Trọng số/thư viện/phần cứng thực tế có trong từng config.json.
`dochieu.md` giải thích từng file.

Notebook : https://colab.research.google.com/drive/1MxH30W6OqrTms-vz6M9clHkfthdVC5kr#scrollTo=0aed2092e4b3
