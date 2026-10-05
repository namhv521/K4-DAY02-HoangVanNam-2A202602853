# Lab Day 2 hoàn chỉnh — Colab

Học viên: **Hoàng Văn Nam — 2A202602853**.

Mở `lab_day2.ipynb` trên Google Colab, chọn GPU và chạy từ trên xuống.
Notebook tự chứa toàn bộ module và `eval.py` gốc; không cần clone repo.
Kết quả/checkpoint lưu Drive; khi bị ngắt, chạy lại cùng cấu hình để resume.
Notebook thực hiện EDA, 5 backbone, ablation A–G, inference/latency, 3 seed final,
đánh giá gốc, Excel 7 sheet, báo cáo và ZIP. Chưa chạy huấn luyện local cho bản này.

Đọc [dochieu.md](../dochieu.md) để hiểu vai trò từng file và quy trình.

## Chuẩn bị đã có từ trước

Run from the repository root, using Python 3.10+:

```powershell
python -m pip install -r code/requirements.txt
python code/prepare.py
python -m unittest discover -s tests
```

`prepare.py` checks the original fold 0, creates EDA plots and metadata under
`artifacts/preparation/`, and overfits nine real train images on CPU.
Review `class_samples.png` and `augmented_samples.png` before experiments.
Read [the preparation report](../report.md) for observed counts, the upstream
label discrepancy, baseline settings and verification limits.

Use `lab_day2.ipynb` on Colab. All stages are implemented; keep the Drive output
location unchanged when resuming an existing campaign.

All future experiments use `train.run(Config(...))`, for example after preflight:

```python
from train import Config, run
run(Config(exp_id="B01", backbone="resnet50", seed=0))
```

Baseline: 224 crop, AdamW, backbone/head LR 1e-4/1e-3, decay 0.05 excluding
bias/norm, 1 epoch warmup + cosine, 12 epochs, batch 64, AMP on CUDA.
`run()` records the actual timm pretrained configuration and library versions.
Checkpoints use only val macro-F1; final test prediction export stays off by
default. Predictions use the original `eval.save_predictions` format.

`starter/` and `eval.py` remain the original course material. The completed
inference, benchmark, orchestration and export modules live in `code/`.
