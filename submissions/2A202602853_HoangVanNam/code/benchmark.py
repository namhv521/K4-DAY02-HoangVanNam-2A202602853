"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations
import copy
import platform
import time
import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.

    TODO:
      - chạy warmup lần đầu rồi bỏ
      - với mỗi lần đo: sync(); t0 = time.perf_counter(); fn(); sync(); lấy hiệu * 1000
      - trả về {"p50": ..., "p95": ..., "p99": ..., "mean": ..., "n": iters}
    Gợi ý: dùng numpy.percentile hoặc torch.quantile.
    """
    if warmup < 10 or iters < 50:
        raise ValueError('At least 10 warmup and 50 timed iterations required')
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(iters):
        if sync:
            sync()
        start = time.perf_counter()
        fn()
        if sync:
            sync()
        timings.append((time.perf_counter() - start) * 1000)
    return dict(zip(('p50', 'p95', 'p99'), map(float, np.percentile(timings, [50, 95, 99]))),
                mean=float(np.mean(timings)), n=iters)


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}

    TODO:
      - model.eval(), torch.inference_mode()
      - dtype: "fp32" | "amp" (autocast) | "fp16" (model.half())
      - gọi bench(...) với sync phù hợp; lấy tên GPU bằng torch.cuda.get_device_name
      - Nhớ: ở batch 1, AMP có thể CHẬM hơn FP32 (slide trang 73): đo thật, đừng giả định
    """
    return _latency(model, batch_size, img_size, dtype, device, warmup, iters, 1)


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view: xấp xỉ K lần một lượt chạy (slide trang 63). TODO: đo thật, so với K * p50."""
    if k_views < 1:
        raise ValueError('Positive number of views required')
    return _latency(model, k_views=k_views, **kw)


def _latency(model, batch_size=1, img_size=224, dtype='fp32', device='cuda',
             warmup=10, iters=100, k_views=1):
    device = torch.device(device)
    if dtype not in ('fp32', 'amp', 'fp16') or batch_size < 1 or img_size < 1:
        raise ValueError('Invalid dtype/batch/image size')
    candidate = copy.deepcopy(model).to(device).eval()
    candidate = candidate.half() if dtype == 'fp16' else candidate.float()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device,
                    dtype=torch.float16 if dtype == 'fp16' else torch.float32)
    def forward():
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=dtype == 'amp'):
            for _ in range(k_views):
                candidate(x)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else None
    result = bench(forward, warmup, iters, sync)
    result.update(gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU ' + platform.processor(),
                  dtype=dtype, batch=batch_size, img_size=img_size, k_views=k_views,
                  images_per_s=batch_size / (result['p50'] / 1000), torch=str(torch.__version__),
                  preprocessing=False, bn_fused=False)
    return result
