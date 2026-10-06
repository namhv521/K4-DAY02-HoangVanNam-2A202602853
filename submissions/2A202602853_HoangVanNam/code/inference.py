"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Cài đặt hoàn chỉnh cho Colab.
Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm phải chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện bạn nên giữ:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    Thực hiện: model.eval(), torch.inference_mode(), (tuỳ chọn) autocast. Trả về numpy.
    """
    model.eval()
    names, labels, outputs = [], [], []
    with torch.inference_mode():
        for x, y, filenames in loader:
            x = x.to(device)
            logits = model(view(x) if view else x)
            names.extend(filenames)
            labels.append(np.asarray(y))
            outputs.append(logits.float().cpu().numpy())
    if not outputs:
        raise ValueError('Empty loader')
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite logits')
    return names, np.concatenate(labels), result


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W). Thực hiện: dùng torch.flip trên chiều rộng (slide trang 75)."""
    return torch.flip(x, [-1])


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật. Trả về list các batch. Đã cài đặt."""
    h, w = x.shape[-2:]
    if not 0 < crop <= min(h, w):
        raise ValueError('Crop must fit input')
    positions = [(0, 0), (0, w-crop), (h-crop, 0), (h-crop, w-crop), ((h-crop)//2, (w-crop)//2)]
    return [x[..., top:top+crop, left:left+crop] for top, left in positions]


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch. Đã cài đặt.

    Lưu ý: model phải chấp nhận ảnh khác kích thước lúc train (CNN có global pooling thì được;
    ViT/Swin cần xử lý riêng vị trí/cửa sổ). Ghi rõ giới hạn bạn gặp.
    """
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError('Positive sizes required')
    return [F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False) for s in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Slide chưa kết luận cách nào luôn tốt hơn: chọn một và ghi rõ, hoặc so sánh cả hai (I03).
    Thực hiện: trả về xác suất (N, 9) đã chuẩn hoá.
    """
    if not logits_per_view or len({np.asarray(z).shape for z in logits_per_view}) != 1:
        raise ValueError('Non-empty, aligned views required')
    if space == 'prob':
        return np.mean([apply_temperature(z, 1.) for z in logits_per_view], axis=0)
    if space == 'logit':
        return apply_temperature(np.mean(logits_per_view, axis=0), 1.)
    raise ValueError('space must be prob or logit')


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed). Đã cài đặt.

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    if not list_of_probs or len({np.asarray(p).shape for p in list_of_probs}) != 1:
        raise ValueError('Non-empty, aligned predictions required')
    for p in list_of_probs:
        if not np.isfinite(p).all() or (np.asarray(p) < 0).any() or not np.allclose(np.sum(p, axis=1), 1):
            raise ValueError('Normalized probabilities required')
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    Thực hiện: tối ưu hoá một tham số (LBFGS trên log T, hoặc tìm lưới thô rồi tinh).
    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    z, y = np.asarray(val_logits, dtype=np.float64), np.asarray(val_labels)
    if z.ndim != 2 or len(z) == 0 or y.shape != (len(z),) or not np.isfinite(z).all():
        raise ValueError('Invalid validation logits/labels')
    if not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= z.shape[1]).any():
        raise ValueError('Invalid labels')
    def nll(log_t):
        scaled = z / np.exp(log_t)
        scaled -= scaled.max(axis=1, keepdims=True)
        return np.mean(np.log(np.exp(scaled).sum(axis=1)) - scaled[np.arange(len(y)), y])
    result = minimize_scalar(nll, bounds=(-5., 5.), method='bounded')
    if not result.success:
        raise RuntimeError('Temperature fit failed')
    return float(np.exp(result.x)) if result.fun < nll(0.) else 1.


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T). Đã cài đặt."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or not np.isfinite(z).all() or not np.isfinite(T) or T <= 0:
        raise ValueError('Finite logits and positive temperature required')
    z = z / T
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Thực hiện:
      - model.eval() trước
      - với từng cặp (Conv2d, BatchNorm2d) liền kề: tạo conv mới (có bias) và thay BN bằng Identity
      - kiểm tra: đầu ra trước/sau gộp lệch nhau cỡ 1e-5 trở xuống (in ra sai số lớn nhất)
    Với kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm), mục này không áp dụng; ghi rõ.
    """
    fused = copy.deepcopy(model).eval()
    # Restrict fusion to known forward-adjacent patterns, never arbitrary siblings.
    from timm.models.resnet import BasicBlock, Bottleneck, ResNet
    from timm.models._efficientnet_blocks import ConvBnAct, DepthwiseSeparableConv, InvertedResidual
    patterns = {ResNet: [('conv1', 'bn1')], BasicBlock: [('conv1', 'bn1'), ('conv2', 'bn2')],
                Bottleneck: [('conv1', 'bn1'), ('conv2', 'bn2'), ('conv3', 'bn3')],
                ConvBnAct: [('conv', 'bn1')],
                DepthwiseSeparableConv: [('conv_dw', 'bn1'), ('conv_pw', 'bn2')],
                InvertedResidual: [('conv_pw', 'bn1'), ('conv_dw', 'bn2'), ('conv_pwl', 'bn3')]}
    for module in fused.modules():
        pairs = []
        if isinstance(module, torch.nn.Sequential):
            children = list(module._modules.items())
            pairs = [(a, b) for (a, _), (b, _) in zip(children, children[1:])]
        pairs += patterns.get(type(module), [])
        for a, b in pairs:
            conv, bn = getattr(module, a), getattr(module, b)
            # timm BatchNormAct2d contains an activation: preserve that activation.
            if isinstance(conv, torch.nn.Conv2d) and isinstance(bn, torch.nn.BatchNorm2d):
                setattr(module, a, torch.nn.utils.fuse_conv_bn_eval(conv, bn))
                activation = copy.deepcopy(getattr(bn, 'act', torch.nn.Identity()))
                drop = copy.deepcopy(getattr(bn, 'drop', torch.nn.Identity()))
                setattr(module, b, torch.nn.Sequential(drop, activation))
    return fused
