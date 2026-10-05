"""Loss and batch-mixing implementations."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LabelSmoothingCE(nn.CrossEntropyLoss):
    def __init__(self, smoothing=.1):
        super().__init__(label_smoothing=smoothing)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError('gamma must be nonnegative')
        self.gamma = gamma
        self.register_buffer('alpha', None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        log_pt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -(1 - log_pt.exp()).pow(self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def build_criterion(kind='ce', **kw):
    if kind == 'ce':
        return nn.CrossEntropyLoss()
    if kind == 'ls':
        return LabelSmoothingCE(kw.get('smoothing', .1))
    if kind == 'focal':
        return FocalLoss(kw.get('gamma', 2.), kw.get('alpha'))
    if kind == 'ce_weighted':
        if kw.get('weight') is None:
            raise ValueError('ce_weighted requires train-only class weights')
        return nn.CrossEntropyLoss(weight=kw['weight'])
    raise ValueError(f'Unknown loss: {kind}')


def class_weights(counts, beta=0.):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if (counts <= 0).any() or not 0 <= beta < 1:
        raise ValueError('Positive counts and 0 <= beta < 1 required')
    weights = 1 / counts if beta == 0 else (1 - beta) / (-torch.expm1(counts * np.log(beta)))
    return (weights / weights.mean()).float()


def mix_batch(x, y, alpha=1., mode='cutmix'):
    if alpha <= 0 or mode not in ('mixup', 'cutmix'):
        raise ValueError('Positive alpha and mode mixup/cutmix required')
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(len(x), device=x.device)
    if mode == 'mixup':
        mixed = lam * x + (1 - lam) * x[perm]
    else:
        h, w = x.shape[-2:]
        cut_w, cut_h = int(w * np.sqrt(1 - lam)), int(h * np.sqrt(1 - lam))
        cx, cy = np.random.randint(w), np.random.randint(h)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + (cut_w + 1) // 2, w)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + (cut_h + 1) // 2, h)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (w * h)
    return mixed, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    a, b, lam = targets
    return lam * criterion(logits, a) + (1 - lam) * criterion(logits, b)
