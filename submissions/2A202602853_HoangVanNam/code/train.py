"""One configurable train loop. Test inference stays off until the final stage."""
from __future__ import annotations
import argparse
import hashlib
import copy
import json
import math
import random
import sys
import time
import platform
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import get_type_hints, get_args
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import compute_metrics, save_predictions
import dataset
import model as models
import losses


@dataclass
class Config:
    exp_id: str = 'T00'
    seed: int = 0
    fold: int = 0
    backbone: str = 'resnet50'
    init: str = 'finetune'
    drop_rate: float = 0.
    img_size: int = 224
    aug: str = 'basic'
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.
    loss: str = 'ce'
    label_smoothing: float = 0.
    focal_gamma: float = 2.
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = .05
    warmup_epochs: float = 1.
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = 'data/images'
    labels_dir: str = 'data/labels'
    out_dir: str = 'runs'
    pred_dir: str = 'predictions'
    save_test_predictions: bool = False
    resume: bool = False
    curves_dir: str = 'curves'


def run_dir(cfg):
    return Path(cfg.out_dir) / cfg.exp_id / f'seed{cfg.seed}'


def pred_path(cfg, split):
    return Path(cfg.pred_dir) / f'{cfg.exp_id}_seed{cfg.seed}_{split}.csv'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_resume_config(cfg, saved):
    current = asdict(cfg)
    ignored = {'resume'}
    if any(current.get(k) != v for k, v in saved.items() if k not in ignored):
        raise ValueError('Resume configuration differs from saved experiment')


def save_checkpoint(state, path):
    temporary = Path(path).with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def build_optimizer(model, cfg):
    return torch.optim.AdamW(models.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    if cfg.epochs <= 0 or steps_per_epoch <= 0 or not 0 <= cfg.warmup_epochs < cfg.epochs:
        raise ValueError('Positive epochs/steps and 0 <= warmup_epochs < epochs required')
    total = cfg.epochs * steps_per_epoch
    warmup = round(cfg.warmup_epochs * steps_per_epoch)
    def factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1., (step - warmup) / max(1, total - warmup))
        return .5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    def __init__(self, model, decay):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0,1)')
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for name, target in self.model.state_dict().items():
            if target.is_floating_point():
                target.lerp_(source[name], 1 - self.decay)
            else:
                target.copy_(source[name])


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=None):
    model.train()
    if cfg.init == 'frozen':
        model.eval()  # includes BN/dropout in the frozen backbone
        model.get_classifier().train()
    total, n = 0., 0
    device = torch.device(device)
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        target = y
        if cfg.mix:
            x, target = losses.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == 'cuda'):
            logits = model(x)
            loss = losses.mixed_loss(criterion, logits, target) if cfg.mix else criterion(logits, target)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        scaler.scale(loss).backward()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:  # no scheduler/EMA update after an AMP-skipped step
            scheduler.step()
            if ema is not None:
                ema.update(model)
        total += float(loss.detach()) * len(y)
        n += len(y)
    if not n:
        raise ValueError('Empty training loader')
    return {'train_loss': total / n, 'lr': optimizer.param_groups[0]['lr']}


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    names, truth, outputs, total, n = [], [], [], 0., 0
    for x, y, filenames in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite evaluation loss')
        total += float(loss) * len(y)
        n += len(y)
        names.extend(filenames)
        truth.append(y.cpu().numpy())
        outputs.append(logits.float().cpu().numpy())
    if not n:
        raise ValueError('Empty evaluation loader')
    return names, np.concatenate(truth), np.concatenate(outputs), total / n


def plot_curves(history, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame.epoch, frame.train_loss, label='train')
    axes[0].plot(frame.epoch, frame.val_loss, label='val')
    axes[0].set(xlabel='Epoch', ylabel='Loss')
    axes[0].legend()
    axes[1].plot(frame.epoch, frame.macro_f1)
    axes[1].set(xlabel='Epoch', ylabel='Val macro-F1')
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(cfg):
    if Path(cfg.exp_id).name != cfg.exp_id or not cfg.exp_id:
        raise ValueError('exp_id must be a non-empty basename')
    if cfg.batch_size < 2 or cfg.epochs < 1:
        raise ValueError('batch_size >= 2 and epochs >= 1 required')
    folder = run_dir(cfg)
    if cfg.resume and folder.exists():
        saved = json.loads((folder / 'config.json').read_text(encoding='utf-8'))['config']
        validate_resume_config(cfg, saved)
        if (folder / 'summary.json').exists():
            return json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        # Before the first committed epoch, restart that epoch from the same seed.
    elif folder.exists() or pred_path(cfg, 'val').exists() or (cfg.save_test_predictions and pred_path(cfg, 'test').exists()):
        raise FileExistsError('Run/predictions already exist; use a new exp_id, do not overwrite final test evidence')
    set_seed(cfg.seed)
    splits = dataset.load_split(cfg.labels_dir, cfg.fold)
    check = dataset.check_split(*splits, cfg.images_dir)
    check['manifest_label_mismatches'] = dataset.check_manifest(splits, cfg.labels_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.build_model(cfg.backbone, init=cfg.init, drop_rate=cfg.drop_rate).to(device)
    from timm.data import resolve_model_data_config
    data_config = resolve_model_data_config(model)
    mean, std = data_config['mean'], data_config['std']
    train_loader = dataset.make_loader(splits[0], cfg.images_dir, dataset.build_transforms(True, cfg.img_size, cfg.aug, mean, std), cfg.batch_size, True, cfg.sampler, cfg.num_workers)
    val_loader = dataset.make_loader(splits[1], cfg.images_dir, dataset.build_transforms(False, cfg.img_size, mean=mean, std=std), cfg.batch_size, False, num_workers=cfg.num_workers)
    counts = splits[0].Label.value_counts().reindex(range(9)).to_numpy()
    weight = losses.class_weights(counts, cfg.class_weight_beta or 0.) if cfg.loss == 'ce_weighted' else None
    criterion = losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma, weight=weight).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and device.type == 'cuda')
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None
    folder.mkdir(parents=True, exist_ok=cfg.resume)
    metadata = {'config': asdict(cfg), 'pretrained_cfg': model.pretrained_cfg,
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                'label_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(cfg.labels_dir).glob('*.csv')},
                'preprocessing': {'mean': mean, 'std': std, 'eval_resize': round(cfg.img_size * 256 / 224), 'crop': cfg.img_size},
                'device': str(device), 'amp_active': scaler.is_enabled(), 'split_check': check,
                'torch': torch.__version__, 'python': platform.python_version(),
                'numpy': np.__version__, 'pandas': pd.__version__}
    import timm
    import torchvision
    metadata.update({'timm': timm.__version__, 'torchvision': torchvision.__version__,
                     'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                     'scheduler': 'per optimizer step, linear warmup then cosine'})
    (folder / 'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    history, best, best_epoch, start_epoch = [], -1., 0, 1
    if cfg.resume and (folder / 'last.pt').exists():
        # Only load trusted checkpoints created locally by this training code.
        state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        if ema:
            ema.model.load_state_dict(state['ema'])
        history, best, best_epoch = state['history'], state['best'], state['best_epoch']
        start_epoch = state['epoch'] + 1
        random.setstate(state['random'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        train_loader.generator.set_state(state['loader_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(start_epoch, cfg.epochs + 1):
        started = time.perf_counter()
        stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        candidate = ema.model if ema else model
        _, y, logits, val_loss = evaluate(candidate, val_loader, criterion, device)
        probs = torch.from_numpy(logits).softmax(1).numpy()
        metrics = compute_metrics(y, probs.argmax(1), probs)
        history.append({'epoch': epoch, **stats, 'val_loss': val_loss, 'macro_f1': metrics['macro_f1'], 'top1': metrics['top1'], 'seconds': time.perf_counter() - started})
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        if metrics['macro_f1'] > best:
            best, best_epoch = metrics['macro_f1'], epoch
            save_checkpoint({'model': candidate.state_dict(), 'epoch': epoch, 'macro_f1': best}, folder / 'best.pt')
        save_checkpoint({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                         'ema': ema.model.state_dict() if ema else None, 'epoch': epoch,
                         'history': history, 'best': best, 'best_epoch': best_epoch,
                         'random': random.getstate(), 'numpy_rng': np.random.get_state(),
                         'torch_rng': torch.get_rng_state(), 'loader_rng': train_loader.generator.get_state(),
                         'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}, folder / 'last.pt')
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(torch.load(folder / 'best.pt', map_location=device, weights_only=True)['model'])
    for split, frame in [('val', splits[1])] + ([('test', splits[2])] if cfg.save_test_predictions else []):
        loader = val_loader if split == 'val' else dataset.make_loader(frame, cfg.images_dir, dataset.build_transforms(False, cfg.img_size, mean=mean, std=std), cfg.batch_size, False, num_workers=cfg.num_workers)
        names, y, logits, _ = evaluate(model, loader, criterion, device)
        np.savez_compressed(folder / f'{split}_logits.npz', filenames=names, y_true=y, logits=logits)
        save_predictions(pred_path(cfg, split), names, y, torch.from_numpy(logits).softmax(1).numpy())
    plot_curves(history, Path(cfg.curves_dir) / f'{cfg.exp_id}_seed{cfg.seed}.png', f'{cfg.exp_id} {cfg.backbone} seed {cfg.seed}')
    summary = {'best_epoch': best_epoch, 'macro_f1_val': best, 'params_m': models.count_params(model), 'gmacs': None,
               'top1_val': next(h['top1'] for h in history if h['epoch'] == best_epoch),
               'gmacs_note': 'Not measured; optional fvcore count_gmacs', 'mean_epoch_seconds': np.mean([h['seconds'] for h in history])}
    try:
        summary['gmacs'] = models.count_gmacs(model, cfg.img_size)
        summary['gmacs_note'] = 'fvcore estimate; one multiply-add counted as one operation'
    except (ImportError, ValueError) as error:
        summary['gmacs_note'] = str(error)
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def parse_overrides(pairs):
    hints = get_type_hints(Config)
    result = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f'Expected KEY=VALUE: {pair}')
        key, value = pair.split('=', 1)
        if key not in hints:
            raise ValueError(f'Unknown Config field: {key}')
        args = get_args(hints[key])
        if value.lower() == 'none' and type(None) in args:
            result[key] = None
            continue
        kind = next((t for t in args if t is not type(None)), hints[key])
        if kind is bool:
            if value.lower() not in ('true', 'false'):
                raise ValueError(f'{key}: use true/false')
            result[key] = value.lower() == 'true'
        else:
            result[key] = kind(value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', nargs='*', default=[])
    args = parser.parse_args()
    print(json.dumps(run(Config(**parse_overrides(args.set))), indent=2))


if __name__ == '__main__':
    main()
