"""Run GUIDE steps 1-5. Select on validation; seal all choices before test."""
from __future__ import annotations
import argparse
import hashlib
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from timm.data import resolve_model_data_config
import dataset
import model as models
import train
import inference as inf
from benchmark import bench
from eval import compute_metrics, save_predictions

STUDENT_ID = '2A202602853'
STUDENT_NAME = 'Hoàng Văn Nam'
DEFAULT_OUTPUT = 'submissions/2A202602853_Hoang_Van_Nam'
BACKBONES = ['resnet50', 'resnext50', 'convnext_tiny', 'deit_small', 'mobilenetv3']
ABLATIONS = [
    ('A', 'scratch', {'init': 'scratch'}), ('A', 'frozen', {'init': 'frozen'}),
    ('B', 'color', {'aug': 'color'}), ('B', 'randaug', {'aug': 'randaug'}),
    ('B', 'mixup', {'mix': 'mixup'}), ('B', 'cutmix', {'mix': 'cutmix'}),
    ('C', 'smoothing', {'loss': 'ls', 'label_smoothing': .1}),
    ('C', 'focal', {'loss': 'focal'}), ('C', 'weighted', {'loss': 'ce_weighted'}),
    ('D', 'balanced', {'sampler': 'balanced'}),
    ('E', 'same_lr', {'lr_backbone': 1e-3}),
    ('F', 'ema', {'ema_decay': .999}),
]
METHODS = ['single', 'hflip_prob', 'five_crop', 'hflip_logit', 'ensemble', 'fused', 'resize256', 'resize288', 'amp']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def select_best(rows):
    if not rows:
        raise ValueError('No validation results')
    return max(rows, key=lambda r: r['macro_f1_val'])


def base_config(output, epochs=12, batch_size=32, workers=2):
    return train.Config(epochs=epochs, batch_size=batch_size, num_workers=workers,
                        out_dir=str(Path(output) / 'runs'), pred_dir=str(Path(output) / 'predictions'),
                        curves_dir=str(Path(output) / 'curves'), resume=True)


def train_record(cfg, axis='', changes=None):
    summary = train.run(cfg)
    metadata = json.loads((train.run_dir(cfg) / 'config.json').read_text(encoding='utf-8'))
    return {'exp_id': cfg.exp_id, 'seed': cfg.seed, 'backbone': cfg.backbone,
            'axis': axis, 'changes': changes or {}, 'config': asdict(cfg),
            'weight_tag': metadata['pretrained_cfg'].get('hf_hub_id', metadata['pretrained_cfg'].get('url', 'scratch'))
                          if cfg.init != 'scratch' else 'scratch', **summary}


def load_network(cfg, device):
    model = models.build_model(cfg.backbone, pretrained=False, init='scratch', drop_rate=cfg.drop_rate).to(device).eval()
    model.load_state_dict(torch.load(train.run_dir(cfg) / 'best.pt', map_location=device, weights_only=True)['model'])
    meta = json.loads((train.run_dir(cfg) / 'config.json').read_text(encoding='utf-8'))
    return model, meta['preprocessing']


def method_views(x, method, size):
    if method == 'five_crop':
        return inf.views_multicrop(x, size)
    if method.startswith('hflip'):
        return [x, inf.view_hflip(x)]
    return [x]


def prediction(cfg, split, method, partner=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, prep = load_network(cfg, device)
    other = load_network(partner, device)[0] if method == 'ensemble' else None
    if method == 'fused':
        fused = inf.fuse_conv_bn(model).to(device)
        x = torch.randn(2, 3, cfg.img_size, cfg.img_size, device=device)
        with torch.inference_mode():
            torch.testing.assert_close(model(x), fused(x), atol=1e-4, rtol=1e-4)
        model = fused
    size = round(cfg.img_size * 256 / 224) if method == 'five_crop' else 256 if method == 'resize256' else 288 if method == 'resize288' else cfg.img_size
    frame = dataset.load_split(cfg.labels_dir)[1 if split == 'val' else 2]
    loader = dataset.make_loader(frame, cfg.images_dir,
                dataset.build_transforms(False, size, mean=prep['mean'], std=prep['std']),
                cfg.batch_size, False, num_workers=cfg.num_workers)
    names, truths, probabilities = [], [], []
    with torch.inference_mode():
        for x, y, filenames in loader:
            x = x.to(device)
            with torch.autocast(device_type=device.type, enabled=method == 'amp' and device.type == 'cuda'):
                logits = [model(v).float().cpu().numpy() for v in method_views(x, method, cfg.img_size)]
            p = inf.aggregate_views(logits, 'logit' if method == 'hflip_logit' else 'prob')
            if other is not None:
                p = inf.ensemble_probs([p, inf.apply_temperature(other(x).float().cpu().numpy(), 1.)])
            names.extend(filenames)
            truths.append(y.numpy())
            probabilities.append(p)
    return names, np.concatenate(truths), np.concatenate(probabilities)


def method_latency(cfg, method, batch, partner=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_network(cfg, device)
    other = load_network(partner, device)[0] if method == 'ensemble' else None
    if method == 'fused':
        model = inf.fuse_conv_bn(model).to(device)
    size = round(cfg.img_size * 256 / 224) if method == 'five_crop' else 256 if method == 'resize256' else 288 if method == 'resize288' else cfg.img_size
    x = torch.randn(batch, 3, size, size, device=device)
    def forward():
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=method == 'amp' and device.type == 'cuda'):
            values = [model(v).softmax(1) if method != 'hflip_logit' else model(v)
                      for v in method_views(x, method, cfg.img_size)]
            output = torch.stack(values).mean(0)
            if method == 'hflip_logit':
                output = output.softmax(1)
            if other is not None:
                output = (output + other(x).softmax(1)) / 2
            return output
    timing = bench(forward, iters=50, sync=(lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else None)
    timing.update(batch=batch, img_size=size, gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                  dtype='amp' if method == 'amp' else 'fp32', bn_fused=method == 'fused', preprocessing=False,
                  images_per_s=batch / (timing['p50'] / 1000), torch=str(torch.__version__),
                  method=method, checkpoint=str(train.run_dir(cfg) / 'best.pt'))
    return timing


def scalar_metrics(y, p):
    values = compute_metrics(y, p.argmax(1), p)
    return {k: values[k] for k in ('macro_f1', 'top1', 'ece', 'balanced_acc', 'nll')}


def run_backbones(state, base, persist):
    for i, name in enumerate(BACKBONES, 1):
        exp = f'B{i:02d}_{name}'
        if any(r['exp_id'] == exp for r in state['Backbones']):
            continue
        cfg = replace(base, exp_id=exp, backbone=name)
        record = train_record(cfg)
        record.update(latency_ms=method_latency(cfg, 'single', 1)['p50'])
        state['Backbones'].append(record)
        persist()
    return replace(base, backbone=select_best(state['Backbones'])['backbone'])


def run_training(state, base, persist):
    cases = [('T00', '', 'baseline', {})] + [(f'T{i:02d}_{name}', axis, name, changes)
                  for i, (axis, name, changes) in enumerate(ABLATIONS, 1)]
    if 'deit' not in base.backbone and 'swin' not in base.backbone and 'vit' not in base.backbone:
        cases += [('T13_resolution', 'G', 'resolution', {'img_size': 256})]
    for exp, axis, name, changes in cases:
        if any(r['exp_id'] == exp for r in state['Training']):
            continue
        state['Training'].append(train_record(replace(base, exp_id=exp, **changes), axis, changes))
        persist()
    anchor = next(r for r in state['Training'] if r['exp_id'] == 'T00')
    combined = {}
    for axis in 'ABCDEFG':
        rows = [r for r in state['Training'] if r['axis'] == axis]
        if rows:
            winner = select_best(rows)
            if winner['macro_f1_val'] > anchor['macro_f1_val']:
                combined.update(winner['changes'])
    if not any(r['exp_id'] == 'T99_combined' for r in state['Training']):
        state['Training'].append(train_record(replace(base, exp_id='T99_combined', **combined), 'combined', combined))
        persist()
    return train.Config(**select_best(state['Training'])['config'])


def run_inference(state, best, base, persist):
    for i, method in enumerate(METHODS):
        if any(r['method'] == method for r in state['Inference']):
            continue
        if method == 'amp' and not torch.cuda.is_available():
            continue
        if (method.startswith('resize') or method == 'ensemble') and (('deit' in best.backbone or 'swin' in best.backbone or 'vit' in best.backbone) and best.img_size != 224):
            continue
        if method.startswith('resize') and any(k in best.backbone for k in ('deit', 'swin', 'vit')):
            continue  # Fixed-resolution transformer weights cannot accept arbitrary sizes.
        if method == 'ensemble' and best.img_size != base.img_size:
            continue
        if method == 'fused':
            net, _ = load_network(best, 'cpu')
            if not any(isinstance(m, torch.nn.BatchNorm2d) for m in net.modules()):
                continue
        _, y, p = prediction(best, 'val', method, base)
        m = scalar_metrics(y, p)
        temperature = inf.fit_temperature(np.log(np.clip(p, 1e-12, 1)), y)
        calibrated = inf.apply_temperature(np.log(np.clip(p, 1e-12, 1)), temperature)
        latency = method_latency(best, method, 1, base)
        bulk = method_latency(best, method, 32, base)
        state['Latency'].extend([dict(exp_id=f'I{i:02d}', **latency), dict(exp_id=f'I{i:02d}', **bulk)])
        state['Inference'].append({'exp_id': f'I{i:02d}', 'method': method, 'config': asdict(best),
                                  'macro_f1_val': m['macro_f1'], 'top1_val': m['top1'], 'ece_val': m['ece'],
                                  'ece_calibrated_val': scalar_metrics(y, calibrated)['ece'], 'temperature': temperature,
                                  'checkpoint': str(train.run_dir(best) / 'best.pt'),
                                  'K': 5 if method == 'five_crop' else 2 if method.startswith('hflip') or method == 'ensemble' else 1,
                                  **latency, 'bulk_images_per_s': bulk['images_per_s']})
        persist()
    # Accuracy first; lower latency resolves exact validation ties.
    winner = max(state['Inference'], key=lambda r: (r['macro_f1_val'], -r['p95']))
    return winner['method']


def prepare_final(state, best, base, method, persist):
    if state.get('sealed'):
        return
    for seed in (0, 1, 2):
        for cfg in (replace(base, exp_id='T00', seed=seed), replace(best, exp_id='F01', seed=seed)):
            train.run(cfg)
        final = replace(best, exp_id='F01', seed=seed)
        baseline = replace(base, exp_id='T00', seed=seed)
        names, y, p = prediction(final, 'val', method, baseline)
        temperature = inf.fit_temperature(np.log(np.clip(p, 1e-12, 1)), y)
        save_predictions(Path(base.pred_dir) / f'F01_seed{seed}_val.csv', names, y,
                         inf.apply_temperature(np.log(np.clip(p, 1e-12, 1)), temperature))
        state['calibrations'][str(seed)] = temperature
        persist()
    final_zero, baseline_zero = replace(best, exp_id='F01', seed=0), replace(base, exp_id='T00', seed=0)
    if not any(r['exp_id'] == 'F01' for r in state['Latency']):
        state['Latency'].extend([dict(exp_id='F01', **method_latency(final_zero, method, batch, baseline_zero)) for batch in (1, 32)])
    state['sealed'] = {'final_config': asdict(best), 'baseline_config': asdict(base), 'method': method,
                       'seeds': [0, 1, 2], 'selection_split': 'val'}
    persist()  # Frozen BEFORE looking at any test predictions.


def run_test(state, persist):
    seal = state['sealed']
    base = train.Config(**seal['baseline_config'])
    best = train.Config(**seal['final_config'])
    for seed in seal['seeds']:
        if any(r['seed'] == seed for r in state['Final']):
            continue
        final, baseline = replace(best, exp_id='F01', seed=seed), replace(base, exp_id='T00', seed=seed)
        cache = Path(base.out_dir).parent / 'test_cache' / f'seed{seed}.npz'
        if not cache.exists():
            names, y, p = prediction(final, 'test', seal['method'], baseline)
            bn, by, bp = prediction(baseline, 'test', 'single')
            if names != bn or not np.array_equal(y, by):
                raise ValueError('Final/baseline test order differs')
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix('.tmp')
            with tmp.open('wb') as f:
                np.savez_compressed(f, names=names, y=y, p=p, baseline=bp)
            tmp.replace(cache)
        values = np.load(cache)
        names, y, p, bp = values['names'].tolist(), values['y'], values['p'], values['baseline']
        calibrated = inf.apply_temperature(np.log(np.clip(p, 1e-12, 1)), state['calibrations'][str(seed)])
        for exp, probs in [('F01', calibrated), ('F01uncal', p), ('T00', bp)]:
            path = Path(base.pred_dir) / f'{exp}_seed{seed}_test.csv'
            if not path.exists():
                save_predictions(path, names, y, probs)
        state['Final'].append({'exp_id': 'F01', 'seed': seed, 'configuration': seal,
                               'temperature': state['calibrations'][str(seed)],
                               **{k + '_test': v for k, v in scalar_metrics(y, calibrated).items()},
                               'baseline': scalar_metrics(y, bp)})
        persist()
    output = Path(base.out_dir).parent
    common = ['--test-csv', str(Path(base.labels_dir) / 'test_subset0.csv'), '--labels', str(Path(base.labels_dir) / 'labels.csv')]
    for tag in ('F01', 'T00'):
        subprocess.run([sys.executable, 'eval.py', 'score', '--pred', str(Path(base.pred_dir) / f'{tag}_seed*_test.csv'),
                        *common, '--tag', tag, '--out', str(output / 'eval_out')], check=True)
    subprocess.run([sys.executable, 'eval.py', 'grade', '--final', str(Path(base.pred_dir) / 'F01_seed*_test.csv'),
                    '--baseline', str(Path(base.pred_dir) / 'T00_seed*_test.csv'),
                    '--uncal', str(Path(base.pred_dir) / 'F01uncal_seed*_test.csv'),
                    '--final-val', str(Path(base.pred_dir) / 'F01_seed*_val.csv'),
                    '--val-csv', str(Path(base.labels_dir) / 'val_subset0.csv'),
                    '--latency-p95-ms', str(next(r['p95'] for r in state['Latency'] if r['exp_id'] == 'F01' and r['batch'] == 1)),
                    '--latency-method', 'proper',
                    *common, '--out', str(output / 'eval_out')], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'backbones', 'training', 'inference', 'final', 'export'], default='all')
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--allow-cpu', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'experiment_state.json'
    state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'student_id': STUDENT_ID, 'student_name': STUDENT_NAME,
        **{k: [] for k in ['Backbones', 'Training', 'Inference', 'Final', 'Latency']}, 'calibrations': {}}
    def persist():
        write_json(path, state)
    base = base_config(output, args.epochs, args.batch_size, args.workers)
    settings = {'epochs': args.epochs, 'batch_size': args.batch_size, 'workers': args.workers}
    if args.stage != 'export' and 'run_settings' in state and state['run_settings'] != settings:
        raise SystemExit('Settings differ from this saved campaign; restore settings or choose a new output folder.')
    if args.stage != 'export':
        state['run_settings'] = settings
        digest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
        if 'source_sha256' in state and state['source_sha256'] != digest:
            raise SystemExit('Code differs from this saved campaign. Use original code or a new output folder.')
        state['source_sha256'] = digest
    if args.stage == 'export':
        persist()
        from deliverables import export
        export(output, state)
        return
    if not torch.cuda.is_available() and not args.allow_cpu:
        persist()
        raise SystemExit('No CUDA GPU. Use Colab/Kaggle GPU, or explicitly --allow-cpu for the long CPU run.')
    if args.epochs < 10:
        raise SystemExit('Official experiments require >=10 epochs.')
    if state.get('sealed') and args.stage not in ('all', 'final'):
        raise SystemExit('Test selection is sealed. Screening is closed for this output directory.')
    persist()
    if not state.get('sealed'):
        base = run_backbones(state, base, persist)
        if args.stage == 'backbones':
            return
        best = run_training(state, base, persist)
        if args.stage == 'training':
            return
        method = run_inference(state, best, replace(base, exp_id='T00'), persist)
        if args.stage == 'inference':
            return
        prepare_final(state, best, base, method, persist)
    run_test(state, persist)
    from deliverables import export
    export(output, state)


if __name__ == '__main__':
    main()
