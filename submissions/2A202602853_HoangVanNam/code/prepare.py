"""GUIDE 1.2-1.3: real-data EDA and a CPU-friendly pipeline sanity check.

Run from the repository root: python code/prepare.py
This checks train images only for overfitting; it never evaluates test predictions.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import timm

import dataset
import model as models
import train
from eda import scan_images
from preflight import overfit_batch

PAPER_COUNTS = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]


def gallery(frame, images_dir, path, transform=None, samples=3):
    fig, axes = plt.subplots(9, samples, figsize=(3 * samples, 24))
    for label in range(9):
        rows = frame[frame.Label == label].head(samples)
        for ax, (_, row) in zip(axes[label], rows.iterrows()):
            with Image.open(Path(images_dir) / row.Filename) as image:
                image = image.convert('RGB')
            if transform is not None:
                image = transform(image)
                image = (image * torch.tensor(dataset.IMAGENET_STD)[:, None, None]
                         + torch.tensor(dataset.IMAGENET_MEAN)[:, None, None]).clamp(0, 1)
                image = image.permute(1, 2, 0).numpy()
            ax.imshow(image)
            ax.set_title(f'{dataset.CLASS_NAMES[label]}\n{row.Filename}', fontsize=8)
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def eda(images_dir, labels_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = dataset.load_split(labels_dir)
    checks = dataset.check_split(*splits, images_dir)
    combined = pd.concat(splits, ignore_index=True)
    checks['manifest_label_mismatches'] = dataset.check_manifest(splits, labels_dir)
    counts = pd.DataFrame({name: frame.Label.value_counts().reindex(range(9), fill_value=0)
                           for name, frame in zip(('train', 'val', 'test'), splits)})
    counts['total'] = counts.sum(axis=1)
    counts['paper'] = PAPER_COUNTS
    counts['difference'] = counts.total - counts.paper
    counts.insert(0, 'class', dataset.CLASS_NAMES)
    counts.to_csv(out / 'class_counts.csv', index_label='label')
    image_stats = scan_images(combined.Filename, images_dir)
    checks.update({**image_stats,
                   'imbalance_ratio': float(counts.total.max() / counts.total.min()),
                   'train_imbalance_ratio': float(counts.train.max() / counts.train.min()),
                   'paper_counts_match': bool((counts.difference == 0).all())})
    (out / 'split_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    ax = counts.set_index('class')[['train', 'val', 'test']].plot.bar(figsize=(12, 5))
    ax.set(xlabel='Class', ylabel='Images', title='DeepWeeds fold 0: fixed splits')
    ax.figure.tight_layout()
    ax.figure.savefig(out / 'class_distribution.png', dpi=150)
    plt.close(ax.figure)
    gallery(splits[0], images_dir, out / 'class_samples.png')
    # No normalization statistics are fitted on val/test: use fixed ImageNet constants.
    lines = ['# EDA — DeepWeeds fold 0', '',
             '| Class | Train | Val | Test | Total | Paper | Difference |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for _, row in counts.iterrows():
        lines.append('| ' + ' | '.join(str(row[c]) for c in counts.columns) + ' |')
    lines += ['', f'Split sizes: {checks["n"]}. Union: {checks["union"]}; overlaps: {checks["overlap"]}; missing: 0.',
              f'All {len(combined)} images passed PIL verification. Sizes: {checks["sizes"]}; modes: {checks["modes"]}; channels: {checks["channels"]}.',
              f'Largest/smallest class: {checks["imbalance_ratio"]:.4f} overall; {checks["train_imbalance_ratio"]:.4f} on train.',
              f'Counts match the supplied Table 1 reference: {checks["paper_counts_match"]}.', '',
              f'Original split vs labels.csv label differences: {checks["manifest_label_mismatches"]}. Keep the original split labels unchanged (S1).', '',
              '![Distribution](class_distribution.png)', '',
              'Three original TRAIN images per class; filenames and class labels are shown:', '',
              '![Train samples](class_samples.png)', '',
              'Input recipe: RGB, train RandomResizedCrop(224) + horizontal flip; val/test Resize(256) + CenterCrop(224).',
              'Use the pretrained weight mean/std (ImageNet defaults for the baseline). Never estimate normalization from test.']
    (out / 'eda.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return checks


def pipeline_check(images_dir, labels_dir, out_dir, backbone='resnet18', img_size=64, max_steps=100):
    """Overfit 9 real train images with a fresh scratch model; not a baseline experiment."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.set_seed(0)
    train_frame, _, _ = dataset.load_split(labels_dir)
    frame = train_frame.groupby('Label', sort=True).head(1).sort_values('Label')
    fixed = dataset.DeepWeedsDataset(frame, images_dir, dataset.build_transforms(False, img_size))
    x = torch.stack([fixed[i][0] for i in range(len(fixed))])
    y = torch.tensor(frame.Label.to_numpy(), dtype=torch.long)
    network = models.build_model(backbone, pretrained=False, init='scratch')
    criterion = torch.nn.CrossEntropyLoss()
    network.eval()
    with torch.no_grad():
        initial_loss = float(criterion(network(x), y))
    if abs(initial_loss - math.log(9)) > .5:
        raise AssertionError(f'Initial CE {initial_loss} far from ln(9)')
    overfit = overfit_batch(network, x, y, max_steps=max_steps)
    history = overfit['history']
    final_loss, accuracy = overfit['eval_loss'], overfit['accuracy']
    result = {'seed': 0, 'backbone': backbone, 'init': 'scratch', 'img_size': img_size,
              'device': 'cpu', 'samples': frame.Filename.tolist(), 'labels': frame.Label.tolist(),
              'initial_loss': initial_loss, 'expected_initial_loss': math.log(9),
              'steps': overfit['steps'], 'final_loss': final_loss, 'accuracy': accuracy,
              'passed': final_loss < .05 and accuracy == 1.,
              'note': 'Sanity check only; no pretrained baseline or validation/test experiment.'}
    pd.DataFrame(history).to_csv(out / 'overfit_history.csv', index=False)
    (out / 'pipeline_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    gallery(train_frame, images_dir, out / 'augmented_samples.png', dataset.build_transforms(True))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h['step'] for h in history], [h['train_loss'] for h in history], label='train mode')
    ax.plot([h['step'] for h in history], [h['eval_loss'] for h in history], label='eval mode')
    ax.set(xlabel='Optimization step', ylabel='CE loss', title='Overfit 9 real TRAIN images')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'overfit_curve.png', dpi=150)
    plt.close(fig)
    if not result['passed']:
        raise AssertionError(f'Batch did not overfit: {result}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-dir', default='data/images')
    parser.add_argument('--labels-dir', default='data/labels')
    parser.add_argument('--out-dir', default='artifacts/preparation')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    checks = eda(args.images_dir, args.labels_dir, args.out_dir)
    pipeline = pipeline_check(args.images_dir, args.labels_dir, args.out_dir)
    baseline = asdict(train.Config())
    metadata = {'baseline': baseline, 'python': platform.python_version(), 'torch': torch.__version__,
                'timm': timm.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
                'cuda_available': torch.cuda.is_available()}
    (Path(args.out_dir) / 'baseline.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'eda': checks, 'pipeline': pipeline}, indent=2))


if __name__ == '__main__':
    main()
