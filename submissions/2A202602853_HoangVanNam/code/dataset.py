"""DeepWeeds validation and shared preprocessing."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
CLASS_NAMES = ['Chinee apple', 'Lantana', 'Parkinsonia', 'Parthenium', 'Prickly acacia', 'Rubber vine', 'Siam weed', 'Snake weed', 'Negative']
IMAGENET_MEAN = (.485, .456, .406)
IMAGENET_STD = (.229, .224, .225)


def load_split(labels_dir, fold=0):
    if fold not in range(5):
        raise ValueError('fold must be 0..4')
    frames = []
    for split in ('train', 'val', 'test'):
        frame = pd.read_csv(Path(labels_dir) / f'{split}_subset{fold}.csv')
        if not {'Filename', 'Label'}.issubset(frame):
            raise ValueError(f'{split}: missing Filename/Label')
        frames.append(frame)
    return tuple(frames)


def check_split(train_df, val_df, test_df, images_dir):
    frames = dict(zip(('train', 'val', 'test'), (train_df, val_df, test_df)))
    names = {}
    for split, frame in frames.items():
        if frame.empty or frame[['Filename', 'Label']].isna().any().any():
            raise ValueError(f'{split}: empty split or missing values')
        if frame.Filename.duplicated().any():
            raise ValueError(f'{split}: duplicate filenames')
        if not frame.Label.isin(range(9)).all() or set(frame.Label) != set(range(9)):
            raise ValueError(f'{split}: invalid labels or missing classes')
        if any(Path(n).name != n or '\\' in n for n in frame.Filename):
            raise ValueError(f'{split}: Filename must be a basename')
        names[split] = set(frame.Filename)
    overlap = {f'{a}_{b}': len(names[a] & names[b]) for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    if any(overlap.values()):
        raise ValueError(f'Split overlap: {overlap}')
    union = set.union(*names.values())
    if len(union) != 17509:
        raise ValueError(f'Expected 17509 distinct files, got {len(union)}')
    missing = sorted(n for n in union if not (Path(images_dir) / n).is_file())
    if missing:
        raise ValueError(f'Missing {len(missing)} images: {missing[:5]}')
    for split, expected in (('train', .6), ('val', .2), ('test', .2)):
        if abs(len(frames[split]) / len(union) - expected) > .01:
            raise ValueError(f'{split}: ratio differs by more than 1 percentage point')
    return {'n': {s: len(f) for s, f in frames.items()},
            'per_class': {s: f.Label.value_counts().reindex(range(9), fill_value=0).sort_index().to_dict() for s, f in frames.items()},
            'overlap': overlap, 'union': len(union), 'missing': 0}


def build_transforms(train, img_size=224, aug='basic', mean=IMAGENET_MEAN, std=IMAGENET_STD):
    if train:
        ops = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
        extras = {'basic': [], 'color': [T.ColorJitter(.2, .2, .2, .05)], 'trivial': [T.TrivialAugmentWide()], 'randaug': [T.RandAugment()]}
        if aug not in extras:
            raise ValueError(f'Unknown augmentation: {aug}')
        ops += extras[aug]
    else:
        ops = [T.Resize(round(img_size * 256 / 224)), T.CenterCrop(img_size)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(mean, std)])


def check_manifest(splits, labels_dir):
    """Audit original labels; preserve the author's fixed split (README S1).

    Upstream fold 0 assigns this one image to class 0, while labels.csv assigns
    class 1. Log this known discrepancy; reject any additional disagreement.
    """
    manifest = pd.read_csv(Path(labels_dir) / 'labels.csv')
    combined = pd.concat(splits, ignore_index=True)
    if manifest.Filename.duplicated().any() or len(manifest) != 17509:
        raise ValueError('Invalid labels.csv manifest')
    reference = manifest.set_index('Filename').Label
    if set(reference.index) != set(combined.Filename):
        raise ValueError('Split filenames differ from labels.csv')
    labels = reference.loc[combined.Filename].to_numpy()
    mismatches = combined.loc[labels != combined.Label.to_numpy(), ['Filename', 'Label']].copy()
    mismatches['manifest_label'] = labels[labels != combined.Label.to_numpy()]
    records = mismatches.to_dict('records')
    known = {'Filename': '20170714-110407-3.jpg', 'Label': 0, 'manifest_label': 1}
    if any(record != known for record in records):
        raise ValueError(f'Unexpected split label disagreement: {records[:5]}')
    return records


class DeepWeedsDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.images_dir, self.transform = Path(images_dir), transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.images_dir / row.Filename) as image:
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row.Label), row.Filename


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df, images_dir, transform, batch_size, train, sampler=None, num_workers=2):
    if sampler not in (None, 'balanced') or (sampler and not train):
        raise ValueError('Sampler must be None or balanced (train only)')
    generator = torch.Generator().manual_seed(torch.initial_seed())
    sampling = None
    if sampler == 'balanced':
        weights = 1 / df.Label.map(df.Label.value_counts()).to_numpy()
        sampling = WeightedRandomSampler(weights, len(df), replacement=True, generator=generator)
    return DataLoader(DeepWeedsDataset(df, images_dir, transform), batch_size=batch_size,
                      shuffle=train and sampling is None, sampler=sampling,
                      drop_last=train and len(df) >= batch_size, pin_memory=torch.cuda.is_available(),
                      num_workers=num_workers, worker_init_fn=seed_worker, generator=generator)
