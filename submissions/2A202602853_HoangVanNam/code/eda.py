"""Inspect image metadata in its original mode, without fitting preprocessing."""
from collections import Counter
from pathlib import Path
from PIL import Image


def scan_images(filenames, images_dir):
    sizes, modes, channels = Counter(), Counter(), Counter()
    for filename in filenames:
        with Image.open(Path(images_dir) / filename) as image:
            sizes[f'{image.width}x{image.height}'] += 1
            modes[image.mode] += 1
            channels[str(len(image.getbands()))] += 1
            image.verify()
    return {'sizes': dict(sizes), 'modes': dict(modes), 'channels': dict(channels),
            'verified_images': sum(sizes.values())}
