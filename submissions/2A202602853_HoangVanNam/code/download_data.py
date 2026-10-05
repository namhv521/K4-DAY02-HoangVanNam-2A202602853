"""Download and verify official DeepWeeds images and fixed fold-0 labels."""
import hashlib
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

MD5 = 'b7b30f96d466fba86016aa5a26606e0f'
IMAGE_URL = 'https://zenodo.org/records/7939060/files/images.zip?download=1'
LABEL_BASE = 'https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels'


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'DeepWeeds-Lab/1.0'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as file:
                shutil.copyfileobj(response, file, length=1024*1024)
            temporary.replace(destination)
            return
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    data = Path('data')
    archive = data / 'images.zip'
    if not archive.exists():
        print('Downloading official images (~490 MB)...', flush=True)
        download(IMAGE_URL, archive)
    checksum = hashlib.md5()
    with archive.open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            checksum.update(chunk)
    if checksum.hexdigest() != MD5:
        raise ValueError('Image archive MD5 differs from the lab specification; remove the bad archive and retry.')
    images = data / 'images'
    images.mkdir(parents=True, exist_ok=True)
    seen = set()
    with zipfile.ZipFile(archive) as zip_file:
        for info in zip_file.infolist():
            if info.is_dir() or not info.filename.lower().endswith('.jpg'):
                continue
            name = Path(info.filename).name
            if name in seen:
                raise ValueError(f'Duplicate image basename in archive: {name}')
            seen.add(name)
            target = images / name
            if not target.exists() or target.stat().st_size != info.file_size:
                temporary = target.with_suffix('.part')
                with zip_file.open(info) as source, temporary.open('wb') as dest:
                    shutil.copyfileobj(source, dest)
                temporary.replace(target)
    if len(seen) != 17509:
        raise ValueError(f'Expected 17509 JPEGs, found {len(seen)}')
    for name in ('labels', 'train_subset0', 'val_subset0', 'test_subset0'):
        destination = data / 'labels' / f'{name}.csv'
        if not destination.exists():
            download(f'{LABEL_BASE}/{name}.csv', destination)
    print(f'MD5 OK; {len(seen)} images and original fold-0 CSVs ready.', flush=True)


if __name__ == '__main__':
    main()
