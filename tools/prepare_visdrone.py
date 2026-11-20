"""Download (if needed) and convert VisDrone2019-DET annotations to COCO JSON format.

VisDrone2019-DET has 10 foreground object categories (mapped to 0..9):
  0: pedestrian, 1: people, 2: bicycle, 3: car, 4: van,
  5: truck, 6: tricycle, 7: awning-tricycle, 8: bus, 9: motor
Ignored regions (score == 0 or category in {0, 11}) are skipped per the
standard VisDrone2019-DET evaluation protocol.
"""

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image


VISDRONE_URLS = {
    'train': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip',
    'val': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip',
}

VISDRONE_CATEGORIES = [
    {'id': 0, 'name': 'pedestrian', 'supercategory': 'person'},
    {'id': 1, 'name': 'people', 'supercategory': 'person'},
    {'id': 2, 'name': 'bicycle', 'supercategory': 'vehicle'},
    {'id': 3, 'name': 'car', 'supercategory': 'vehicle'},
    {'id': 4, 'name': 'van', 'supercategory': 'vehicle'},
    {'id': 5, 'name': 'truck', 'supercategory': 'vehicle'},
    {'id': 6, 'name': 'tricycle', 'supercategory': 'vehicle'},
    {'id': 7, 'name': 'awning-tricycle', 'supercategory': 'vehicle'},
    {'id': 8, 'name': 'bus', 'supercategory': 'vehicle'},
    {'id': 9, 'name': 'motor', 'supercategory': 'vehicle'},
]


def download_and_extract(url: str, dest_root: Path, split_folder: str) -> Path:
    split_dir = dest_root / split_folder
    if (split_dir / 'images').exists() and (split_dir / 'annotations').exists():
        print(f'[VisDrone] Found existing {split_dir}')
        return split_dir

    dest_root.mkdir(parents=True, exist_ok=True)
    zip_path = dest_root / f'{split_folder}.zip'
    if not zip_path.exists():
        print(f'[VisDrone] Downloading {url} -> {zip_path} ...')
        urllib.request.urlretrieve(url, zip_path)

    print(f'[VisDrone] Extracting {zip_path} ...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_root)

    if zip_path.exists():
        zip_path.unlink()
    return split_dir


def convert_split_to_coco(split_dir: Path, out_json: Path) -> dict:
    img_dir = split_dir / 'images'
    ann_dir = split_dir / 'annotations'
    img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])

    images = []
    annotations = []
    ann_id = 1
    small_count = 0

    for img_id, img_path in enumerate(img_files, start=1):
        with Image.open(img_path) as im:
            width, height = im.size

        images.append({
            'id': img_id,
            'file_name': img_path.name,
            'width': width,
            'height': height,
        })

        txt_path = ann_dir / f'{img_path.stem}.txt'
        if not txt_path.exists():
            continue

        with txt_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().rstrip(',')
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 6:
                    continue
                x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                score = int(parts[4])
                cat_raw = int(parts[5])

                # Keep valid foreground boxes in categories 1..10
                if score == 0 or cat_raw < 1 or cat_raw > 10:
                    continue

                x1 = max(0.0, min(x, float(width)))
                y1 = max(0.0, min(y, float(height)))
                x2 = max(0.0, min(x + w, float(width)))
                y2 = max(0.0, min(y + h, float(height)))
                bw, bh = x2 - x1, y2 - y1
                if bw <= 1.0 or bh <= 1.0:
                    continue

                area = bw * bh
                if area < 32.0 * 32.0:
                    small_count += 1

                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': cat_raw - 1,  # 0-indexed [0..9]
                    'bbox': [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                    'area': round(area, 2),
                    'iscrowd': 0,
                })
                ann_id += 1

    coco_dict = {
        'info': {'description': 'VisDrone2019-DET in COCO format'},
        'licenses': [],
        'categories': VISDRONE_CATEGORIES,
        'images': images,
        'annotations': annotations,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open('w', encoding='utf-8') as f:
        json.dump(coco_dict, f)

    stats = {
        'split': split_dir.name,
        'num_images': len(images),
        'num_annotations': len(annotations),
        'small_obj_ratio': round(small_count / max(len(annotations), 1), 4),
        'out_json': str(out_json),
    }
    print(f'[VisDrone] Converted {stats}')
    return stats


def find_existing_visdrone_split(search_roots, split_folder: str):
    for root in search_roots:
        root_p = Path(root)
        if not root_p.exists():
            continue
        for candidate in root_p.rglob(split_folder):
            if (candidate / 'images').exists() and (candidate / 'annotations').exists():
                return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./dataset/visdrone')
    parser.add_argument('--search-input', type=str, nargs='*', default=['/kaggle/input'])
    parser.add_argument('--skip-download', action='store_true')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ann_out_dir = data_root / 'annotations'

    splits = {
        'train': ('VisDrone2019-DET-train', ann_out_dir / 'instances_train.json'),
        'val': ('VisDrone2019-DET-val', ann_out_dir / 'instances_val.json'),
    }

    summary = {}
    for key, (folder_name, out_json) in splits.items():
        found = find_existing_visdrone_split([data_root, *args.search_input], folder_name)
        if found is not None:
            split_dir = found
            target_link = data_root / folder_name
            if not target_link.exists():
                data_root.mkdir(parents=True, exist_ok=True)
                try:
                    os.symlink(split_dir.resolve(), target_link, target_is_directory=True)
                    split_dir = target_link
                except OSError:
                    pass
        elif not args.skip_download:
            split_dir = download_and_extract(VISDRONE_URLS[key], data_root, folder_name)
        else:
            raise FileNotFoundError(f'Could not find {folder_name} and --skip-download was set.')

        summary[key] = convert_split_to_coco(split_dir, out_json)

    with (data_root / 'dataset_stats.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
