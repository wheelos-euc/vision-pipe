# Copyright 2026 The WheelOS Team. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created Date: 2026-03-03
# Author: daohu527


import os
import json
import shutil
import logging
import random
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List
from PIL import Image
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetConverter:
    def __init__(self, input_dirs: List[str], output_dir: str, split_ratio: float = 0.8):
        self.input_dirs = [Path(d) for d in input_dirs]
        self.output_dir = Path(output_dir)
        self.split_ratio = split_ratio
        self.classes = []

        self.dirs = {
            'train_img': self.output_dir / "images/train",
            'val_img': self.output_dir / "images/val",
            'train_lab': self.output_dir / "labels/train",
            'val_lab': self.output_dir / "labels/val",
        }

    def _init_dirs(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def _scan_classes(self):
        found_classes = set()
        for idir in self.input_dirs:
            for json_file in idir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for shape in data.get('shapes', []):
                            found_classes.add(shape['label'])
                except:
                    continue
        self.classes = sorted(list(found_classes))
        logger.info(f"Detected classes: {self.classes}")

    @staticmethod
    def _convert_bbox(size, points):
        dw, dh = 1. / size[0], 1. / size[1]
        x1, y1 = points[0]
        x2, y2 = points[1]
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        return ((xmin+xmax)/2*dw, (ymin+ymax)/2*dh, (xmax-xmin)*dw, (ymax-ymin)*dh)

    def _process_file(self, task):
        img_path, subset = task
        json_path = img_path.with_suffix('.json')
        if not json_path.exists():
            return False
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            with Image.open(img_path) as img:
                w, h = img.size

            yolo_lines = []
            for s in data['shapes']:
                if s['shape_type'] != 'rectangle':
                    continue
                cls_id = self.classes.index(s['label'])
                bbox = self._convert_bbox((w, h), s['points'])
                yolo_lines.append(
                    f"{cls_id} " + " ".join([f"{a:.6f}" for a in bbox]))

            if not yolo_lines:
                return False
            unique_name = f"{img_path.parent.name}_{img_path.name}"
            shutil.copy2(img_path, self.dirs[f'{subset}_img'] / unique_name)
            with open(self.dirs[f'{subset}_lab'] / Path(unique_name).with_suffix('.txt'), 'w') as f:
                f.write("\n".join(yolo_lines))
            return True
        except:
            return False

    def run(self):
        if not self.input_dirs:
            logger.error(
                "No subdirectories found; please check the input path.")
            return
        self._init_dirs()
        self._scan_classes()
        all_tasks = []
        for idir in self.input_dirs:
            files = [f for f in idir.iterdir() if f.suffix.lower() in [
                '.jpg', '.png', '.jpeg']]
            random.shuffle(files)
            split = int(len(files) * self.split_ratio)
            for i, f in enumerate(files):
                all_tasks.append((f, 'train' if i < split else 'val'))

        with ProcessPoolExecutor() as exe:
            list(exe.map(self._process_file, all_tasks))

        with open(self.output_dir / 'data.yaml', 'w') as f:
            yaml.dump({'path': str(self.output_dir.absolute()), 'train': 'images/train',
                       'val': 'images/val', 'names': {i: n for i, n in enumerate(self.classes)}}, f)
        logger.info(f"Dataset saved to: {self.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Labelme multi-folder to YOLO dataset converter")
    parser.add_argument('--input', type=str, default='./',
                        help='Root path that contains multiple subfolders')
    parser.add_argument('--output', type=str,
                        default='./YOLO_Dataset', help='Output dataset path')
    parser.add_argument('--ratio', type=float, default=0.8,
                        help='Train split ratio (0-1)')

    args = parser.parse_args()

    # 1. Scan top-level subdirectories (exclude the output folder)
    root_path = Path(args.input)
    subdirs = [str(f.path) for f in os.scandir(root_path)
               if f.is_dir() and f.name != Path(args.output).name]

    logger.info(f"Found {len(subdirs)} subdirectories under {args.input}")

    # 2. Run conversion: build YOLO-style dataset and write data.yaml
    converter = DatasetConverter(subdirs, args.output, split_ratio=args.ratio)
    converter.run()
