#!/usr/bin/env python3
"""
Convert Severstal Steel Defect Detection Supervisely annotations (bitmap/RLE)
to YOLO format.
"""

from pathlib import Path
import json
import base64
import zlib
import io
from collections import Counter
from typing import Optional, Dict

import cv2
import numpy as np

# Severstal class mapping (from meta.json)
SEVERSTAL_CLASSES = ["defect_1", "defect_2", "defect_3", "defect_4"]
CLASS_TO_ID = {name: i for i, name in enumerate(SEVERSTAL_CLASSES)}


def decode_bitmap(bitmap_data: str) -> np.ndarray:
    """
    Decode a Supervisely bitmap (base64 + zlib + PNG) to a binary mask.
    Returns: uint8 numpy array (0 or 255).
    """
    raw = base64.b64decode(bitmap_data)
    decompressed = zlib.decompress(raw)
    img = cv2.imdecode(np.frombuffer(decompressed, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode bitmap")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binarize: any non-zero pixel is part of the defect
    mask = (img > 0).astype(np.uint8) * 255
    return mask


def bitmap_to_bbox(
    mask: np.ndarray,
    origin: list[int],
    img_w: int,
    img_h: int,
) -> Optional[tuple[float, float, float, float]]:
    """
    Compute YOLO normalized bbox (x_center, y_center, width, height) from a bitmap mask.
    origin: [x, y] offset of the mask in the full image.
    Returns None if the mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    # Mask coordinates are relative to origin
    x_min = xs.min() + origin[0]
    x_max = xs.max() + origin[0]
    y_min = ys.min() + origin[1]
    y_max = ys.max() + origin[1]

    # Clamp to image boundaries
    x_min = max(0, min(img_w - 1, x_min))
    y_min = max(0, min(img_h - 1, y_min))
    x_max = max(0, min(img_w - 1, x_max))
    y_max = max(0, min(img_h - 1, y_max))

    w = x_max - x_min
    h = y_max - y_min

    if w <= 0 or h <= 0:
        return None

    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    width = w / img_w
    height = h / img_h

    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return x_center, y_center, width, height


def convert_severstal_to_yolo(
    images_dir: str,
    ann_dir: str,
    output_labels_dir: str,
    class_map: Optional[Dict[str, int]] = None,
) -> None:
    """
    Convert Severstal Supervisely JSON annotations (bitmap format) to YOLO format.
    """
    images_path = Path(images_dir)
    ann_path = Path(ann_dir)
    output_path = Path(output_labels_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cmap = class_map or CLASS_TO_ID

    image_files = {f.stem: f for f in images_path.iterdir() if f.is_file()}
    ann_files = list(ann_path.glob("*.json"))

    print(f"Found {len(image_files)} images, {len(ann_files)} annotation files")

    class_counter = Counter()
    missing_ann = 0
    converted = 0
    skipped_empty = 0

    for ann_file in ann_files:
        stem = ann_file.stem
        # Supervisely JSON: name.jpg.json -> stem becomes name.jpg
        if stem.endswith(".jpg"):
            stem = stem[:-4]

        if stem not in image_files:
            missing_ann += 1
            continue

        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_width = data["size"]["width"]
        img_height = data["size"]["height"]

        yolo_lines = []

        for obj in data.get("objects", []):
            class_title = obj.get("classTitle", "")
            if class_title not in cmap:
                print(f"Warning: Unknown class '{class_title}' in {ann_file}")
                continue

            class_id = cmap[class_title]
            class_counter[class_title] += 1

            # Severstal uses bitmap geometry
            if obj.get("geometryType") != "bitmap":
                continue

            bitmap = obj.get("bitmap", {})
            bmp_data = bitmap.get("data", "")
            origin = bitmap.get("origin", [0, 0])

            if not bmp_data:
                continue

            try:
                mask = decode_bitmap(bmp_data)
                bbox = bitmap_to_bbox(mask, origin, img_width, img_height)
            except Exception as e:
                print(f"Warning: Failed to decode bitmap in {ann_file}: {e}")
                continue

            if bbox is None:
                skipped_empty += 1
                continue

            x_center, y_center, width, height = bbox
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        # Write YOLO label file
        output_file = output_path / f"{stem}.txt"
        output_file.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
        converted += 1

    print(f"Converted: {converted} files")
    print(f"Missing annotations for: {missing_ann} images")
    print(f"Skipped empty bitmaps: {skipped_empty}")
    print("\nClass distribution:")
    for name in SEVERSTAL_CLASSES:
        print(f"  {name}: {class_counter[name]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Severstal Supervisely bitmap annotations to YOLO format"
    )
    parser.add_argument("--images", required=True, help="Path to images folder")
    parser.add_argument("--ann", required=True, help="Path to annotations folder")
    parser.add_argument(
        "--output", required=True, help="Output labels folder (YOLO format)"
    )
    args = parser.parse_args()

    convert_severstal_to_yolo(args.images, args.ann, args.output)