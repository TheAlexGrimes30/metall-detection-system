#!/usr/bin/env python3
"""
Convert Supervisely JSON annotations to YOLO format for GC10-DET dataset.
"""

from pathlib import Path
import json
from collections import Counter

CLASS_NAMES = [
    "crease",
    "crescent_gap",
    "inclusion",
    "oil_spot",
    "punching_hole",
    "rolled_pit",
    "silk_spot",
    "waist_folding",
    "water_spot",
    "welding_line",
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
# Handle class name variations in the dataset
CLASS_ALIASES = {
    "waist folding": "waist_folding",
}


def convert_supervisely_to_yolo(
    images_dir: str,
    ann_dir: str,
    output_labels_dir: str,
) -> None:
    images_path = Path(images_dir)
    ann_path = Path(ann_dir)
    output_path = Path(output_labels_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = {f.stem: f for f in images_path.iterdir() if f.is_file()}
    ann_files = list(ann_path.glob("*.json"))

    print(f"Found {len(image_files)} images, {len(ann_files)} annotation files")

    class_counter = Counter()
    missing_ann = 0
    converted = 0

    for ann_file in ann_files:
        # Annotation file name: img_XX_XXXXXXXX_XXXX.jpg.json
        # Image file name: img_XX_XXXXXXXX_XXXX.jpg
        stem = ann_file.stem
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
            # Handle aliases
            class_title = CLASS_ALIASES.get(class_title, class_title)
            if class_title not in CLASS_TO_ID:
                print(f"Warning: Unknown class '{class_title}' in {ann_file}")
                continue

            class_id = CLASS_TO_ID[class_title]
            class_counter[class_title] += 1

            # Supervisely rectangle: exterior = [[x1, y1], [x2, y2]]
            points = obj["points"]["exterior"]
            x1, y1 = points[0]
            x2, y2 = points[1]

            # Clamp corner coordinates to image boundaries FIRST
            x1 = max(0, min(img_width - 1, x1))
            y1 = max(0, min(img_height - 1, y1))
            x2 = max(0, min(img_width - 1, x2))
            y2 = max(0, min(img_height - 1, y2))

            # Ensure x1 < x2, y1 < y2 after clamping
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            box_w = x_max - x_min
            box_h = y_max - y_min

            if box_w > 0 and box_h > 0:
                # Convert to YOLO format (normalized center x, y, width, height)
                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = box_w / img_width
                height = box_h / img_height

                # Final safety clamp (should already be in [0,1] after corner clamping)
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))

                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

        # Write YOLO label file
        output_file = output_path / f"{stem}.txt"
        output_file.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
        converted += 1

    print(f"Converted: {converted} files")
    print(f"Missing annotations for: {missing_ann} images")
    print("\nClass distribution:")
    for name in CLASS_NAMES:
        print(f"  {name}: {class_counter[name]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Supervisely JSON to YOLO format"
    )
    parser.add_argument("--images", required=False, help="Path to images folder")
    parser.add_argument("--ann", required=False, help="Path to annotations folder")
    parser.add_argument(
        "--output", required=False, help="Output labels folder (YOLO format)"
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run end-to-end self-test"
    )
    args = parser.parse_args()

    convert_supervisely_to_yolo(args.images, args.ann, args.output)