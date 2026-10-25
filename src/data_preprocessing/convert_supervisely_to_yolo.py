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


def run_self_test() -> bool:
    """
    End-to-end self-test: create synthetic Supervisely JSON,
    convert to YOLO, then validate with the EDA validator.
    Returns True if all checks pass.
    """
    import tempfile
    from pathlib import Path

    NUM_CLASSES = len(CLASS_NAMES)

    # Create a minimal synthetic annotation matching GC10-DET structure
    test_json = {
        "description": "",
        "tags": [],
        "size": {"height": 1000, "width": 2048},
        "objects": [
            {
                "id": 1,
                "classId": 1,
                "description": "",
                "geometryType": "rectangle",
                "labelerLogin": "test",
                "createdAt": "2024-01-01T00:00:00.000Z",
                "updatedAt": "2024-01-01T00:00:00.000Z",
                "tags": [],
                "classTitle": "inclusion",
                "points": {
                    "exterior": [[100, 200], [300, 400]],
                    "interior": []
                }
            },
            {
                "id": 2,
                "classId": 2,
                "description": "",
                "geometryType": "rectangle",
                "labelerLogin": "test",
                "createdAt": "2024-01-01T00:00:00.000Z",
                "updatedAt": "2024-01-01T00:00:00.000Z",
                "tags": [],
                "classTitle": "punching_hole",
                "points": {
                    "exterior": [[500, 600], [800, 900]],
                    "interior": []
                }
            },
            # Test edge case: box extending beyond image boundary
            {
                "id": 3,
                "classId": 3,
                "description": "",
                "geometryType": "rectangle",
                "labelerLogin": "test",
                "createdAt": "2024-01-01T00:00:00.000Z",
                "updatedAt": "2024-01-01T00:00:00.000Z",
                "tags": [],
                "classTitle": "oil_spot",
                "points": {
                    "exterior": [[-50, -50], [2100, 1100]],  # Outside image bounds
                    "interior": []
                }
            },
        ]
    }

    # Create temp directories
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        images_dir = tmpdir / "images"
        ann_dir = tmpdir / "ann"
        output_dir = tmpdir / "labels"

        images_dir.mkdir()
        ann_dir.mkdir()

        # Write test JSON
        ann_file = ann_dir / "test_image.jpg.json"
        ann_file.write_text(json.dumps(test_json))

        # Create dummy image file (just for stem matching)
        (images_dir / "test_image.jpg").write_bytes(b"fake")

        # Run conversion
        convert_supervisely_to_yolo(str(images_dir), str(ann_dir), str(output_dir))

        # Verify output
        label_file = output_dir / "test_image.txt"
        if not label_file.exists():
            print("FAIL: Label file not created")
            return False

        lines = label_file.read_text().strip().splitlines()
        if len(lines) != 3:
            print(f"FAIL: Expected 3 boxes, got {len(lines)}")
            return False

        # Validate each line matches YOLO format and bounds
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                print(f"FAIL: Line {i} has {len(parts)} parts, expected 5")
                return False

            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            # Check bounds
            if not (0 <= class_id < NUM_CLASSES):
                print(f"FAIL: Line {i} invalid class_id {class_id}")
                return False
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                print(f"FAIL: Line {i} coords out of bounds: x={x}, y={y}, w={w}, h={h}")
                return False
            if w <= 0 or h <= 0:
                print(f"FAIL: Line {i} zero/negative size: w={w}, h={h}")
                return False

            # Check box within image bounds (with epsilon)
            eps = 1e-6
            if x - w/2 < -eps or x + w/2 > 1+eps or y - h/2 < -eps or y + h/2 > 1+eps:
                print(f"FAIL: Line {i} box exceeds image bounds")
                return False

        # Verify class mapping
        expected_classes = [2, 4, 3]  # inclusion, punching_hole, oil_spot
        actual_classes = [int(l.split()[0]) for l in lines]
        if actual_classes != expected_classes:
            print(f"FAIL: Class IDs mismatch. Expected {expected_classes}, got {actual_classes}")
            return False

        # Verify the edge-case box was clamped correctly
        # The oil_spot box was [[-50, -50], [2100, 1100]] -> should become full image [0,0] to [2047,999]
        oil_spot_line = lines[2]
        _, x, y, w, h = oil_spot_line.split()
        x, y, w, h = float(x), float(y), float(w), float(h)
        # Should be centered at ~0.5, 0.5 with width/height ~1.0
        if not (abs(x - 0.5) < 0.01 and abs(y - 0.5) < 0.01 and abs(w - 1.0) < 0.01 and abs(h - 1.0) < 0.01):
            print(f"FAIL: Edge-case clamping failed. Got x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}")
            return False

        print("SUCCESS: All self-test checks passed!")
        print(f"  Generated labels: {lines}")
        return True


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

    if args.self_test:
        success = run_self_test()
        exit(0 if success else 1)

    if not (args.images and args.ann and args.output):
        parser.error("--images, --ann, and --output are required unless --self-test is used")

    convert_supervisely_to_yolo(args.images, args.ann, args.output)