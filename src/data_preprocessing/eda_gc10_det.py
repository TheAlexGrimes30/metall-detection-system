from pathlib import Path
from collections import Counter
import argparse
import hashlib

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CLASS_NAMES = [
    "crease", "crescent_gap", "inclusion", "oil_spot",
    "punching_hole", "rolled_pit", "silk_spot",
    "waist_folding", "water_spot", "welding_line"
]

NUM_CLASSES = len(CLASS_NAMES)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ============================================================
# Dataset discovery
# ============================================================

def find_images(root):
    excluded = {"eda_results", "processed", "runs", ".git"}

    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
        and not any(part.lower() in excluded for part in p.parts)
    )


def find_label(image_path, root):
    """
    Matches images/train/image.jpg to labels/train/image.txt.
    Also supports a flat images/ and labels/ layout.
    """
    parts = list(image_path.parts)
    lower_parts = [p.lower() for p in parts]

    if "images" in lower_parts:
        idx = lower_parts.index("images")
        label_parts = parts[:]
        label_parts[idx] = "labels"

        candidate = Path(*label_parts).with_suffix(".txt")

        if candidate.is_file():
            return candidate

    candidates = [
        p for p in root.rglob(f"{image_path.stem}.txt")
        if "labels" in [part.lower() for part in p.parts]
        and not any(
            part.lower() in {"eda_results", "processed", "runs"}
            for part in p.parts
        )
    ]

    return candidates[0] if len(candidates) == 1 else None


def get_split(path):
    parts = [p.lower() for p in path.parts]

    for split in ("train", "valid", "val", "validation", "test"):
        if split in parts:
            return "val" if split == "validation" else split

    return "unspecified"


# ============================================================
# YOLO annotation analysis
# ============================================================

def read_yolo_labels(label_path):
    """
    YOLO format:
    class_id x_center y_center width height

    Coordinates are normalized to [0, 1].
    Returns valid boxes and annotation errors.
    """
    if label_path is None or not label_path.is_file():
        return [], "missing_label_file"

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return [], "unreadable_label_file"

    if not any(line.strip() for line in lines):
        return [], "empty_label_file"

    boxes = []

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            boxes.append({
                "valid": False,
                "error": f"line_{line_number}_expected_5_values"
            })
            continue

        try:
            class_id = int(values[0])
            x, y, w, h = map(float, values[1:])
        except ValueError:
            boxes.append({
                "valid": False,
                "error": f"line_{line_number}_non_numeric"
            })
            continue

        finite = np.all(np.isfinite([x, y, w, h]))

        valid = (
            0 <= class_id < NUM_CLASSES
            and finite
            and all(0 <= v <= 1 for v in (x, y, w, h))
            and w > 0
            and h > 0
            and x - w / 2 >= -1e-6
            and x + w / 2 <= 1 + 1e-6
            and y - h / 2 >= -1e-6
            and y + h / 2 <= 1 + 1e-6
        )

        boxes.append({
            "valid": bool(valid),
            "class_id": class_id,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": w * h,
            "aspect_ratio": w / h if h > 0 else np.nan,
            "error": "" if valid else f"line_{line_number}_invalid_values"
        })

    status = (
        "ok"
        if all(box.get("valid", False) for box in boxes)
        else "invalid_annotations"
    )

    return boxes, status


# ============================================================
# Image descriptive statistics
# ============================================================

def image_statistics(path):
    stats = {
        "filename": path.name,
        "path": str(path),
        "extension": path.suffix.lower(),
        "file_size_bytes": path.stat().st_size,
        "readable": False
    }

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        stats["error"] = "cannot_read_image"
        return stats

    stats["readable"] = True
    stats["height"], stats["width"] = image.shape[:2]
    stats["pixels"] = stats["height"] * stats["width"]
    stats["channels"] = 1 if image.ndim == 2 else image.shape[2]
    stats["dtype"] = str(image.dtype)

    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)

    gray = gray.astype(np.float64)

    stats.update({
        "gray_mean": gray.mean(),
        "gray_std": gray.std(),
        "gray_min": gray.min(),
        "gray_max": gray.max(),
        "gray_median": np.median(gray),
        "gray_p05": np.percentile(gray, 5),
        "gray_p95": np.percentile(gray, 95),
        "gray_dynamic_range": gray.max() - gray.min()
    })

    if image.ndim == 3 and image.shape[2] >= 3:
        stats["blue_mean"] = image[:, :, 0].mean()
        stats["green_mean"] = image[:, :, 1].mean()
        stats["red_mean"] = image[:, :, 2].mean()
    else:
        stats["blue_mean"] = np.nan
        stats["green_mean"] = np.nan
        stats["red_mean"] = np.nan

    return stats


# ============================================================
# Plotting
# ============================================================

def save_histogram(series, title, xlabel, output_path):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=30, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================
# Main EDA
# ============================================================

def run_eda(root, output):
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    image_paths = find_images(root)

    if not image_paths:
        raise RuntimeError(f"No images found in {root}")

    image_rows = []
    object_rows = []
    error_rows = []
    hashes = Counter()

    for i, path in enumerate(image_paths, start=1):
        stats = image_statistics(path)
        stats["split"] = get_split(path)

        label_path = find_label(path, root)
        boxes, label_status = read_yolo_labels(label_path)

        stats["label_path"] = str(label_path) if label_path else ""
        stats["label_status"] = label_status

        try:
            hashes[hashlib.md5(path.read_bytes()).hexdigest()] += 1
        except OSError:
            pass

        valid_boxes = [b for b in boxes if b.get("valid", False)]

        stats["num_objects"] = len(valid_boxes)
        stats["invalid_lines"] = sum(
            not b.get("valid", False) for b in boxes
        )

        image_rows.append(stats)

        for line_number, box in enumerate(boxes, start=1):
            if not box.get("valid", False):
                error_rows.append({
                    "image": str(path),
                    "label": str(label_path) if label_path else "",
                    "line": line_number,
                    "error": box.get("error", "invalid_annotation")
                })
                continue

            object_rows.append({
                "image": path.name,
                "split": stats["split"],
                "class_id": box["class_id"],
                "class_name": CLASS_NAMES[box["class_id"]],
                "x_center": box["x"],
                "y_center": box["y"],
                "width_fraction": box["w"],
                "height_fraction": box["h"],
                "area_fraction": box["area"],
                "aspect_ratio": box["aspect_ratio"],
                "width_pixels": box["w"] * stats.get("width", np.nan),
                "height_pixels": box["h"] * stats.get("height", np.nan)
            })

        if i % 250 == 0:
            print(f"Processed {i}/{len(image_paths)} images")

    images = pd.DataFrame(image_rows)
    objects = pd.DataFrame(object_rows)
    errors = pd.DataFrame(
        error_rows,
        columns=["image", "label", "line", "error"]
    )

    # Export raw statistics.
    images.to_csv(output / "image_statistics.csv", index=False)
    objects.to_csv(output / "object_statistics.csv", index=False)
    errors.to_csv(output / "annotation_errors.csv", index=False)

    # Descriptive statistics for images.
    numeric_columns = [
        "width", "height", "pixels", "channels", "file_size_bytes",
        "gray_mean", "gray_std", "gray_min", "gray_max",
        "gray_median", "gray_p05", "gray_p95",
        "gray_dynamic_range", "num_objects", "invalid_lines"
    ]
    numeric_columns = [c for c in numeric_columns if c in images.columns]

    images[numeric_columns].describe().T.to_csv(
        output / "image_descriptive_statistics.csv"
    )

    # Distribution of classes, including classes with zero objects.
    counts = (
        objects["class_id"].value_counts().to_dict()
        if not objects.empty else {}
    )

    classes = pd.DataFrame({
        "class_id": range(NUM_CLASSES),
        "class_name": CLASS_NAMES,
        "object_count": [int(counts.get(i, 0)) for i in range(NUM_CLASSES)]
    })

    total_objects = max(1, int(classes["object_count"].sum()))
    classes["percentage"] = classes["object_count"] / total_objects * 100
    classes.to_csv(output / "class_distribution.csv", index=False)

    plt.figure(figsize=(11, 6))
    plt.bar(classes["class_name"], classes["object_count"])
    plt.title("GC10-DET: object count by class")
    plt.xlabel("Class")
    plt.ylabel("Bounding boxes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output / "class_distribution.png", dpi=150)
    plt.close()

    # Bounding-box statistics and distributions.
    if not objects.empty:
        bbox_columns = [
            "width_fraction", "height_fraction", "area_fraction",
            "aspect_ratio", "width_pixels", "height_pixels",
            "x_center", "y_center"
        ]

        objects.groupby("class_name")[bbox_columns].describe().to_csv(
            output / "bounding_box_descriptive_statistics.csv"
        )

        for column, title, xlabel in [
            ("width_fraction", "Bounding-box width", "Normalized width"),
            ("height_fraction", "Bounding-box height", "Normalized height"),
            ("area_fraction", "Bounding-box area", "Normalized area"),
            ("aspect_ratio", "Bounding-box aspect ratio", "Width / height"),
            ("x_center", "Horizontal object position", "Normalized x"),
            ("y_center", "Vertical object position", "Normalized y")
        ]:
            save_histogram(
                objects[column], title, xlabel,
                output / f"{column}.png"
            )

        plt.figure(figsize=(7, 6))
        plt.scatter(
            objects["width_fraction"],
            objects["height_fraction"],
            alpha=0.4
        )
        plt.xlabel("Normalized width")
        plt.ylabel("Normalized height")
        plt.title("Bounding-box width vs height")
        plt.tight_layout()
        plt.savefig(output / "bbox_width_vs_height.png", dpi=150)
        plt.close()

    # Image-level distributions.
    for column, title, xlabel in [
        ("width", "Image width distribution", "Width in pixels"),
        ("height", "Image height distribution", "Height in pixels"),
        ("file_size_bytes", "Image file size", "Bytes"),
        ("gray_mean", "Mean grayscale intensity", "Intensity"),
        ("gray_std", "Grayscale contrast", "Standard deviation"),
        ("num_objects", "Objects per image", "Bounding boxes")
    ]:
        save_histogram(
            images[column], title, xlabel,
            output / f"{column}_distribution.png"
        )

    # Split statistics.
    images.groupby("split").agg(
        image_count=("filename", "count"),
        readable_count=("readable", "sum"),
        mean_objects=("num_objects", "mean"),
        median_objects=("num_objects", "median")
    ).to_csv(output / "split_statistics.csv")

    # Correlation matrix.
    corr_columns = [
        c for c in [
            "width", "height", "pixels", "file_size_bytes",
            "gray_mean", "gray_std", "num_objects"
        ]
        if c in images.columns
    ]

    correlation = images[corr_columns].corr(numeric_only=True)
    correlation.to_csv(output / "image_feature_correlations.csv")

    if not correlation.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot = ax.imshow(
            correlation.values, vmin=-1, vmax=1, aspect="auto"
        )
        ax.set_xticks(
            range(len(correlation.columns)),
            correlation.columns, rotation=45, ha="right"
        )
        ax.set_yticks(range(len(correlation.index)), correlation.index)
        fig.colorbar(plot, ax=ax, label="Correlation")
        ax.set_title("Image feature correlation matrix")
        fig.tight_layout()
        fig.savefig(output / "image_feature_correlations.png", dpi=150)
        plt.close(fig)

    # Summary.
    summary = {
        "total_images": len(images),
        "readable_images": int(images["readable"].sum()),
        "unreadable_images": int((~images["readable"]).sum()),
        "valid_bounding_boxes": len(objects),
        "images_without_valid_objects": int((images["num_objects"] == 0).sum()),
        "missing_label_files": int(
            (images["label_status"] == "missing_label_file").sum()
        ),
        "empty_label_files": int(
            (images["label_status"] == "empty_label_file").sum()
        ),
        "images_with_invalid_annotations": int(
            (images["label_status"] == "invalid_annotations").sum()
        ),
        "invalid_annotation_lines": len(errors),
        "byte_identical_duplicate_groups": sum(
            n > 1 for n in hashes.values()
        )
    }

    pd.Series(summary, name="value").to_csv(
        output / "dataset_summary.csv", header=True
    )

    with (output / "EDA_REPORT.txt").open("w", encoding="utf-8") as report:
        report.write("GC10-DET EDA REPORT\n")
        report.write("=" * 50 + "\n\n")

        for key, value in summary.items():
            report.write(f"{key}: {value}\n")

        report.write("\nCLASS DISTRIBUTION\n")
        report.write(classes.to_string(index=False))

        report.write("\n\nIMAGE DESCRIPTIVE STATISTICS\n")
        report.write(images[numeric_columns].describe().to_string())

    print("\nEDA completed.")
    print(f"Images: {len(images)}")
    print(f"Valid objects: {len(objects)}")
    print(f"Annotation errors: {len(errors)}")
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", required=True,
        help="Path to the root folder of GC10-DET"
    )
    parser.add_argument(
        "--output", default="eda_results",
        help="Directory for EDA results"
    )
    args = parser.parse_args()

    run_eda(Path(args.root), Path(args.output))
    
    
    
#     
# python eda_gc10_det.py \
#     --root ~/Datasets/GC10-DET \
#     --output ./eda_results
# or
# from pathlib import Path
# from eda_gc10_det import run_eda
# 
# run_eda(
#     Path("~/Datasets/GC10-DET").expanduser(),
#     Path("./eda_results")
# )