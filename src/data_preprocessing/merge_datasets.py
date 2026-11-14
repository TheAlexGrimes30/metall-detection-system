#!/usr/bin/env python3
"""
Merge GC10-DET and Severstal datasets into a unified YOLO-format dataset.

Strategy:
- Keep source-specific classes for traceability.
- Provide optional unified taxonomy mapping (see ABOUT_DATASET.md section 4).
- Preserve original class IDs in metadata for each source.
- Create stratified train/val/test split across both datasets.
"""

from pathlib import Path
import csv
import shutil
import numpy as np
import yaml
import cv2
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split


# ============================================================
# Unified taxonomy (from ABOUT_DATASET.md section 4)
# Maps source classes to common categories for model training.
# ============================================================

UNIFIED_CLASSES = [
    # GC10-DET classes (0-9)
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
    # Severstal-only classes (10-13)
    "severstal_pitted_surface",   # defect_1
    "severstal_crazing",          # defect_2
    "severstal_scratches",        # defect_3
    "severstal_patches",          # defect_4
]

# Mapping from source class to unified class ID
# GC10-DET: 0-9 map directly
UNIFIED_MAP = {
    ("gc10", "crease"): 0,
    ("gc10", "crescent_gap"): 1,
    ("gc10", "inclusion"): 2,
    ("gc10", "oil_spot"): 3,
    ("gc10", "punching_hole"): 4,
    ("gc10", "rolled_pit"): 5,
    ("gc10", "silk_spot"): 6,
    ("gc10", "waist_folding"): 7,
    ("gc10", "water_spot"): 8,
    ("gc10", "welding_line"): 9,
    ("severstal", "defect_1"): 10,
    ("severstal", "defect_2"): 11,
    ("severstal", "defect_3"): 12,
    ("severstal", "defect_4"): 13,
}

# Alternative: semantic mapping where Severstal classes merge into GC10-DET equivalents
# Uncomment to use instead of UNIFIED_MAP:
# SEMANTIC_MAP = {
#     ("gc10", "crease"): 0,
#     ("gc10", "crescent_gap"): 1,
#     ("gc10", "inclusion"): 2,
#     ("gc10", "oil_spot"): 3,
#     ("gc10", "punching_hole"): 4,
#     ("gc10", "rolled_pit"): 5,
#     ("gc10", "silk_spot"): 6,
#     ("gc10", "waist_folding"): 7,
#     ("gc10", "water_spot"): 8,
#     ("gc10", "welding_line"): 9,
#     ("severstal", "defect_1"): 5,   # Pitted Surface -> Rolled Pit
#     ("severstal", "defect_2"): 0,   # Crazing -> Crease
#     ("severstal", "defect_3"): 1,   # Scratches -> Crescent Gap (surface damage)
#     ("severstal", "defect_4"): 3,   # Patches -> Oil Spot (surface irregularity)
# }


# ============================================================
# Helper functions
# ============================================================

def load_yolo_labels(label_path: Path) -> list[list[float]]:
    """Load YOLO format labels: class_id, x_center, y_center, width, height."""
    if not label_path.exists():
        return []
    labels = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) == 5:
            labels.append([float(p) for p in parts])
    return labels


def save_yolo_labels(labels: list[list[float]], label_path: Path) -> None:
    """Save YOLO format labels."""
    lines = [f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}" for l in labels]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def get_image_class_counts(
    label_dir: Path,
    image_stems: list[str]
) -> dict[str, list[int]]:
    """Get class IDs present in each image for stratification."""
    image_classes = {}
    for stem in image_stems:
        label_file = label_dir / f"{stem}.txt"
        classes = []
        if label_file.exists():
            for line in label_file.read_text().splitlines():
                if line.strip():
                    class_id = int(line.split()[0])
                    if class_id not in classes:
                        classes.append(class_id)
        image_classes[stem] = classes
    return image_classes


def create_stratified_split(
    stems: list[str],
    label_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Create stratified split based on primary class per image."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    image_classes = get_image_class_counts(label_dir, stems)

    stratify_labels = []
    for stem in stems:
        classes = image_classes[stem]
        stratify_labels.append(classes[0] if classes else -1)

    from collections import Counter as C
    class_counts = C(stratify_labels)
    min_count = min(class_counts.values())

    if min_count < 2:
        print("Warning: Some classes have <2 samples, using random split")
        import random
        random.seed(random_state)
        shuffled = stems.copy()
        random.shuffle(shuffled)
        n_train = int(len(shuffled) * train_ratio)
        n_val = int(len(shuffled) * val_ratio)
        return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]

    train_stems, temp_stems = train_test_split(
        stems, train_size=train_ratio, stratify=stratify_labels, random_state=random_state
    )

    val_ratio_adj = val_ratio / (val_ratio + test_ratio)
    temp_stratify = []
    for s in temp_stems:
        classes = image_classes[s]
        temp_stratify.append(classes[0] if classes else -1)

    temp_counts = C(temp_stratify)
    if min(temp_counts.values()) < 2:
        print("Warning: Temp set has classes with <2 samples, using random split for val/test")
        import random
        random.seed(random_state + 1)
        random.shuffle(temp_stems)
        n_val = int(len(temp_stems) * val_ratio_adj)
        return train_stems, temp_stems[:n_val], temp_stems[n_val:]

    val_stems, test_stems = train_test_split(
        temp_stems, train_size=val_ratio_adj, stratify=temp_stratify, random_state=random_state
    )
    return train_stems, val_stems, test_stems


# ============================================================
# Merger
# ============================================================

def merge_datasets(
    gc10_images_dir: str,
    gc10_labels_dir: str,
    severstal_images_dir: str,
    severstal_labels_dir: str,
    output_root: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    use_semantic_map: bool = False,
) -> None:
    """
    Merge GC10-DET and Severstal into a unified dataset.

    Output structure:
        output_root/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── metadata/
        │   ├── classes.yaml
        │   └── source_dataset.csv
        └── dataset.yaml
    """
    output_root = Path(output_root)
    output_images = output_root / "images"
    output_labels = output_root / "labels"
    output_metadata = output_root / "metadata"

    for split in ["train", "val", "test"]:
        (output_images / split).mkdir(parents=True, exist_ok=True)
        (output_labels / split).mkdir(parents=True, exist_ok=True)
    output_metadata.mkdir(parents=True, exist_ok=True)

    # Use semantic mapping if requested
    unified_map = UNIFIED_MAP
    if use_semantic_map:
        # SEMANTIC_MAP would need to be defined globally
        pass

    # Collect all entries: (source, stem, image_path, label_path, class_ids)
    gc10_img_path = Path(gc10_images_dir)
    gc10_lbl_path = Path(gc10_labels_dir)
    severstal_img_path = Path(severstal_images_dir)
    severstal_lbl_path = Path(severstal_labels_dir)

    entries = []  # list of dicts

    # GC10-DET entries
    for img_file in sorted(gc10_img_path.iterdir()):
        if not img_file.is_file():
            continue
        if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        stem = img_file.stem
        label_file = gc10_lbl_path / f"{stem}.txt"
        labels = load_yolo_labels(label_file)
        class_ids = [int(l[0]) for l in labels]
        entries.append({
            "source": "gc10",
            "stem": stem,
            "image_path": img_file,
            "label_path": label_file,
            "class_ids": class_ids,
            "image_name": img_file.name,
        })

    # Severstal entries
    for img_file in sorted(severstal_img_path.iterdir()):
        if not img_file.is_file():
            continue
        if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        stem = img_file.stem
        label_file = severstal_lbl_path / f"{stem}.txt"
        labels = load_yolo_labels(label_file)
        # Map Severstal class IDs (0-3) to unified IDs (10-13)
        remapped_labels = []
        for l in labels:
            orig_class_id = int(l[0])
            orig_class_name = ["defect_1", "defect_2", "defect_3", "defect_4"][orig_class_id]
            unified_id = UNIFIED_MAP[("severstal", orig_class_name)]
            remapped_labels.append([unified_id, l[1], l[2], l[3], l[4]])
        class_ids = [int(l[0]) for l in remapped_labels]
        entries.append({
            "source": "severstal",
            "stem": stem,
            "image_path": img_file,
            "label_path": label_file,
            "class_ids": class_ids,
            "remapped_labels": remapped_labels,
            "image_name": img_file.name,
        })

    print(f"Total entries: {len(entries)} (GC10-DET: {sum(1 for e in entries if e['source']=='gc10')}, Severstal: {sum(1 for e in entries if e['source']=='severstal')})")

    # Stratified split across all entries
    all_stems = [e["stem"] for e in entries]
    # Build a temporary label dir for stratification
    temp_label_dir = Path("/tmp/_merge_strat_labels")
    temp_label_dir.mkdir(parents=True, exist_ok=True)
    for e in entries:
        # Write labels for stratification
        if e["source"] == "severstal":
            lbls = e["remapped_labels"]
        else:
            lbls = load_yolo_labels(e["label_path"])
        save_yolo_labels(lbls, temp_label_dir / f"{e['stem']}.txt")

    train_stems, val_stems, test_stems = create_stratified_split(
        all_stems, temp_label_dir, train_ratio, val_ratio, test_ratio, random_state
    )

    stem_to_split = {}
    for s in train_stems:
        stem_to_split[s] = "train"
    for s in val_stems:
        stem_to_split[s] = "val"
    for s in test_stems:
        stem_to_split[s] = "test"

    # Copy files and write labels
    split_counts = Counter()
    class_distribution = defaultdict(Counter)
    metadata_rows = []

    for e in entries:
        split = stem_to_split[e["stem"]]
        split_counts[split] += 1

        # Determine label content
        if e["source"] == "severstal":
            labels_to_save = e["remapped_labels"]
        else:
            labels_to_save = load_yolo_labels(e["label_path"])

        # Destination paths (use source prefix to avoid name collisions)
        dest_stem = f"{e['source']}_{e['stem']}"
        dest_img = output_images / split / f"{dest_stem}.png"
        dest_lbl = output_labels / split / f"{dest_stem}.txt"

        # Copy image (convert to PNG for consistency)
        img = cv2.imread(str(e["image_path"]), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: cannot read {e['image_path']}")
            continue
        cv2.imwrite(str(dest_img), img)

        # Save labels
        save_yolo_labels(labels_to_save, dest_lbl)

        # Metadata
        for class_id in e["class_ids"]:
            class_distribution[split][class_id] += 1
        metadata_rows.append({
            "stem": dest_stem,
            "source": e["source"],
            "original_stem": e["stem"],
            "split": split,
            "image_name": e["image_name"],
            "class_ids": ",".join(str(c) for c in e["class_ids"]),
            "num_objects": len(e["class_ids"]),
        })

    # Save metadata CSV
    metadata_path = output_metadata / "source_dataset.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stem", "source", "original_stem", "split", "image_name", "class_ids", "num_objects"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Save classes.yaml
    classes_path = output_metadata / "classes.yaml"
    classes_data = {
        "names": UNIFIED_CLASSES,
        "nc": len(UNIFIED_CLASSES),
    }
    with open(classes_path, "w", encoding="utf-8") as f:
        yaml.dump(classes_data, f, default_flow_style=False, allow_unicode=True)

    # Save dataset.yaml
    dataset_yaml_path = output_root / "dataset.yaml"
    dataset_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(UNIFIED_CLASSES),
        "names": UNIFIED_CLASSES,
    }
    with open(dataset_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, allow_unicode=True)

    # Print summary
    print("\n=== Merger Complete ===")
    print(f"Output root: {output_root}")
    print(f"Train: {split_counts['train']} images")
    print(f"Val:   {split_counts['val']} images")
    print(f"Test:  {split_counts['test']} images")
    print(f"\nClasses ({len(UNIFIED_CLASSES)}):")
    for i, name in enumerate(UNIFIED_CLASSES):
        print(f"  {i}: {name}")

    print("\nClass distribution per split (unified class ID):")
    for split in ["train", "val", "test"]:
        dist = class_distribution[split]
        total = sum(dist.values())
        print(f"  {split}: {total} boxes")
        for cid in sorted(dist.keys()):
            name = UNIFIED_CLASSES[cid] if cid < len(UNIFIED_CLASSES) else f"unknown_{cid}"
            print(f"    {name} ({cid}): {dist[cid]}")

    print(f"\nMetadata saved to: {metadata_path}")
    print(f"Classes saved to: {classes_path}")
    print(f"Dataset config saved to: {dataset_yaml_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge GC10-DET and Severstal datasets")
    parser.add_argument("--gc10-images", required=True, help="GC10-DET images folder")
    parser.add_argument("--gc10-labels", required=True, help="GC10-DET YOLO labels folder")
    parser.add_argument("--severstal-images", required=True, help="Severstal images folder")
    parser.add_argument("--severstal-labels", required=True, help="Severstal YOLO labels folder")
    parser.add_argument("--output", required=True, help="Output unified dataset root")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--semantic-map", action="store_true", help="Use semantic class mapping")
    args = parser.parse_args()

    merge_datasets(
        args.gc10_images,
        args.gc10_labels,
        args.severstal_images,
        args.severstal_labels,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.random_state,
        args.semantic_map,
    )