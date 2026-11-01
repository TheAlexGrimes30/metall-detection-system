from pathlib import Path
import cv2
import numpy as np
import random
from typing import Optional
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TARGET_SIZE = (256, 256)
TARGET_FORMAT = "PNG"

NUM_CLASSES = 10

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

# Dataset statistics from EDA (grayscale)
DATASET_MEAN = 85.695
DATASET_STD = 25.074

# Classes needing special augmentation (small objects)
SMALL_OBJECT_CLASSES = {2, 4}  # inclusion, punching_hole


# ============================================================
# Loading
# ============================================================

def load_images_from_folder(
    folder: str
) -> list[tuple[str, np.ndarray]]:
    """Load supported images and preserve their filenames."""
    folder_path = Path(folder)

    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = []

    for path in sorted(folder_path.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Warning: cannot read image: {path}")
            continue

        images.append((path.name, image))

    return images


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


# ============================================================
# Preprocessing
# ============================================================

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to single-channel grayscale."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def resize_image(
    image: np.ndarray,
    size: tuple[int, int] = TARGET_SIZE
) -> np.ndarray:
    """Resize image to the target size."""
    return cv2.resize(image, size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image using dataset statistics: (image - mean) / std.
    Input: uint8 [0, 255], Output: float32 normalized.
    """
    image = image.astype(np.float32)
    image = (image - DATASET_MEAN) / DATASET_STD
    return image


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize for visualization: (image * std) + mean -> uint8."""
    image = image * DATASET_STD + DATASET_MEAN
    return np.clip(image, 0, 255).astype(np.uint8)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess a GC10-DET image for training.

    Steps:
    1. Convert to grayscale (dataset is single-channel).
    2. Resize to target size (256x256).
    3. Normalize using dataset mean/std.

    Returns:
        Float32 normalized image, shape (H, W).
    """
    image = convert_to_grayscale(image)
    image = resize_image(image, TARGET_SIZE)
    image = normalize_image(image)
    return image


def scale_bboxes(
    labels: list[list[float]],
    orig_size: tuple[int, int],
    target_size: tuple[int, int] = TARGET_SIZE
) -> list[list[float]]:
    """
    Scale YOLO bounding boxes from original to target size.
    YOLO format is normalized [0,1], so no scaling needed if both
    original and target maintain aspect ratio.
    However, if aspect ratio changes, we need to adjust.

    Since all GC10-DET images are 2048x1000 and target is 256x256,
    aspect ratio changes (2.048:1 -> 1:1), so we must scale.
    """
    orig_h, orig_w = orig_size
    target_h, target_w = target_size

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    scaled = []
    for label in labels:
        class_id, x, y, w, h = label
        # Convert normalized to absolute, scale, then back to normalized
        x_abs = x * orig_w
        y_abs = y * orig_h
        w_abs = w * orig_w
        h_abs = h * orig_h

        x_new = (x_abs * scale_x) / target_w
        y_new = (y_abs * scale_y) / target_h
        w_new = (w_abs * scale_x) / target_w
        h_new = (h_abs * scale_y) / target_h

        # Clamp to valid range
        x_new = np.clip(x_new, 0, 1)
        y_new = np.clip(y_new, 0, 1)
        w_new = np.clip(w_new, 1e-6, 1)
        h_new = np.clip(h_new, 1e-6, 1)

        # Validate box still makes sense
        if w_new > 0.001 and h_new > 0.001:
            scaled.append([class_id, x_new, y_new, w_new, h_new])

    return scaled


# ============================================================
# Augmentation
# ============================================================

def random_flip(
    image: np.ndarray,
    labels: list[list[float]],
    p: float = 0.5
) -> tuple[np.ndarray, list[list[float]]]:
    """Random horizontal flip with bbox adjustment."""
    if random.random() < p:
        image = cv2.flip(image, 1)
        new_labels = []
        for label in labels:
            class_id, x, y, w, h = label
            new_labels.append([class_id, 1.0 - x, y, w, h])
        return image, new_labels
    return image, labels


def random_rotate(
    image: np.ndarray,
    labels: list[list[float]],
    max_angle: float = 10.0,
    p: float = 0.3
) -> tuple[np.ndarray, list[list[float]]]:
    """Random rotation with bbox adjustment (approximate)."""
    if random.random() < p:
        angle = random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Approximate bbox rotation (center point rotation)
        new_labels = []
        cos_a = np.cos(np.deg2rad(angle))
        sin_a = np.sin(np.deg2rad(angle))
        for label in labels:
            class_id, x, y, bw, bh = label
            # Rotate center point
            x_centered = x - 0.5
            y_centered = y - 0.5
            x_rot = x_centered * cos_a - y_centered * sin_a + 0.5
            y_rot = x_centered * sin_a + y_centered * cos_a + 0.5
            # Keep width/height same (approximation for small angles)
            new_labels.append([class_id,
                              np.clip(x_rot, 0, 1),
                              np.clip(y_rot, 0, 1),
                              bw, bh])
        return image, new_labels
    return image, labels


def random_brightness_contrast(
    image: np.ndarray,
    brightness_range: tuple[float, float] = (-0.2, 0.2),
    contrast_range: tuple[float, float] = (0.8, 1.2),
    p: float = 0.5
) -> np.ndarray:
    """Random brightness and contrast adjustment."""
    if random.random() < p:
        brightness = random.uniform(*brightness_range)
        contrast = random.uniform(*contrast_range)
        # Image is normalized float32, adjust in normalized space
        image = image * contrast + brightness
        # Clip to reasonable range (about 4 std from mean)
        image = np.clip(image, -4, 4)
    return image


def random_noise(
    image: np.ndarray,
    noise_std_range: tuple[float, float] = (0.0, 0.1),
    p: float = 0.3
) -> np.ndarray:
    """Add Gaussian noise."""
    if random.random() < p:
        noise_std = random.uniform(*noise_std_range)
        noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
        image = image + noise
    return image


def augment_image(
    image: np.ndarray,
    labels: list[list[float]],
    apply_flip: bool = True,
    apply_rotate: bool = True,
    apply_brightness_contrast: bool = True,
    apply_noise: bool = True
) -> tuple[np.ndarray, list[list[float]]]:
    """
    Apply full augmentation pipeline to image and labels.
    """
    if apply_flip:
        image, labels = random_flip(image, labels)
    if apply_rotate:
        image, labels = random_rotate(image, labels)
    if apply_brightness_contrast:
        image = random_brightness_contrast(image)
    if apply_noise:
        image = random_noise(image)
    return image, labels


# ============================================================
# Mosaic / MixUp Augmentation
# ============================================================

def mosaic_augmentation(
    images: list[np.ndarray],
    labels_list: list[list[list[float]]],
    output_size: tuple[int, int] = TARGET_SIZE
) -> tuple[np.ndarray, list[list[float]]]:
    """
    Mosaic augmentation: combine 4 images into one 2x2 grid.
    Particularly useful for small object detection.
    """
    assert len(images) == 4, "Mosaic requires exactly 4 images"
    assert len(labels_list) == 4

    h, w = output_size
    # Create 2x2 mosaic canvas
    mosaic_img = np.zeros((h * 2, w * 2), dtype=np.float32)
    mosaic_labels = []

    # Fixed quadrants: each is h x w
    quadrants = [
        (0, 0, w, h),       # top-left
        (w, 0, w, h),       # top-right
        (0, h, w, h),       # bottom-left
        (w, h, w, h),       # bottom-right
    ]

    for i, (img, lbls) in enumerate(zip(images, labels_list)):
        x1, y1, quad_w, quad_h = quadrants[i]
        x2, y2 = x1 + quad_w, y1 + quad_h

        # Resize image to quadrant size
        img_resized = cv2.resize(img, (quad_w, quad_h))

        # Place in mosaic
        mosaic_img[y1:y2, x1:x2] = img_resized

        # Adjust labels from quadrant coords to mosaic coords
        for label in lbls:
            class_id, x, y, bw, bh = label
            # x, y are normalized [0,1] within quadrant
            # Convert to mosaic coordinates
            x_abs = x * quad_w + x1
            y_abs = y * quad_h + y1
            bw_abs = bw * quad_w
            bh_abs = bh * quad_h

            # Normalize to full mosaic (2w x 2h)
            x_new = x_abs / (w * 2)
            y_new = y_abs / (h * 2)
            bw_new = bw_abs / (w * 2)
            bh_new = bh_abs / (h * 2)

            if bw_new > 0.001 and bh_new > 0.001:
                mosaic_labels.append([class_id, x_new, y_new, bw_new, bh_new])

    # Random crop from mosaic to output size
    max_x = w  # Can crop from x=0 to x=w
    max_y = h  # Can crop from y=0 to y=h
    crop_x = random.randint(0, max_x)
    crop_y = random.randint(0, max_y)

    final_img = mosaic_img[crop_y:crop_y+h, crop_x:crop_x+w]

    # Filter and adjust labels to crop region
    final_labels = []
    for label in mosaic_labels:
        class_id, x, y, bw, bh = label
        # Convert to absolute mosaic coordinates
        x_abs = x * (w * 2)
        y_abs = y * (h * 2)
        bw_abs = bw * (w * 2)
        bh_abs = bh * (h * 2)

        # Check if box overlaps with crop region
        if (x_abs + bw_abs/2 >= crop_x and x_abs - bw_abs/2 <= crop_x + w and
            y_abs + bh_abs/2 >= crop_y and y_abs - bh_abs/2 <= crop_y + h):
            # Clip box to crop region
            x1_clip = max(x_abs - bw_abs/2, crop_x)
            y1_clip = max(y_abs - bh_abs/2, crop_y)
            x2_clip = min(x_abs + bw_abs/2, crop_x + w)
            y2_clip = min(y_abs + bh_abs/2, crop_y + h)

            if x2_clip > x1_clip and y2_clip > y1_clip:
                # Convert back to normalized crop coordinates
                x_final = ((x1_clip + x2_clip) / 2 - crop_x) / w
                y_final = ((y1_clip + y2_clip) / 2 - crop_y) / h
                bw_final = (x2_clip - x1_clip) / w
                bh_final = (y2_clip - y1_clip) / h
                final_labels.append([class_id, x_final, y_final, bw_final, bh_final])

    return final_img, final_labels


def mixup_augmentation(
    image1: np.ndarray,
    labels1: list[list[float]],
    image2: np.ndarray,
    labels2: list[list[float]],
    alpha: float = 0.2
) -> tuple[np.ndarray, list[list[float]]]:
    """
    MixUp augmentation: linear combination of two images and labels.
    """
    lam = np.random.beta(alpha, alpha)
    mixed_img = lam * image1 + (1 - lam) * image2

    # Combine labels (both sets kept)
    mixed_labels = labels1 + labels2

    return mixed_img, mixed_labels


# ============================================================
# Stratified Split
# ============================================================

def get_image_class_counts(
    labels_folder: str,
    image_files: list[str]
) -> dict[str, list[int]]:
    """
    Get class presence counts for each image for stratification.
    Returns dict: image_name -> list of class_ids present.
    """
    labels_path = Path(labels_folder)
    image_classes = {}

    for img_name in image_files:
        stem = Path(img_name).stem
        label_file = labels_path / f"{stem}.txt"
        classes = []
        if label_file.exists():
            for line in label_file.read_text().splitlines():
                if line.strip():
                    class_id = int(line.split()[0])
                    if class_id not in classes:
                        classes.append(class_id)
        image_classes[img_name] = classes

    return image_classes


def create_stratified_split(
    image_files: list[str],
    labels_folder: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> tuple[list[str], list[str], list[str]]:
    """
    Create stratified train/val/test split based on class presence.
    Uses the most frequent class per image for stratification to avoid
    sparse multi-label combinations.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    image_classes = get_image_class_counts(labels_folder, image_files)

    # Use the most common class per image as stratification label
    # For images with no objects, use -1
    stratify_labels = []
    for img in image_files:
        classes = image_classes[img]
        if classes:
            stratify_labels.append(classes[0])  # Use first (most common) class
        else:
            stratify_labels.append(-1)

    # Check if any class has only 1 sample
    from collections import Counter
    class_counts = Counter(stratify_labels)
    min_count = min(class_counts.values())
    
    if min_count < 2:
        # Fall back to random split if stratification not possible
        print("Warning: Some classes have <2 samples, using random split")
        random.seed(random_state)
        shuffled = image_files.copy()
        random.shuffle(shuffled)
        n_train = int(len(shuffled) * train_ratio)
        n_val = int(len(shuffled) * val_ratio)
        return shuffled[:n_train], shuffled[n_train:n_train+n_val], shuffled[n_train+n_val:]

    # First split: train vs (val+test)
    train_files, temp_files = train_test_split(
        image_files,
        train_size=train_ratio,
        stratify=stratify_labels,
        random_state=random_state
    )

    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    temp_stratify = []
    for f in temp_files:
        classes = image_classes[f]
        temp_stratify.append(classes[0] if classes else -1)
    
    # Check temp set stratification
    temp_class_counts = Counter(temp_stratify)
    if min(temp_class_counts.values()) < 2:
        print("Warning: Temp set has classes with <2 samples, using random split for val/test")
        random.seed(random_state + 1)
        random.shuffle(temp_files)
        n_val = int(len(temp_files) * val_ratio_adjusted)
        return train_files, temp_files[:n_val], temp_files[n_val:]

    val_files, test_files = train_test_split(
        temp_files,
        train_size=val_ratio_adjusted,
        stratify=temp_stratify,
        random_state=random_state
    )

    return train_files, val_files, test_files


def save_split(
    train_files: list[str],
    val_files: list[str],
    test_files: list[str],
    output_dir: str
) -> None:
    """Save split file lists to text files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, files in [("train.txt", train_files), ("val.txt", val_files), ("test.txt", test_files)]:
        (output_path / name).write_text("\n".join(files) + "\n")

    print(f"Split saved to {output_dir}:")
    print(f"  Train: {len(train_files)}")
    print(f"  Val:   {len(val_files)}")
    print(f"  Test:  {len(test_files)}")


# ============================================================
# Pipeline
# ============================================================

def preprocess_folder(
    input_folder: str,
    labels_folder: str,
    output_images_folder: str,
    output_labels_folder: str,
    apply_augmentation: bool = False
) -> None:
    """Preprocess GC10-DET images and preserve YOLO annotations."""
    output_images = Path(output_images_folder)
    output_labels = Path(output_labels_folder)

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    images = load_images_from_folder(input_folder)

    for filename, image in images:
        label_path = Path(labels_folder) / f"{Path(filename).stem}.txt"
        labels = load_yolo_labels(label_path)

        # Original image size for bbox scaling
        orig_h, orig_w = image.shape[:2]

        processed = preprocess_image(image)
        scaled_labels = scale_bboxes(labels, (orig_h, orig_w))

        if apply_augmentation:
            processed, scaled_labels = augment_image(processed, scaled_labels)

        # Save processed image (as normalized float32 PNG for training)
        # For visualization, save denormalized version
        image_output = output_images / f"{Path(filename).stem}.npy"
        np.save(image_output, processed.astype(np.float32))

        # Also save denormalized PNG for inspection
        vis_output = output_images / f"{Path(filename).stem}.png"
        vis_img = denormalize_image(processed)
        cv2.imwrite(str(vis_output), vis_img)

        label_output = output_labels / f"{Path(filename).stem}.txt"
        save_yolo_labels(scaled_labels, label_output)

    print(f"Processed {len(images)} images.")


def preprocess_and_split(
    input_folder: str,
    labels_folder: str,
    output_root: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> None:
    """
    Full pipeline: preprocess images, create stratified split,
    and organize into train/val/test folders.
    """
    output_root = Path(output_root)
    images_folder = output_root / "images"
    output_labels_folder = output_root / "labels"
    input_labels_folder = Path(labels_folder)

    # Get all image files
    all_images = load_images_from_folder(input_folder)
    image_files = [name for name, _ in all_images]

    # Create stratified split using INPUT labels folder
    train_files, val_files, test_files = create_stratified_split(
        image_files, str(input_labels_folder), train_ratio, val_ratio, test_ratio, random_state
    )

    # Save split info
    save_split(train_files, val_files, test_files, output_root / "splits")

    # Process each split
    for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        split_img_dir = images_folder / split_name
        split_lbl_dir = output_labels_folder / split_name

        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for filename in split_files:
            # Find the image data
            image_data = None
            for name, img in all_images:
                if name == filename:
                    image_data = img
                    break

            if image_data is None:
                continue

            label_path = input_labels_folder / f"{Path(filename).stem}.txt"
            labels = load_yolo_labels(label_path)

            orig_h, orig_w = image_data.shape[:2]
            processed = preprocess_image(image_data)
            scaled_labels = scale_bboxes(labels, (orig_h, orig_w))

            # Apply augmentation only to training set
            if split_name == "train":
                processed, scaled_labels = augment_image(processed, scaled_labels)

            # Save
            np.save(split_img_dir / f"{Path(filename).stem}.npy", processed.astype(np.float32))
            vis_img = denormalize_image(processed)
            cv2.imwrite(str(split_img_dir / f"{Path(filename).stem}.png"), vis_img)
            save_yolo_labels(scaled_labels, split_lbl_dir / f"{Path(filename).stem}.txt")

    print("Preprocessing and split complete.")


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GC10-DET preprocessing pipeline")
    parser.add_argument("--input", required=True, help="Input images folder")
    parser.add_argument("--labels", required=True, help="YOLO labels folder")
    parser.add_argument("--output", required=True, help="Output root folder")
    parser.add_argument("--split", action="store_true", help="Create train/val/test split")
    parser.add_argument("--augment", action="store_true", help="Apply augmentation")
    args = parser.parse_args()

    if args.split:
        preprocess_and_split(args.input, args.labels, args.output)
    else:
        preprocess_folder(args.input, args.labels,
                         f"{args.output}/images", f"{args.output}/labels",
                         apply_augmentation=args.augment)