from pathlib import Path
import cv2
import numpy as np


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

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            print(f"Warning: cannot read image: {path}")
            continue

        images.append((path.name, image))

    return images


# ============================================================
# Preprocessing
# ============================================================

def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert image to RGB color space.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(
    image: np.ndarray,
    size: tuple[int, int] = TARGET_SIZE
) -> np.ndarray:
    """
    Resize image to the target size.
    """
    return cv2.resize(image, size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image.

    Placeholder for future preprocessing:
    - pixel normalization
    - contrast normalization
    - histogram equalization
    - etc.
    """
    # TODO
    pass


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess a GC10-DET image.

    Steps:
    1. Convert image to RGB.
    2. Resize image to the target size.

    Note:
        Bounding box annotations must also be resized
        using the same scale factors.
    """
    image = convert_to_rgb(image)
    image = resize_image(image, TARGET_SIZE)

    return image


# ============================================================
# Saving
# ============================================================

def save_images(
    images: list[tuple[str, np.ndarray]],
    output_folder: str
) -> None:
    """
    Save all processed images to a folder.
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for filename, image in images:
        output_file = output_path / f"{Path(filename).stem}.png"

        # OpenCV expects BGR rather than RGB
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        success = cv2.imwrite(str(output_file), image)

        if not success:
            print(f"Warning: could not save image: {output_file}")


def save_images(
    images: list[np.ndarray],
    output_folder: str
) -> None:
    """
    Save all processed images to a folder.
    """
    # TODO
    pass


# ============================================================
# Pipeline
# ============================================================

def preprocess_folder(
    input_folder: str,
    labels_folder: str,
    output_images_folder: str,
    output_labels_folder: str
) -> None:
    """Preprocess GC10-DET images and preserve YOLO annotations."""
    output_images = Path(output_images_folder)
    output_labels = Path(output_labels_folder)

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    images = load_images_from_folder(input_folder)

    for filename, image in images:
        label_path = Path(labels_folder) / f"{Path(filename).stem}.txt"

        # Validate labels before saving anything for this image.
        labels = validate_yolo_labels(label_path)

        processed = preprocess_image(image)

        image_output = output_images / f"{Path(filename).stem}.png"
        label_output = output_labels / f"{Path(filename).stem}.txt"

        # OpenCV expects BGR when writing.
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

        if not cv2.imwrite(str(image_output), processed_bgr):
            raise IOError(f"Could not save image: {image_output}")

        # Empty lists produce empty label files for negative examples.
        label_output.write_text(
            "\n".join(labels) + ("\n" if labels else "")
        )

    print(f"Processed {len(images)} images.")

# ============================================================
# Future preprocessing functions
# ============================================================

def remove_noise(image: np.ndarray) -> np.ndarray:
    """Remove image noise."""
    return cv2.GaussianBlur(
        image,
        ksize=(3, 3),
        sigmaX=0
    )


def adjust_contrast(image: np.ndarray) -> np.ndarray:
    """Adjust image contrast."""
    alpha = 1.2  # Contrast factor
    beta = 0     # Brightness offset

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def normalize_brightness(image: np.ndarray) -> np.ndarray:
    """Normalize image brightness."""
    return cv2.normalize(
        image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )


def crop_image(image: np.ndarray) -> np.ndarray:
    """Crop image."""
    # TODO
    pass


def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply data augmentation."""
    # TODO
    pass


# ============================================================
# YOLO related
# ============================================================

def validate_yolo_labels(
    label_path: Path
) -> list[str]:
    """Read and validate YOLO bounding-box annotations."""
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    valid_lines = []

    for line_number, line in enumerate(
        label_path.read_text().splitlines(), start=1
    ):
        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_number}: expected 5 values"
            )

        try:
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
        except ValueError as exc:
            raise ValueError(
                f"{label_path}:{line_number}: invalid annotation"
            ) from exc

        if not 0 <= class_id < NUM_CLASSES:
            raise ValueError(
                f"{label_path}:{line_number}: invalid class ID {class_id}"
            )

        if not all(
            np.isfinite(v) and 0 <= v <= 1
            for v in (x, y, width, height)
        ):
            raise ValueError(
                f"{label_path}:{line_number}: coordinates must be in [0, 1]"
            )

        if width <= 0 or height <= 0:
            raise ValueError(
                f"{label_path}:{line_number}: box dimensions must be positive"
            )

        if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6:
            raise ValueError(
                f"{label_path}:{line_number}: box exceeds image bounds"
            )

        if y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
            raise ValueError(
                f"{label_path}:{line_number}: box exceeds image bounds"
            )

        valid_lines.append(
            f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}"
        )

    return valid_lines


# ============================================================
# Example
# ============================================================

# if __name__ == "__main__":
#     preprocess_folder(
#         input_folder="dataset/images/train",
#         labels_folder="dataset/labels/train",
#         output_images_folder="processed/images/train",
#         output_labels_folder="processed/labels/train",
#     )