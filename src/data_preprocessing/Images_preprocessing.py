from pathlib import Path
import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TARGET_SIZE = (256, 256)
TARGET_FORMAT = "PNG"


# ============================================================
# Loading
# ============================================================

def load_images_from_folder(folder: str) -> list[np.ndarray]:
    """
    Load all supported images from a folder.

    Parameters
    ----------
    folder : str
        Path to the folder with images.

    Returns
    -------
    list[np.ndarray]
        Loaded images.
    """
    folder_path = Path(folder)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = []

    for path in folder_path.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        image = cv2.imread(str(path))

        if image is None:
            print(f"Warning: could not read image: {path}")
            continue

        images.append(image)

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
    Apply all preprocessing steps to one image.
    """
    image = convert_to_rgb(image)
    image = resize_image(image, TARGET_SIZE)
    image = remove_noise(image)
    image = normalize_brightness(image)
    image = adjust_contrast(image)

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
    output_folder: str
) -> None:
    """
    Load, preprocess and save all images from a folder.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Folder not found: {input_folder}")

    for path in input_path.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        image = cv2.imread(str(path))

        if image is None:
            print(f"Warning: could not read image: {path}")
            continue

        processed_image = preprocess_image(image)

        output_file = output_path / f"{path.stem}.png"

        # RGB -> BGR because OpenCV writes images in BGR order
        processed_image = cv2.cvtColor(
            processed_image,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(str(output_file), processed_image)


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
