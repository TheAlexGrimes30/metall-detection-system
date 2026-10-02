from pathlib import Path
from PIL import Image


# ============================================================
# Configuration
# ============================================================

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TARGET_SIZE = (256, 256)
TARGET_FORMAT = "PNG"


# ============================================================
# Loading
# ============================================================

def load_images_from_folder(folder: str) -> list[Image.Image]:
    """
    Load all supported images from a folder.

    Parameters
    ----------
    folder : str
        Path to the folder with images.

    Returns
    -------
    list[Image.Image]
        Loaded images.
    """
    # TODO:
    # - check that folder exists
    # - find image files
    # - load images
    # - handle corrupted files
    pass


# ============================================================
# Preprocessing
# ============================================================

def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert image to RGB color space.
    """
    # TODO
    pass


def resize_image(
    image: Image.Image,
    size: tuple[int, int] = TARGET_SIZE
) -> Image.Image:
    """
    Resize image to the target size.
    """
    # TODO
    pass


def normalize_image(image: Image.Image) -> Image.Image:
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


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Apply all preprocessing steps to one image.
    """
    # TODO:
    # image = convert_to_rgb(image)
    # image = resize_image(image)
    # image = normalize_image(image)
    pass


# ============================================================
# Saving
# ============================================================

def save_image(
    image: Image.Image,
    output_path: str,
    image_format: str = TARGET_FORMAT
) -> None:
    """
    Save processed image in the target format.
    """
    # TODO
    pass


def save_images(
    images: list[Image.Image],
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
    # TODO:
    # 1. Load images
    # 2. Preprocess each image
    # 3. Save processed images
    pass


# ============================================================
# Future preprocessing functions
# ============================================================

def remove_noise(image: Image.Image) -> Image.Image:
    """Remove image noise."""
    # TODO
    pass


def adjust_contrast(image: Image.Image) -> Image.Image:
    """Adjust image contrast."""
    # TODO
    pass


def normalize_brightness(image: Image.Image) -> Image.Image:
    """Normalize image brightness."""
    # TODO
    pass


def crop_image(image: Image.Image) -> Image.Image:
    """Crop image."""
    # TODO
    pass


def augment_image(image: Image.Image) -> Image.Image:
    """Apply data augmentation."""
    # TODO
    pass
