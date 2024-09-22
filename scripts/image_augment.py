import os
import random
from PIL import Image
from typing import Tuple
from scripts.image_processing import rm_scale


def augment_image(image_path: str, output_dir: str, box_size: Tuple[int, int] = (400, 400), factor: int = 0):
    """
    Runs image augmentation methods on an image multiple times based on the factor.
    :param image_path: Path to the input image.
    :param output_dir: Directory where augmented images will be saved.
    :param box_size: Desired output image size as a tuple (width, height).
    :param factor: Number of times to apply each augmentation method.
    """
    methods = ["Flip", "Crop", "Rotate", "Scaling"]

    # Open the image using Pillow
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error: Unable to open image at {image_path}. Exception: {e}")
        return

    image = rm_scale(image=image)
    # If factor is 0, skip augmentation
    if factor == 0:
        return

    for method in methods:
        for i in range(factor):
            augmented_image = image.copy()
            if method == "Flip":
                augmented_image = flip_image(augmented_image)
                method_suffix = "_f"
            elif method == "Crop":
                augmented_image = crop_image(augmented_image)
                method_suffix = "_c"
            elif method == "Rotate":
                augmented_image = rotate_image(augmented_image)
                method_suffix = "_ro"
            elif method == "Scaling":
                augmented_image = scale_image(augmented_image)
                method_suffix = "_s"

            # Resize to box_size before saving
            augmented_image = augmented_image.resize(box_size, Image.Resampling.LANCZOS)

            # Save the augmented image
            image_name, image_ext = os.path.splitext(os.path.basename(image_path))
            augmented_image_name = f"{image_name}{method_suffix}{i}{image_ext}"
            augmented_image_path = os.path.join(output_dir, augmented_image_name)
            augmented_image.save(augmented_image_path)


def flip_image(image):
    # Randomly flip horizontally or vertically
    flip_method = random.choice(['horizontal', 'vertical'])
    if flip_method == 'horizontal':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return image.transpose(Image.FLIP_TOP_BOTTOM)


def crop_image(image):
    width, height = image.size
    # Random crop dimensions (80% to 100% of the original size)
    crop_scale = random.uniform(0.8, 1.0)
    new_width = int(width * crop_scale)
    new_height = int(height * crop_scale)
    # Random top-left corner for cropping
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    right = left + new_width
    bottom = top + new_height
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


def rotate_image(image):
    # Rotate between -30 and 30 degrees
    angle = random.uniform(-30, 30)
    rotated_image = image.rotate(angle, resample=Image.BICUBIC, expand=True)
    return rotated_image


def scale_image(image):
    width, height = image.size
    scale_factor = random.uniform(0.8, 1.2)  # Scale between 80% and 120%
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return scaled_image
