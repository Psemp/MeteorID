import numpy as np

from PIL import Image


def rm_scale(image: Image, crop_percentage: float = 0.075) -> Image:

    # Adjust box dimensions
    width, height = image.size
    left = int(width * crop_percentage)
    upper = int(height * crop_percentage)
    right = int(width * (1 - crop_percentage))
    lower = int(height * (1 - crop_percentage))

    crop_box = (left, upper, right, lower)
    cropped_image = image.crop(crop_box)

    return cropped_image


def resize_center(image: Image, output_size: tuple = (400, 400)) -> Image:
    width, height = image.size

    new_width = min(width, height)
    left = (width - new_width) / 2
    top = (height - new_width) / 2
    right = (width + new_width) / 2
    bottom = (height + new_width) / 2

    # crop @ center
    cropped_image = image.crop((left, top, right, bottom))

    # resize at output_size resolution
    resized_image = cropped_image.resize(output_size, Image.Resampling.LANCZOS)

    return resized_image


def crop_strips(image: Image, strip_width: int = 800) -> tuple:
    width, height = image.size

    left_crop = strip_width
    right_crop = width - strip_width

    # Crop the left and right strips
    left_strip = image.crop((0, 0, left_crop, height))
    right_strip = image.crop((right_crop, 0, width, height))

    return left_strip, right_strip


def get_average_grid_px(image: Image) -> list:
    width, height = image.size
    grid_size = 3  # 3x3 grid

    # Ensure the image is square
    assert width == height, "The image must be square."

    # Step size determination
    step_size = width // grid_size

    grid_avg_values = []

    for i in range(grid_size):
        for j in range(grid_size):
            # Grid cell's BBOX
            left = i * step_size
            top = j * step_size
            right = left + step_size
            bottom = top + step_size

            # Crop the grid
            grid_cell = image.crop((left, top, right, bottom))

            # Averages the pixel value of each cell
            np_grid_cell = np.array(grid_cell)
            avg_r = np.mean(np_grid_cell[:, :, 0])
            avg_g = np.mean(np_grid_cell[:, :, 1])
            avg_b = np.mean(np_grid_cell[:, :, 2])

            # Append the average values as a tuple
            grid_avg_values.append((avg_r, avg_g, avg_b))

    return grid_avg_values


def get_crops_staircase_pattern(strip: Image, crop_size: tuple = (400, 400)) -> list:
    width, height = strip.size
    crop_width, crop_height = crop_size

    crops = []

    # Iterate over the height of the strip
    for y in range(0, height - crop_height + 1, crop_height):
        # Top left crop
        left_crop = strip.crop((0, y, crop_width, y + crop_height))
        crops.append(left_crop)

        # Top right crop (aligned to the right edge)
        if width > crop_width:
            right_crop = strip.crop((width - crop_width, y, width, y + crop_height))
            crops.append(right_crop)

    # Handle the last line if there's at least 100px remaining
    remaining_height = height % crop_height
    if remaining_height >= 100:
        y = height - crop_height
        left_crop = strip.crop((0, y, crop_width, height))
        crops.append(left_crop)

        if width > crop_width:
            right_crop = strip.crop((width - crop_width, y, width, height))
            crops.append(right_crop)

    return crops


def calculate_difference(original_values: list, crop_values: list):
    # Sum of absolute differences across RGB channels for each grid point
    return np.sum(np.abs(np.array(original_values) - np.array(crop_values)))


def get_best_crop(original_image: Image, strip: Image, crop_size: tuple = (400, 400)) -> Image:

    pixel_average_original = get_average_grid_px(image=original_image)

    crops = get_crops_staircase_pattern(strip=strip, crop_size=crop_size)

    best_crop = None
    best_delta = float("inf")

    for crop in crops:
        delta = calculate_difference(
            original_values=pixel_average_original,
            crop_values=get_average_grid_px(image=crop)
            )
        if delta < best_delta:
            best_delta = delta
            best_crop = crop

    return best_crop


def process_image(image_path: str, save_directory: str) -> None:
    # crop to remove scale :
    image = Image.open(fp=image_path)
    image_name = image_path.split("/")[-1].split(".")[0]
    image = rm_scale(image=image)

    # center main image
    centered_image = resize_center(image=image)

    left_strip, right_strip = crop_strips(image=image)

    best_left = get_best_crop(original_image=centered_image, strip=left_strip)
    best_right = get_best_crop(original_image=centered_image, strip=right_strip)

    rotate_left = np.random.choice(np.arange(0, 360, 90))
    rotate_right = np.random.choice(np.arange(0, 360, 90))

    if rotate_left != 0:
        best_left = best_left.rotate(rotate_left)

    if rotate_right != 0:
        best_right = best_right.rotate(rotate_right)

    # Saving :
    if save_directory[-1] != "/":
        save_directory = save_directory + "/"

    # Save the processed images
    best_left.save(fp=f"{save_directory}{image_name}_l.jpg", format="JPEG")
    best_right.save(fp=f"{save_directory}{image_name}_r.jpg", format="JPEG")
    centered_image.save(fp=f"{save_directory}{image_name}_c.jpg", format="JPEG")
