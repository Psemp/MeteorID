import os
import glob
import concurrent.futures

from tqdm import tqdm
from scripts.image_processing import process_image


def process_batch(image_directory: str, save_directory: str, max_workers: int = 4) -> None:
    # Ensure save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    image_paths = glob.glob(os.path.join(image_directory, '*.*'))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, image_path, save_directory) for image_path in image_paths]

    # Wrap the futures with tqdm to display progress
    for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        pass

    for future in futures:
        try:
            future.result()
        except Exception as e:
            print(f"An error occurred: {e}")
