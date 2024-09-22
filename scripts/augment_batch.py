import os
import concurrent.futures
from tqdm import tqdm
from scripts.image_augment import augment_image
import pandas as pd


def process_image_task(args):
    """
    Function to process a single image.
    :param args: Tuple containing (image_path, output_dir, box_size, factor)
    """
    image_path, output_dir, box_size, factor = args
    augment_image(image_path, output_dir, box_size, factor)


def augment_mp():
    df = pd.read_pickle(filepath_or_buffer="../data/work_met_img_type_2.pkl")
    # Factors dictionary
    factors = {
        "H5": 1,
        "L6": 1,
        "H6": 2,
        "LL5": 4,
        "H5-6": 4,
        "LL6": 4,
        "L5": 4
    }

    # Save directory
    save_dir = "../data/processed_images/"
    os.makedirs(save_dir, exist_ok=True)

    # Prepare tasks
    tasks = []
    for index, row in df.iterrows():
        mtype = row["mtype"]
        factor = factors.get(mtype, 0)  # Default factor is 0 if mtype not in factors
        image_names = row["images"]
        if not isinstance(image_names, list):
            # If the "images" column contains a single image name, convert it to a list
            image_names = [image_names]
        for image_name in image_names:
            image_path = f"../imgs/{image_name}"
            output_dir = save_dir  # Modify if you want to organize outputs differently
            box_size = (400, 400)
            tasks.append((image_path, output_dir, box_size, factor))

    # Process images in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image_task, task) for task in tasks]

        # Wrap the futures with tqdm to display progress
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

        # Check for exceptions
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    augment_mp()
