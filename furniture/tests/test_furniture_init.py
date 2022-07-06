import os

import imageio
import numpy as np
from tqdm import tqdm

from config import create_parser
from env.furniture_baxter import FurnitureBaxterEnv
from env.models import furniture_name2id, furniture_names
from env.models.base import RandomizationError
from util.logger import logger


def test_baxter_furniture_reset(take_picture=False, img_folder="test_imgs"):
    """
    Goes through all furniture xml on baxter.
    Instantiates the scene and takes a snapshot
    Makes sure that RandomizationError does not happen
    Args:
        take_picture: if you want to snapshot the furniture initialization
        img_folder: folder for the snapshots
    """
    if take_picture:
        os.makedirs(img_folder, exist_ok=True)
    parser = create_parser(env="FurnitureBaxterEnv")
    config, unparsed = parser.parse_known_args()
    config.unity = True

    env = FurnitureBaxterEnv(config)
    pbar = tqdm(furniture_names)
    failed_furn = []
    num_failed = 0
    for furn_name in pbar:
        pbar.set_description(f"Testing {furn_name}", True)
        furn_id = furniture_name2id[furn_name]
        try:
            env.reset(furn_id)
        except (RandomizationError, Exception) as e:
            logger.error(e)
            logger.error(f"Baxter reset test failed at {furn_name}")
            failed_furn.append(furn_name)
            num_failed += 1
        img = (env.render("rgb_array")[0] * 255).astype(np.uint8)
        if take_picture:
            img_path = os.path.join(img_folder, f"baxter_{furn_name}.png")
            imageio.imwrite(img_path, img)

    num_success = len(furniture_names) - num_failed
    print(f"Test Summary:" + "-" * 80)
    print(f"Successfully Reset {num_success} furnitures")
    print(f"Failed {num_failed} furnitures")

    print("Failed Furnitures" + "-" * 80)
    for failed in failed_furn:
        print(failed)


if __name__ == "__main__":
    test_baxter_furniture_reset(True)
