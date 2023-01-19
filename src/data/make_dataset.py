# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import pandas as pd
import shutil


def create_dataframe():  # input_filepath, output_filepath
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    dir_root = Path(__file__).parent.parent.parent
    dataset_raw_images = Path(dir_root, "./data/raw/food-101/images")
    dataset_raw_labels = Path(dir_root, "./data/raw/food-101/meta/train.json")
    dataset_processed = Path(dir_root, "./data/food-101/processed/*")

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    df = pd.read_json(dataset_raw_labels)
    df = pd.DataFrame(df)

    # final_df['image_path'] = df1.apply( lambda row: (str(dataset_raw_images) + "/" + str(df1['churros'])), axis=0)
    # final_df['label'] = df1.apply( lambda row: 'churros', axis=1)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())
    create_dataframe()
