"""
    The file contains code for reading the dataset and compiling the dataset. 

    Author: Sabari Manohar
    Date:   March, 2024
"""

import os
from datetime import datetime
import logging
import sys

import pandas as pd

import config_load

# initiate a logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def merge_multiple_dataframe():
    """_summary_
       Check for datasets, compile them together, and write to an output file.
    """
    df_final = pd.DataFrame()
    ingested_file_names = []

    # use lazy string formating in logging functions
    message_ = str(config_load.INPUT_FOLDER_PATH)
    logging.info("Reading from %s" % message_)

    for file in os.listdir(config_load.INPUT_FOLDER_PATH):
        file_path = os.path.join(config_load.INPUT_FOLDER_PATH, file)
        tmp = pd.read_csv(file_path)
        ingested_file_names.append(file)

        df_final = pd.concat([df_final, tmp], axis=0)

    logging.info("Dropping duplicates")
    df_final.drop_duplicates(inplace=True)

    logging.info("Saving ingested dataframe")
    with open(os.path.join(config_load.OUTPUT_DATA_PATH, 'ingestedfiles.txt'), "w") as file:
        file.write(f"Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        file.write("\n".join(ingested_file_names))

    logging.info("Saving ingested data")
    df_final.to_csv(os.path.join(config_load.OUTPUT_DATA_PATH, 'finaldata.csv'), index=False)


if __name__ == '__main__':
    logging.info("Running ingestion.py")
    merge_multiple_dataframe()
