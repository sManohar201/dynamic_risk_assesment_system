"""
    Deployment: Load the trained model and associated metrics and save it for deployment. 
    Author: Sabari Manohar
    Date:   March, 2024
"""
import sys
import os
import logging
import shutil

from config_load import OUTPUT_DATA_PATH, PROD_DEPLOYMENT_PATH, MODEL_PATH


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def store_model_into_pickle():
    """_summary_
    Copy the latest pickle file, the latestscore.txt value, 
    and the ingestfiles.txt file into the deployment directory.
    """
    logging.info("Deploy trained model to production env")
    logging.info("Copying trainedmodel.pkl, ingestfiles.txt and latestscore.txt")

    ingested_file_path = os.path.join(OUTPUT_DATA_PATH, 'ingestedfiles.txt')
    shutil.copy2(ingested_file_path, PROD_DEPLOYMENT_PATH)

    trained_model_path = os.path.join(MODEL_PATH, 'trainedmodel.pkl')
    shutil.copy2(trained_model_path, PROD_DEPLOYMENT_PATH)

    score_file_path = os.path.join(MODEL_PATH, 'latestscore.txt')
    shutil.copy2(score_file_path, PROD_DEPLOYMENT_PATH)


if __name__ == '__main__':
    logging.info("Run deployment.py")
    store_model_into_pickle()
