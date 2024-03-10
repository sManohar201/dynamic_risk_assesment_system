"""
    This script automates ML model scoring, monitoring, and re-deploying as and 
    when necessary.
    Author: Sabari Manohar
    Date:   March, 2024
"""

import training
import deployment
import scoring
import diagnostics
import re
import reporting
import logging
import os
import sys
from copy import deepcopy
import ingestion
import pandas as pd
from sklearn import metrics

from config_load import PROD_DEPLOYMENT_PATH, TEST_DATA_PATH, INPUT_FOLDER_PATH, OUTPUT_DATA_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def fullprocess():
    ##################Check and read new data
    logging.info("Checking and read new data")

    #first, read ingestedfiles.txt
    with open(os.path.join(PROD_DEPLOYMENT_PATH, "ingestedfiles.txt")) as file:
        ingested_files = {line.strip('\n') for line in file.readlines()[1:]}

    # second, determine whether the source data folder has files that aren't
    # listed in ingestedfiles.txt
    input_source_files = set(os.listdir(INPUT_FOLDER_PATH))

    ##################Deciding whether to proceed, part 1
    # if you found new data, you should proceed. otherwise, do end the process
    # here
    if len(input_source_files.difference(ingested_files)) == 0:
        logging.info("No new data found")
        return None
    # Ingesting new data
    logging.info("Ingesting new data")
    ingestion.merge_multiple_dataframe()

    ##################Checking for model drift
    logging.info("Checking for model drift")

    # check whether the score from the deployed model is different from the
    # score from the model that uses the newest ingested data
    with open(os.path.join(PROD_DEPLOYMENT_PATH, "latestscore.txt")) as file:
        deployed_model_score = re.findall(r'\d*\.?\d+', file.read())[0]
        deployed_model_score = float(deployed_model_score)

    final_df = pd.read_csv(os.path.join(OUTPUT_DATA_PATH, 'finaldata.csv'))
    y_predicted = diagnostics.model_predictions(deepcopy(final_df))
    y_gt = final_df.pop('exited')
    x_df = final_df.drop(['corporation'], axis=1)

    new_f1_score = metrics.f1_score(y_gt.values, y_predicted)

    ##################Deciding whether to proceed, part 2
    logging.info(f"Deployed model score = {deployed_model_score}")
    logging.info(f"New F1 score = {new_f1_score}")

    # if you found model drift, you should proceed. otherwise, do end the
    # process here
    if(new_f1_score >= deployed_model_score):
        logging.info("Model drift : None")
        return None

    # training the model again
    logging.info("Re-training the model")
    training.train_model()
    logging.info("Re-scoring the model")
    scoring.score_model()

    ##################Re-deployment
    logging.info("Re-deployment of the model")

    #if you found evidence for model drift, re-run the deployment.py script
    deployment.store_model_into_pickle()

    ##################Diagnostics and reporting
    logging.info("Run diagnostics and reporting for the newly deployed model")

    # run diagnostics.py and reporting.py for the re-deployed model
    reporting.plot_confusion_matrix()
    os.system("python src/apicalls.py")

if __name__ == '__main__':
    logging.info("run fullprocess")
    fullprocess()







