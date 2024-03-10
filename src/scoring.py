"""
    Model scoring: load the trained model and score it against the test data. 
    Author: Sabari Manohar
    Date:   March, 2024
"""

import os
import pickle
import sys
import logging

import pandas as pd
from sklearn import metrics

from config_load import TEST_DATA_PATH, MODEL_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def score_model():
    """
    This function should take a trained model, load test data, and calculate 
    an F1 score for the model relative to the test data
    it should write the result to the latestscore.txt file.
    """
    logging.info("Loading the testdata.csv")
    test_data = pd.read_csv(os.path.join(TEST_DATA_PATH, 'testdata.csv'))

    logging.info("Loading the trained model pkl file")
    with open(os.path.join(MODEL_PATH, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    logging.info("Preparing the test data")
    y_gt = test_data.pop('exited')
    x_test_data = test_data.drop(['corporation'], axis=1)

    logging.info("validation with the test data")
    y_prediction = model.predict(x_test_data)
    f1 = metrics.f1_score(y_gt, y_prediction)

    logging.info("Saving scores to text file")
    with open(os.path.join(MODEL_PATH, 'latestscore.txt'), 'w', encoding='utf-8') as file:
        file.write(f"f1 score for the model = {f1}")

    return str(f1)

if __name__ == '__main__':
    score_model()
