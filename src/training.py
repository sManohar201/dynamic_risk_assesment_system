"""
    This script takes the ingested dataset and trains a regression model.
    Author: Sabari Manohar
    Date:   March, 2024
"""

import os
import sys
import logging
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

from config_load import MODEL_PATH, OUTPUT_DATA_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def train_model():
    """_summary_
        Train a logistic regression model from accumulated data and 
        export model as pkl file. 
    """

    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    logging.info("Loading and preparing finaldata.csv")
    data_ = pd.read_csv(os.path.join(OUTPUT_DATA_PATH, 'finaldata.csv'))
    y_data = data_.pop('exited')
    x_data = data_.drop(['corporation'], axis=1)

    #fit the logistic regression to your data
    logging.info("Training model")
    model.fit(x_data, y_data)
    #write the trained model to your workspace in a file called trainedmodel.pkl
    savingpath = os.path.join(MODEL_PATH, 'trainedmodel.pkl')
    with open(savingpath, 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    logging.info("Running training.py")
    train_model()
