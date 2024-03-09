"""
    This script implements function for diagnostic capabilities. 
    Author: Sabari Manohar
    Date:   March, 2024
"""
import os
import sys
import timeit
import pickle
import logging
import subprocess
import pandas as pd
import numpy as np

from config_load import OUTPUT_DATA_PATH, TEST_DATA_PATH, PROD_DEPLOYMENT_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def model_predictions():
    """
    read the deployed model and a test dataset, calculate predictions
    """
    logging.info("Reading testdata")
    data_path = os.path.join(TEST_DATA_PATH, 'testdata.csv')
    dataset = pd.read_csv(data_path)
    # take model
    logging.info("Predict with model")
    _ = dataset.pop('exited')
    x = dataset.drop(['corporation'], axis=1)
    model_path = os.path.join(PROD_DEPLOYMENT_PATH, 'trainedmodel.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(x)
    #return value should be a list containing all predictions
    return y_pred

def dataframe_summary():
    """ 
    calculate summary statistics here
    collect dataset
    """
    datasetpath = os.path.join(OUTPUT_DATA_PATH, 'finaldata.csv')
    dataset = pd.read_csv(datasetpath)
    # select numeric columns
    numeric_col_index = np.where(dataset.dtypes != object)[0]
    print(numeric_col_index)
    numeric_col = dataset.columns[numeric_col_index].tolist()
    # compute statistics per numeric column
    means = dataset[numeric_col].mean(axis=0).tolist()
    medians = dataset[numeric_col].median(axis=0).tolist()
    stddevs = dataset[numeric_col].std(axis=0).tolist()

    statistics = means
    statistics.extend(medians)
    statistics.extend(stddevs)
    #return value should be a list containing all summary statistics
    return statistics

def missing_data():
    """
    calculate missing data information 
    """
    test_data = pd.read_csv(os.path.join(TEST_DATA_PATH, 'testdata.csv'))

    results = []
    for column in test_data.columns:
        na = test_data[column].isna().sum()
        non_na = test_data[column].count()
        total = non_na + na

        results.append([column, str(int((na/total)*100))+"%"])
    return results

def execution_time():
    """
    calculate timing of training.py and ingestion.py
    """
    results = []
    for process in ['training.py', 'ingestion.py']:
        begin_time = timeit.default_timer()
        process = f"./src/{process}"
        os.system(f"python3 {process}")
        time_total = timeit.default_timer() - begin_time
        results.append([process, time_total])

    #return a list of 2 timing values in seconds
    return results

def outdated_packages_list():
    """
     Check outdated dependencies
    """
    #get a list of updated packages
    outdated_packages = subprocess.check_output([
                'pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    return str(outdated_packages)

if __name__ == '__main__':
    print(model_predictions())
    print(dataframe_summary())
    print(execution_time())
    print(outdated_packages_list())
