"""
    This script calls the api endpoints through python functions. 
    Author: Sabari Manohar  
    Date: March, 2024
"""
import requests
import logging
import os
import sys

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

from config_load import TEST_DATA_PATH, MODEL_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logging.info(
    "Request post /prediction for %s" % os.path.join(TEST_DATA_PATH, 'testdata.csv'))
response_prediction = requests.post(
    f'{URL}/prediction',
    json={
        'filepath': os.path.join(TEST_DATA_PATH, 'testdata.csv')}).text

logging.info("Request get /scoring")
response_score = requests.get(f'{URL}/scoring').text

logging.info("Request get /diagnostics")
response_diagnostics = requests.get(f'{URL}/diagnostics').text

logging.info("Request get /summarystats")
response_stats = requests.get(f'{URL}/summarystats').text

logging.info("Generating report in a text file format")
with open(os.path.join(MODEL_PATH, 'apireturns.txt'), 'w') as file:
    file.write('Ingested Data info\n\n')
    file.write('Statistics Summary info\n')
    file.write(response_stats)
    file.write('\nDiagnostics Summary info\n')
    file.write(response_diagnostics)
    file.write('\n\nTest Data info\n\n')
    file.write('Model Predictions\n')
    file.write(response_prediction)
    file.write('\nModel Score info\n')
    file.write(response_score)