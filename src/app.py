"""
    Set up api calls for reports, performance of the model, and diagnostics.
    Author: Sabari Manohar
    Date:   March, 2024
"""
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import diagnostics
import logging
from scoring import score_model
import json
import os
import sys

from config_load import OUTPUT_DATA_PATH, MODEL_PATH, TEST_DATA_PATH, PROD_DEPLOYMENT_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

prediction_model = None

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    """
    Calculate model predictions
    """
    #call the prediction function you created in Step 3
    logging.info('Predict model parameters')
    y_pred = diagnostics.model_predictions()
    #add return value for prediction outputs
    return str(y_pred)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring_stats():        
    """
        F1 score of the deployed model on test data.
    """
    #check the score of the deployed model
    logging.info('get scoring statistics')
    model_path = os.path.join(PROD_DEPLOYMENT_PATH, 'trainedmodel.pkl')
    score = score_model()
    #add return value (a single F1 score number)
    return str(score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():        
    """ 
        Get mean, median, and mode for each column
    """
    #check means, medians, and modes for each column
    logging.info('get summary statistics')
    col_stats = diagnostics.dataframe_summary()
    #return a list of all calculated summary statistics
    return str(col_stats)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diag_stats():        
    '''
    Diagnostics for the model
    '''
    logging.info("get diagnostics statistics")
    #check timing and percent NA values
    missing = diagnostics.missing_data()
    time_elasted = diagnostics.execution_time()
    out_pkg_list = diagnostics.outdated_packages_list()
    diagnostics_result = [missing, time_elasted, out_pkg_list]
    return diagnostics_result 

if __name__ == "__main__":    
    logging.info("Run reporting api call.")
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
