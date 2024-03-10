"""
    Reporting: Generate PDF report with confusion matrix and other important 
    metrics. 
    Author: Sabari Manohar 
    Date: March, 2024
"""

import sys
import logging
import os

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics

from config_load import OUTPUT_DATA_PATH, MODEL_PATH, TEST_DATA_PATH
import diagnostics


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def plot_confusion_matrix():
    """
    Calcul
    """
    data_frame = pd.read_csv(os.path.join(TEST_DATA_PATH, "testdata.csv"))
    y_gt = data_frame['exited']
    logging.info("Run model predictions on test data.")
    y_pred = diagnostics.model_predictions()
    conf_mat = metrics.confusion_matrix(y_gt, y_pred)
    plot = metrics.ConfusionMatrixDisplay(conf_mat).plot()
    logging.info("Save the plot in %s" % MODEL_PATH)
    plt.savefig(os.path.join(MODEL_PATH, 'confusionmatrix.png'))


if __name__ == "__main__":
    logging.info("Run confusion matrix plot!")
    plot_confusion_matrix()