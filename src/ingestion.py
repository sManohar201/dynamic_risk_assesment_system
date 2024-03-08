"""
    The file contains code for reading the dataset and compiling the dataset. 

    Author: Sabari Manohar
    Date:   March, 2024
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
print(config)



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    pass



# if __name__ == '__main__':
    # merge_multiple_dataframe()
    # pass
