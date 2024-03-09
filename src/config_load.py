"""
    This script reads the folder paths from the config.json file.

    Author: Sabari Manohar
    Date  : March, 2024
"""
import json
import os


# open config.json
# extract appropriate path variable
config_path = os.path.join(os.path.abspath('./'), 'config.json')
with open(config_path, 'r', encoding='utf-8') as file:
    CONFIG = json.load(file)

# practice data folder
INPUT_FOLDER_PATH = os.path.join(
    os.path.abspath('./'),
    'dataset',
    CONFIG['input_folder_path'])

# ingestion data folder
OUTPUT_DATA_PATH = os.path.join(
    os.path.abspath('./'),
    'dataset',
    CONFIG['output_folder_path'])

# test data folder 
TEST_DATA_PATH = os.path.join(
    os.path.abspath('./'),
    'dataset',
    CONFIG['test_data_path'])

# model store path 
MODEL_PATH = os.path.join(
    os.path.abspath('./'),
    'models',
    CONFIG['output_model_path'])

# production model store path
PROD_DEPLOYMENT_PATH = os.path.join(os.path.abspath(
    './'), 'models', CONFIG['prod_deployment_path'])

