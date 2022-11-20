""" constants.py: Contains constants used throughout the module """
import os

# data
DATA_SET = "data_set"

# paths
PATH_ROOT = os.path.join(os.path.expanduser("~"), "Documents", "vine_tomato_detection")
PATH_DATA_SET = os.path.join(PATH_ROOT, DATA_SET)
PATH_DATA = os.path.join(PATH_DATA_SET, "data", )
PATH_RESULTS = os.path.join(PATH_DATA_SET, "results")
PATH_JSON = os.path.join(PATH_DATA_SET, 'json')
