import os

import pandas as pd
import pymongo

from detector.utils import DB_NAME, COLLECTION_NAME

sample_file = "samples.csv"
cpd_max_sum = 0.95

DEVICE_NAME = os.environ.get('DEVICE_NAME')
if DEVICE_NAME:
    print(f'Found ENV value for DEVICE_NAME: {DEVICE_NAME}')
else:
    DEVICE_NAME = "Unknown"
    print(f"Didn't find ENV value for DEVICE_NAME, default to: {DEVICE_NAME}")

MONGO_HOST = os.environ.get('MONGO_HOST')
if MONGO_HOST:
    print(f'Found ENV value for MONGO_HOST: {MONGO_HOST}')
else:
    MONGO_HOST = "localhost"
    print(f"Didn't find ENV value for MONGO_HOST, default to: {MONGO_HOST}")


# @utils.print_execution_time
def get_full_data(latency_slo=None):
    mongo_client = pymongo.MongoClient(MONGO_HOST)[DB_NAME]

    # TODO: Must filter according to IDs
    metrics = pd.DataFrame(list(mongo_client[COLLECTION_NAME].find()))
    print(metrics.size)

def get_latest_load(latency_slo=None):
    mongo_client = pymongo.MongoClient(MONGO_HOST)[DB_NAME]

    # TODO: Must filter according to IDs
    laptop = mongo_client['Laptop'].find_one()
    print(laptop)


if __name__ == "__main__":
    # 1) Provider
    # Skipped! Assumed at Nano
    # Utilizes 30% CPU, 15% Memory, No GPU, Consumption depending on fps

    # 2) Processor
    get_full_data()
    # get_latest_load()
