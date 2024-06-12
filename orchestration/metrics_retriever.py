import os

import pandas as pd
import pymongo

from detector import utils
from detector.utils import DB_NAME, COLLECTION_NAME, export_samples

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


@utils.print_execution_time
def retrieve_full_data():
    mongo_client = pymongo.MongoClient(MONGO_HOST)[DB_NAME]

    # TODO: Must filter according to IDs
    df = pd.DataFrame(list(mongo_client[COLLECTION_NAME].find()))
    # export_samples(metrics, sample_file)

    # distinct_services = df['service'].unique()
    # distinct_device_types = df['device_type'].unique()
    #
    # print("Distinct services:", distinct_services)
    # print("Distinct device types:", distinct_device_types)

    unique_pairs_df = df[['service', 'device_type']].drop_duplicates()
    unique_pairs = list(unique_pairs_df.itertuples(index=False, name=None))

    print(unique_pairs)
    print(df.size)

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
    retrieve_full_data()
    # get_latest_load()
