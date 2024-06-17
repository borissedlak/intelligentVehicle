import os
from datetime import datetime, timedelta

import pandas as pd
import pymongo
from prometheus_api_client import PrometheusConnect

import utils
from utils import DB_NAME, COLLECTION_NAME, export_samples

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
def retrieve_full_data():
    mongo_client = pymongo.MongoClient(MONGO_HOST)[DB_NAME]
    df = pd.DataFrame(list(mongo_client[COLLECTION_NAME].find()))

    export_samples(df, sample_file)
    print(f"Reading {df.shape[0]} samples from mongoDB")

    unique_pairs = utils.get_service_host_pairs(df)
    print(f"Contains pairs for {unique_pairs}")


def transform_metrics_for_MKP():
    df = pd.read_csv(sample_file)
    unique_pairs = utils.get_service_host_pairs(df)

    for (service, device_type) in unique_pairs:
        filtered = df[(df['service'] == service) & (df['device_type'] == device_type)]
        print(f"{(service, device_type)} with {filtered.shape[0]} samples")
        utils.train_to_BN(filtered, service_name=service)

        # conditions = {'pixel': 480, 'fps': 25}
        #
        # mask = pd.Series([True] * len(filtered))
        # for column, value in conditions.items():
        #     mask = mask & (df[column] == value)
        #
        # filtered = filtered[mask]

        condition = filtered['delta'] < 1000 / 20 #filtered['fps']
        percentage = (condition.sum() / len(filtered)) * 100
        print(f"In_time fulfilled for {int(percentage)} %")

        # infer_slo_fulfillment(filtered, conditions)


def infer_slo_fulfillment(df, conditions):
    pass


def get_latest_load(device_name="Laptop"):
    # Connect to Prometheus
    prom = PrometheusConnect(url="http://localhost:9090", disable_ssl=True)

    # Define the query
    query = 'cpu_load{device_name="' + device_name + '"}'

    # Query the latest value
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=10)  # Query the last 5 minutes for safety

    # Get the metric data
    metric_data = prom.get_metric_range_data(
        metric_name=query,
        start_time=start_time,
        end_time=end_time
    )

    # Check if data is present
    if metric_data:
        # Get the latest value
        latest_value = metric_data[0]['values'][-1]
        timestamp, cpu_load = latest_value
        print(f"Timestamp: {timestamp}, CPU Load: {cpu_load}")
    else:
        print("No data found for the given query.")


if __name__ == "__main__":
    # 1) Provider
    # Skipped! Assumed at Nano
    # Utilizes 30% CPU, 15% Memory, No GPU, Consumption depending on fps

    # 2) Processor
    retrieve_full_data()
    transform_metrics_for_MKP()
    # get_latest_load()
