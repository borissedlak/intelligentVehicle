import os
import warnings
from datetime import datetime, timedelta

import pandas as pd
import pymongo
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader
from prometheus_api_client import PrometheusConnect

import utils
from utils import DB_NAME, COLLECTION_NAME

warnings.filterwarnings("ignore", category=Warning, module='pgmpy')
sample_file = "samples.csv"
cpd_max_sum = 0.95

DEVICE_NAME = os.environ.get('DEVICE_NAME')
if DEVICE_NAME:
    print(f'Found ENV value for DEVICE_NAME: {DEVICE_NAME}')
else:
    DEVICE_NAME = "Unknown"
    print(f"Didn't find ENV value for DEVICE_NAME, default to: {DEVICE_NAME}")

LEADER_HOST = os.environ.get('LEADER_HOST')
if LEADER_HOST:
    print(f'Found ENV value for LEADER_HOST: {LEADER_HOST}')
else:
    LEADER_HOST = "localhost"
    print(f"Didn't find ENV value for LEADER_HOST, default to: {LEADER_HOST}")


# @utils.print_execution_time
def retrieve_full_data():
    mongo_client = pymongo.MongoClient(LEADER_HOST)[DB_NAME]
    df = pd.DataFrame(list(mongo_client[COLLECTION_NAME].find()))

    utils.export_samples(df, sample_file)
    print(f"Reading {df.shape[0]} samples from mongoDB")

    unique_pairs = utils.get_service_host_pairs(df)
    print(f"Contains pairs for {unique_pairs}")


def prepare_models():
    df = pd.read_csv(sample_file)
    unique_pairs = utils.get_service_host_pairs(df)

    df = utils.prepare_samples(df)

    for (service, device_type) in unique_pairs:
        filtered = df[(df['service'] == service) & (df['device_type'] == device_type)]
        print(f"{(service, device_type)} with {filtered.shape[0]} samples")

        model = utils.train_to_BN(filtered, service_name=service, export_file=f"{service}_{device_type}_model.xml")

        true = utils.get_true(utils.infer_slo_fulfillment(VariableElimination(model), ['in_time']))
        print(f"In_time fulfilled for {int(true * 100)} %")

    return len(unique_pairs)


@utils.print_execution_time # takes roughly 45ms for 1 sample
def update_models_new_samples(model_name, samples):
    model = XMLBIFReader(model_name).get_model()
    # updated_model = utils.train_to_BN(samples, service_name="CV", export_file=model_name)

    # past_data_length = len(self.past_training_data)
    # if hasattr(self, 'backup_data'):
    #     past_data_length += len(self.backup_data)

    samples = utils.prepare_samples(samples, conversion=False)
    model.fit_update(samples, n_prev_samples=1000)  # TODO: What is the correct length here? Depends on the batch size I'd say
    utils.export_model_to_path(model, model_name)


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

    if metric_data:
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
    prepare_models()
    # get_latest_load(device_name='Orin')

    # model_path = f"CV_{DEVICE_NAME}_model.xml"
    # model = XMLBIFReader(model_path).get_model()
    #
    # services = {"name": 'CV', 'slo_var': ["in_time"], 'constraints': {'pixel': '480', 'fps': '5'}}
    # slo = utils.get_true(utils.infer_slo_fulfillment(model, services['slo_var'], services['constraints']))
    # print(slo)
