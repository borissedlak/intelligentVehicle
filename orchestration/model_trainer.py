import itertools
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import pymongo
import utils
from pandas.errors import EmptyDataError
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader
from prometheus_api_client import PrometheusConnect
from utils import DB_NAME, COLLECTION_NAME

logger = logging.getLogger("vehicle")
sample_file = "samples.csv"

DEVICE_NAME = utils.get_ENV_PARAM("DEVICE_NAME", "Unknown")
LEADER_HOST = utils.get_ENV_PARAM("LEADER_HOST", "localhost")

PREV_SAMPLES_LENGTH = 300  # Idea: This is also a hyperparameter, initially I should be small and then larger later


# @utils.print_execution_time
def retrieve_full_data():
    mongo_client = pymongo.MongoClient(LEADER_HOST)[DB_NAME]
    df = pd.DataFrame(list(mongo_client[COLLECTION_NAME].find()))

    utils.export_samples(df, sample_file)
    print(f"Reading {df.shape[0]} samples from mongoDB")

    # unique_pairs = utils.get_service_host_pairs(df)
    # print(f"Contains pairs for {unique_pairs}")


# TODO: I must finally pin the DAG I think
def prepare_models(fill_param_tables=True):
    try:
        df = pd.read_csv(sample_file)
        df = utils.prepare_samples(df)
    except FileNotFoundError as e:  # Cannot place both in a line, that's weird ...
        logger.error(e)
        df = pd.DataFrame()
    except EmptyDataError as e:
        logger.error(e)
        df = pd.DataFrame()

    if fill_param_tables:
        line_param = []
        bin_values = [x * 0.95 for x in utils.split_into_bins(utils.NUMBER_OF_BINS)][1:utils.NUMBER_OF_BINS + 1]
        for (source_pixel, source_fps, service, device, cpu, gpu, memory, delta, energy, isolated) in (
                itertools.product([480, 720, 1080], [5, 10, 15, 20], ['CV'], ['Laptop', 'Orin'], bin_values, bin_values, bin_values,
                                  [1, 999], [1, 999], [True, False])):
            line_param.append({'pixel': source_pixel, 'fps': source_fps, 'cpu': cpu, 'memory': memory, 'gpu': gpu, 'delta': delta,
                               'consumption': energy, 'service': service, 'device_type': device, 'isolated': isolated})
        df_param_fill = utils.prepare_samples(pd.DataFrame(line_param))
        df = pd.concat([df, df_param_fill], ignore_index=True)

    unique_pairs = utils.get_service_host_pairs(df)
    for (service, device_type) in unique_pairs:
        filtered = df[(df['service'] == service) & (df['device_type'] == device_type)]
        print(f"{(service, device_type)} with {filtered.shape[0]} samples")

        model = utils.train_to_BN(filtered, service_name=service, export_file=f"{service}_{device_type}_model.xml")

        true = utils.get_true(utils.infer_slo_fulfillment(VariableElimination(model), ['in_time']))
        print(f"In_time fulfilled for {int(true * 100)} %")

    return len(unique_pairs)


@utils.print_execution_time  # takes roughly 45ms for 1 sample
def update_models_new_samples(model_name, samples):
    model = XMLBIFReader(model_name).get_model()

    samples = utils.prepare_samples(samples)
    model.fit_update(samples, n_prev_samples=PREV_SAMPLES_LENGTH)
    utils.export_model_to_path(model, model_name)


# TODO: This takes way too long, doing the HTTP request in the processing cycle is nonsense
@utils.print_execution_time
def get_latest_load(instance, metric_types=["cpu", "gpu", "memory"], device_name="Laptop"):
    # Connect to Prometheus
    prom = PrometheusConnect(url=f"http://{LEADER_HOST}:9090", disable_ssl=True)

    # Query the latest value
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=5)  # Query the last 5 minutes for safety

    metrics_lib = {}
    for m in metric_types:
        query = m + '_load{instance="' + instance + ':8000"}' # device_name="' + device_name + '",

        metric_data = prom.get_metric_range_data(
            metric_name=query,
            start_time=start_time,
            end_time=end_time
        )

        if metric_data:
            latest_value = metric_data[0]['values'][-1]
            metrics_lib = metrics_lib | {m: latest_value[1]}
        else:
            print("No data found for the given query.")
            metrics_lib = metrics_lib | {m: None}

    return metrics_lib


def convert_prometheus_to_category(current_load):
    current_load_list = list(map(float, list(current_load.values())))
    return np.digitize(list(current_load_list), utils.split_into_bins(utils.NUMBER_OF_BINS)) - 1


if __name__ == "__main__":
    # retrieve_full_data()
    # prepare_models()
    print(get_latest_load(instance="host.docker.internal"))
    print(get_latest_load(instance="192.168.31.183"))
