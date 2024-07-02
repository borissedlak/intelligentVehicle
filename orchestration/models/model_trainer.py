import itertools
import logging
from datetime import datetime, timedelta

import pandas as pd
import pymongo
from pandas.errors import EmptyDataError
from pgmpy.base import DAG
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader
from prometheus_api_client import PrometheusConnect

import utils
from utils import DB_NAME, COLLECTION_NAME

logger = logging.getLogger("vehicle")
sample_file = "samples.csv"

DEVICE_NAME = utils.get_ENV_PARAM("DEVICE_NAME", "Unknown")
# LEADER_HOST = utils.get_ENV_PARAM("LEADER_HOST", "localhost")

# PREV_SAMPLES_LENGTH = 300  # Idea: This is also a hyperparameter, initially I should be small and then larger later
PREV_SAMPLES_LENGTH = {utils.create_model_name(service, device): 1 for service in ['CV', 'QR', 'LI'] for device in ['Laptop', 'Orin']}


# @utils.print_execution_time
def retrieve_full_data(mongo_host):
    mongo_client = pymongo.MongoClient(mongo_host)[DB_NAME]
    df = pd.DataFrame(list(mongo_client[COLLECTION_NAME].find()))

    df['pixel'] = df['pixel'].apply(lambda x: 1080 if pd.isna(x) else x)
    df['pixel'] = df['pixel'].astype(int)

    df['rate'] = df['rate'].apply(lambda x: 0.0 if pd.isna(x) else x)
    df['rate'] = df['rate'].astype(float)

    utils.export_samples(df, sample_file)
    print(f"Reading {df.shape[0]} samples from mongoDB")


def prepare_models(fill_cpt_all_values=True):
    dag_cv = DAG()
    dag_cv.add_nodes_from(["pixel", "fps", "isolated", "cpu", "in_time", "gpu", "memory", "energy_saved", "is_leader", "rate_60"])
    dag_cv.add_edges_from([("pixel", "cpu"), ("pixel", "in_time"), ("fps", "cpu"), ("fps", "in_time"), ("fps", "gpu"), ("isolated", "cpu"),
                           ("isolated", "in_time"), ("isolated", "gpu"), ("isolated", "memory"), ("isolated", "energy_saved"),
                           ("cpu", "energy_saved"), ("gpu", "energy_saved"), ("is_leader", "energy_saved"), ("pixel", "rate_60"),
                           ("pixel", "gpu")])
    dag_qr = DAG()
    dag_qr.add_nodes_from(["pixel", "fps", "isolated", "cpu", "in_time", "gpu", "memory", "energy_saved", "is_leader"])
    dag_qr.add_edges_from([("pixel", "cpu"), ("pixel", "in_time"), ("fps", "cpu"), ("fps", "in_time"), ("fps", "gpu"), ("isolated", "cpu"),
                           ("isolated", "in_time"), ("isolated", "gpu"), ("isolated", "memory"), ("isolated", "energy_saved"),
                           ("cpu", "energy_saved"), ("gpu", "energy_saved"), ("is_leader", "energy_saved")])
    dag_li = DAG()
    dag_li.add_nodes_from(["mode", "fps", "isolated", "cpu", "in_time", "gpu", "memory", "energy_saved", "is_leader"])
    dag_li.add_edges_from([("mode", "cpu"), ("mode", "gpu"), ("fps", "cpu"), ("fps", "in_time"), ("fps", "gpu"),
                           ("isolated", "cpu"), ("isolated", "in_time"), ("isolated", "gpu"), ("isolated", "memory"),
                           ("isolated", "energy_saved"), ("cpu", "energy_saved"), ("gpu", "energy_saved"), ("is_leader", "energy_saved")])
    dag_services = {'CV': dag_cv, 'QR': dag_qr, 'LI': dag_li}

    try:
        df = pd.read_csv(sample_file)
        df = utils.prepare_samples(df)
    except FileNotFoundError as e:  # Cannot place both in a line, that's weird ...
        logger.error(e)
        df = pd.DataFrame()
    except EmptyDataError as e:
        logger.error(e)
        df = pd.DataFrame()

    if fill_cpt_all_values:
        line_param = []
        bin_values = [x * 0.95 for x in utils.split_into_bins(utils.NUMBER_OF_BINS)][1:utils.NUMBER_OF_BINS + 1]
        for (source_pixel, source_fps, service, device, cpu, gpu, memory, delta, energy, isolated, leader, mode, rate) in (
                itertools.product([480, 720, 1080], [5, 10, 15], ['CV', 'QR', 'LI'], ['Laptop', 'Orin'], bin_values, bin_values,
                                  bin_values, [1, 999], [1, 999], [True, False], [True, False], ['single', 'double'], [0.0, 1.0])):
            line_param.append({'pixel': source_pixel, 'fps': source_fps, 'cpu': cpu, 'memory': memory, 'gpu': gpu, 'delta': delta,
                               'consumption': energy, 'service': service, 'device_type': device, 'isolated': isolated, 'is_leader': leader,
                               'mode': mode, 'rate': rate})
        df_param_fill = utils.prepare_samples(pd.DataFrame(line_param))
        df = pd.concat([df, df_param_fill], ignore_index=True)

    unique_pairs = utils.get_service_host_pairs(df)
    for (service, device_type) in unique_pairs:
        filtered = df[(df['service'] == service) & (df['device_type'] == device_type)]
        print(f"{(service, device_type)} with {filtered.shape[0]} samples")

        if service == 'LI':
            del filtered['pixel']
        if service in ['CV', 'QR']:
            del filtered['mode']
        if service in ['LI', 'QR']:
            del filtered['rate_60']

        del filtered['device_type']
        del filtered['service']

        model_name = f"{service}_{device_type}_model.xml"
        model = utils.train_to_BN(filtered, service_name=f"{service}_{device_type}", export_file=model_name, dag=dag_services[service])
        # update_models_new_samples(model_name, filtered, call_direct=True)

        true = utils.get_true(utils.infer_slo_fulfillment(VariableElimination(model), ['in_time']))
        print(f"In_time fulfilled for {int(true * 100)} %")

    return len(unique_pairs)


@utils.print_execution_time  # takes roughly 45ms for 1 sample
def update_models_new_samples(model_name, samples, call_direct=False):
    path = ("models/" if not call_direct else "") + model_name
    model = XMLBIFReader(path).get_model()

    samples = utils.prepare_samples(samples)
    del samples['device_type']
    del samples['service']

    model.fit_update(samples, n_prev_samples=PREV_SAMPLES_LENGTH[model_name])
    PREV_SAMPLES_LENGTH[model_name] += len(samples)
    utils.export_model_to_path(model, "models/" + model_name)


# @utils.print_execution_time
def get_latest_load(prometheus_host, instance):
    # Connect to Prometheus
    prom = PrometheusConnect(url=f"http://{prometheus_host}:9090", disable_ssl=True)

    # Query the latest value
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=5)  # Query the last 5 minutes for safety

    metrics_lib = {}
    for m in ["cpu", "gpu", "memory"]:
        query = m + '_load{instance="' + instance + ':8000"}'  # device_name="' + device_name + '",

        metric_data = prom.get_metric_range_data(
            metric_name=query,
            start_time=start_time,
            end_time=end_time
        )

        if metric_data:
            latest_value = metric_data[0]['values'][-1]
            metrics_lib = metrics_lib | {m: latest_value[1]}
        else:
            logger.error(f"Prometheus server found no instance='{instance}', defaulting to 100 % occupied")
            metrics_lib = metrics_lib | {m: 100.0}

    return metrics_lib


if __name__ == "__main__":
    retrieve_full_data(utils.get_local_ip())
    # prepare_models()
