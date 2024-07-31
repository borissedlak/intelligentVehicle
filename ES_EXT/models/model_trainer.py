import csv
import itertools
import logging
from datetime import datetime, timedelta

import pandas as pd
import pymongo
from pgmpy.base import DAG
from prometheus_api_client import PrometheusConnect

import utils
from ES_EXT import util_fgcs
from utils import DB_NAME, COLLECTION_NAME

logger = logging.getLogger("vehicle")
sample_file = "samples.csv"

DEVICE_NAME = utils.get_ENV_PARAM("DEVICE_NAME", "Unknown")


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
    dag_cv.add_nodes_from(["pixel", "fps", "in_time", "energy_saved", "rate_60"])
    dag_cv.add_edges_from([("pixel", "in_time"), ("fps", "in_time"), ("pixel", "energy_saved"), ("pixel", "energy_saved"),
                           ("fps", "energy_saved"), ("pixel", "rate_60")])
    dag_qr = DAG()
    dag_qr.add_nodes_from(["pixel", "fps", "in_time", "energy_saved"])
    dag_qr.add_edges_from([("pixel", "in_time"), ("fps", "in_time"), ("pixel", "energy_saved"), ("fps", "energy_saved")])
    dag_li = DAG()
    dag_li.add_nodes_from(["mode", "fps", "in_time", "energy_saved"])
    dag_li.add_edges_from([("mode", "energy_saved"), ("fps", "in_time"), ("fps", "energy_saved")])
    dag_services = {'CV': dag_cv, 'QR': dag_qr, 'LI': dag_li}

    if fill_cpt_all_values:
        line_param = []
        bin_values = [x * 0.95 for x in utils.split_into_bins(utils.NUMBER_OF_BINS)][1:utils.NUMBER_OF_BINS + 1]
        for (source_pixel, source_fps, service, device, delta, energy, mode, rate) in (
                itertools.product([480, 720, 1080], [5, 10, 15, 20, 25], ['CV', 'QR', 'LI'], [DEVICE_NAME],
                                  [1, 1, 1, 1, 1, 999], [1, 1, 1, 1, 1, 999], ['single', 'double'], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0])):
            line_param.append({'pixel': source_pixel, 'fps': source_fps, 'delta': delta,
                               'consumption': energy, 'service': service, 'device_type': device, 'mode': mode, 'rate': rate})
        df_param_fill = util_fgcs.prepare_samples(pd.DataFrame(line_param))
        df = pd.concat([df_param_fill], ignore_index=True)

    unique_pairs = utils.get_service_host_pairs(df_param_fill)
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

        model_name = f"ES_EXT/models/{service}_{device_type}_model.xml"
        utils.train_to_BN(filtered, service_name=f"{service}_{device_type}", export_file=model_name, dag=dag_services[service])
        filtered.to_csv(f"ES_EXT/models/backup/backup_{service}_{device_type}.csv", index=False)

        # true = utils.get_true(utils.infer_slo_fulfillment(VariableElimination(model), ['in_time']))
        # print(f"In_time fulfilled for {int(true * 100)} %")

        # with open(f"ES_EXT/results/slo_f/slo_f_{service}_{device_type}.csv", 'w', newline='') as csv_file:
        #     csv_writer = csv.writer(csv_file)
        #     csv_writer.writerow(["service", "device_type", "timestamp", "pixel", "fps", "pv", "surprise"])
    return len(unique_pairs)


# @utils.print_execution_time  # takes roughly 45ms for 1 sample
# def update_models_new_samples(model_name, samples, call_direct=False):
#     path = ("models/" if not call_direct else "") + model_name
#     model = XMLBIFReader(path).get_model()
#
#     samples = utils.prepare_samples(samples)
#     del samples['device_type']
#     del samples['service']
#
#     model.fit_update(samples, n_prev_samples=PREV_SAMPLES_LENGTH[model_name])
#     PREV_SAMPLES_LENGTH[model_name] += len(samples)
#     utils.export_model_to_path(model, "models/" + model_name)


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
            metrics_lib.update({m: latest_value[1]})
        else:
            logger.error(f"Prometheus server found no instance='{instance}', defaulting to 100 % occupied")
            metrics_lib.update({m: 100.0})

    return metrics_lib


if __name__ == "__main__":
    # retrieve_full_data("localhost")
    prepare_models()
