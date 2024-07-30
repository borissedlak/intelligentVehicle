import copy
import time
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
import pgmpy
import psutil
import requests
import torch
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from pgmpy.base import DAG
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from scipy.interpolate import griddata

import utils

header_csv = 'execution_time,timestamp,cpu_utilization,memory_usage,pixel,fps,success,distance,consumption,stream_count\n'


def print_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000.0
        print(f"{func.__name__} took {execution_time_ms:.0f} ms to execute")
        return result

    return wrapper


def diffAsStringInMS(a: datetime, b: datetime):
    return int((a - b).microseconds / 1000)


def getExecutionTime(a: datetime, b: datetime):
    if a is not None and b is not None:
        # print(f' {name} took {diffAsStringInMS(a, b)}ms')
        return diffAsStringInMS(a, b)
    return 0


# @print_execution_time # takes ~1ms with both files
def write_execution_times(write_store, number_threads=1):
    f = open(f'../data/Last_Batch.csv', 'w+')
    ph = open(f'../data/Performance_History.csv', 'a')
    f.write(header_csv)

    if number_threads > 1:
        result = []
        i = 0
        while i < len(write_store):
            result.append(write_store[i])
            i += number_threads
        write_store = result

    # TODO: This assumes that still all elements are in the write_store, but the duplicates were filtered out
    sum_other_elements = 0
    for tup in write_store[number_threads:]:
        sum_other_elements += tup[7]
    avg_distance = sum_other_elements / (len(write_store) - number_threads)
    for i in range(number_threads):
        write_store[i] = write_store[i][:7] + (np.floor(avg_distance),) + write_store[i][8:]

    for (delta, ts, cpu, memory, pixel, fps, detected, distance, consumption, thread_number) in write_store:
        f.write(f'{delta},{ts},{cpu},{memory},{pixel},{fps},{detected},{distance},{consumption},{thread_number}\n')
        ph.write(f'{delta},{ts},{cpu},{memory},{pixel},{fps},{detected},{distance},{consumption},{thread_number}\n')

    f.close()
    ph.close()


def gpu_available():
    return torch.cuda.is_available()


def clear_performance_history(path):
    try:
        # Open the file in "w" mode, which creates an empty file or clears existing content
        with open(path, 'w+') as ph:
            ph.write(header_csv)
        print(f"File '{path}' has been cleared.")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_cpu_in_bin():
    cpu = psutil.cpu_percent()
    return pd.cut(pd.Series([cpu]), bins=[0, 50, 70, 90, 100], labels=['Low', 'Mid', 'High', 'Very_High'],
                  include_lowest=True)[0]


def get_center_from_box(box):
    x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return center_x, center_y


# TODO: Maybe revert and simply take the absolute pixel distance and find some use case where this matches
def get_relative_distance_between_points(p1, p2, img):
    p1_x, p1_y = p1
    p2_x, p2_y = p2

    # The intention was to get the relative distance in different resolutions. The results match quite, but are a bit off
    return np.ceil(np.linalg.norm(np.array([p1_x / img.shape[1], p1_y / img.shape[0]]) - np.array(
        [p2_x / img.shape[1], p2_y / img.shape[0]])) * 1000)
    # return math.ceil(math.sqrt(((p1_x - p2_x)/img.shape[1])**2 + ((p1_y - p2_y)/img.shape[0])**2) * 1000)


def upload_file():
    # The API endpoint to communicate with
    url_post = "http://192.168.1.153:5000/upload"

    # A POST request to tthe API
    files = {'file': open('../data/Performance_History.csv', 'rb')}
    post_response = requests.post(url_post, files=files)

    # Print the response
    post_response_json = post_response.content
    print(post_response_json)


def print_in_red(text):
    print("\x1b[31m" + text + "\x1b[0m")


# @print_execution_time
def get_surprise_for_data(model: BayesianNetwork, data, slo_vars):
    inference = VariableElimination(get_mbs_as_bn(model, slo_vars))

    bic_sum = 0.0
    # try:
    for variable in slo_vars:
        cpd = get_mbs_as_bn(model, [variable]).get_cpds(variable)
        log_likelihood = 0.0
        evidence_variables = model.get_markov_blanket(variable)

        # if 'consumption' in evidence_variables:
        #     evidence_variables.remove('consumption')

        for _, row in data.iterrows():
            evidence = {col: row[col] for col in evidence_variables}
            query_result = inference.query(variables=[variable], evidence=evidence)
            state_index = cpd.__getattribute__("state_names")[variable].index(row[variable])
            p = query_result.values[state_index]
            log_likelihood += np.log(p if p > 0 else 1e-10)

        k = len(cpd.get_values().flatten()) - len(cpd.variables)

        n = len(data)
        bic = -2 * log_likelihood + k * np.log(n)
        bic_sum += bic
    # except ValueError or KeyError as e:
    #     print_in_red(f"Should not happen after safeguard function!!!!" + str(e))

    return bic_sum


# @print_execution_time # takes ~2ms
def get_mbs_as_bn(model: DAG or BayesianNetwork, center: [str]):
    mb_list = []
    for node in center:
        mb_list.extend(model.get_markov_blanket(node))
    mb = copy.deepcopy(model)

    mb_list.extend(center)
    for n in model.nodes:
        if n not in mb_list:
            mb.remove_node(n)

    return mb


def interpolate_values(matrix):
    x = np.arange(matrix.shape[1])
    y = np.arange(matrix.shape[0])
    xx, yy = np.meshgrid(x, y)

    # Flatten the data and coordinates
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    data_flat = matrix.flatten()

    valid_indices = data_flat != -1  # -1 indicates empty values
    x_valid = x_flat[valid_indices]
    y_valid = y_flat[valid_indices]
    data_valid = data_flat[valid_indices]

    # Interpolate missing values using griddata
    m = 'linear' if len(data_valid) >= 4 else 'nearest'
    interpolated_data = griddata((x_valid, y_valid), data_valid, (xx, yy), method=m)

    mask = np.isnan(interpolated_data)
    interpolated_data[mask] = griddata((x_valid, y_valid), data_valid, (xx[mask], yy[mask]), method='nearest')
    return interpolated_data


def prepare_samples(samples: pd.DataFrame, export_path=None, conversion=True):
    if conversion:
        # samples["delta"] = samples["delta"].apply(np.floor).astype(int)
        # samples["cpu"] = samples["cpu"].apply(np.floor).astype(int)
        # samples["memory"] = samples["memory"].apply(np.floor).astype(int)
        samples['in_time'] = samples['delta'] <= (1000 / samples['fps'])
        samples['energy_saved'] = samples['consumption'] <= 25

        # samples['cpu'] = pd.cut(samples['cpu'], bins=utils.split_into_bins(utils.NUMBER_OF_BINS),
        #                         labels=list(range(utils.NUMBER_OF_BINS)), include_lowest=True)
        # samples['memory'] = pd.cut(samples['memory'], bins=utils.split_into_bins(utils.NUMBER_OF_BINS),
        #                            labels=list(range(utils.NUMBER_OF_BINS)), include_lowest=True)
        # samples['gpu'] = pd.cut(samples['gpu'], bins=utils.split_into_bins(utils.NUMBER_OF_BINS),
        #                         labels=list(range(utils.NUMBER_OF_BINS)), include_lowest=True)
        if hasattr(samples, 'rate'):
            samples["rate"] = samples["rate"].astype(float)
            samples['rate_60'] = samples['rate'] >= 0.60

    # samples['cpu'] = samples['cpu'].astype(str)
    # samples['memory'] = samples['memory'].astype(str)
    # samples['gpu'] = samples['gpu'].astype(str)
    samples['fps'] = samples['fps'].astype(str)
    samples['in_time'] = samples['in_time'].astype(str)
    samples['energy_saved'] = samples['energy_saved'].astype(str)

    if hasattr(samples, 'rate_60'):
        samples['rate_60'] = samples['rate_60'].astype(str)
    if hasattr(samples, 'pixel'):
        samples['pixel'] = samples['pixel'].astype(str)

    if hasattr(samples, '_id'):
        del samples['_id']
    if hasattr(samples, 'timestamp'):
        del samples['timestamp']
    if hasattr(samples, 'delta'):
        del samples['delta']
    if hasattr(samples, 'consumption'):
        del samples['consumption']
    if hasattr(samples, 'rate'):
        del samples['rate']

    if hasattr(samples, 'memory'):
        del samples['memory']
    if hasattr(samples, 'cpu'):
        del samples['cpu']
    if hasattr(samples, 'gpu'):
        del samples['gpu']

    if export_path is not None:
        samples.to_csv(export_path, index=False)
        print(f"Exported sample file to: {export_path}")

    return samples


def export_BN_to_graph(bn: BayesianNetwork or pgmpy.base.DAG, root=None, try_visualization=False, vis_ls=None,
                       save=False,
                       name=None, show=True, color_map=None):
    if vis_ls is None:
        vis_ls = ["fdp"]
    else:
        vis_ls = vis_ls

    if name is None:
        name = root

    if try_visualization:
        vis_ls = ['neato', 'dot', 'twopi', 'fdp', 'sfdp', 'circo']

    for s in vis_ls:
        pos = graphviz_layout(bn, root=root, prog=s)
        nx.draw(
            bn, pos, with_labels=True, arrowsize=20, node_size=1500,  # alpha=1.0, font_weight="bold",
            node_color=color_map
        )
        if save:
            plt.box(False)
            plt.savefig(f"{name}.png", dpi=400, bbox_inches="tight")  # default dpi is 100
        if show:
            plt.show()


def cap_0_1(num: float):
    if num < 0.0:
        return 0.0
    elif num > 1.0:
        return 1.0
    return num


def get_true(param):
    if len(param.variables) > 2:
        raise Exception("How come?")
    if len(param.variables) == 2:
        if param.values.shape == (1, 1):
            if (param.__getattribute__("state_names")[param.variables[0]][0] == 'True' and
                    param.__getattribute__("state_names")[param.variables[1]][0] == 'True'):
                return 1
            else:
                return 0
        elif param.values.shape == (2, 1):
            if (param.__getattribute__("state_names")[param.variables[0]][0] == 'True' or
                    param.__getattribute__("state_names")[param.variables[1]][0] == 'True'):
                return param.values[1][0]
            else:
                return 0
        elif param.values.shape == (1, 2):
            if (param.__getattribute__("state_names")[param.variables[0]][0] == 'True' or
                    param.__getattribute__("state_names")[param.variables[1]][0] == 'True'):
                return param.values[0][1]
            else:
                return 0
        elif param.values.shape == (2, 2):
            return param.values[1][1]
        else:
            return param.values[1]
    elif len(param.variables) == 1:
        if param.values.shape == (2, 1):
            return param.values[1]
        elif param.__getattribute__("state_names")[param.variables[0]][0] == True:
            return 1
        else:
            return 0
        # else param.values[0]
