import copy
import csv
import fnmatch
import logging
import os
import time
from itertools import combinations

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import pgmpy
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from pgmpy.base import DAG
from pgmpy.estimators import AICScore, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFWriter

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def instantiate_advanced_logger(package):
    class CustomFormatter(logging.Formatter):
        grey = "\x1b[38;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

        FORMATS = {
            logging.DEBUG: grey + format + reset,
            logging.INFO: reset + format + reset,
            logging.WARNING: yellow + format + reset,
            logging.ERROR: red + format + reset,
            logging.CRITICAL: bold_red + format + reset
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    logger = logging.getLogger(package)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(CustomFormatter())
    logger.addHandler(stream_handler)

    return logger


log = logging.getLogger("vehicle")

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def print_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000.0
        print(f"{func.__name__} took {execution_time_ms:.0f} ms to execute")
        return result

    return wrapper


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def merge_image_with_overlay(image, boxes, scores, class_ids, mask_alpha=0.4):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]

        draw_box(det_img, box, color)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
             thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image: np.ndarray, text: str, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
              font_size: float = 0.001, text_thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1),
                  (x1 + tw, y1 - th), color, -1)

    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness,
                       cv2.LINE_AA)


def draw_masks(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


def merge_lists_of_dicts(list1, list2):
    merged_list = []
    for dict1, dict2 in zip(list1, list2):
        merged_dict = {**dict1, **dict2}
        merged_list.append(merged_dict)
    return merged_list


def merge_single_dicts(dict1, dict2):
    return {**dict1, **dict2}


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


def get_mb_name(service, host):
    # sorted_strings = sorted([s1, s2], reverse=False)
    # return '-'.join(sorted_strings)
    return service + '-' + host


def infer_slo_fulfillment(bn_model_VE: VariableElimination, slo_variables, constraints=None):
    if constraints is None:
        constraints = {}
    evidence = constraints  # | {'device_type': device_type}
    result = bn_model_VE.query(variables=slo_variables, evidence=evidence)

    return result


def verify_all_parameters_known(model: BayesianNetwork, data, params):
    for variable in params:
        for _, row in data.iterrows():
            if row[variable] not in model.__getattribute__("states")[variable]:
                return False

        # for v in model.get_markov_blanket(variable):
        #     for _, row in data.iterrows():
        #         if row[v] not in model.__getattribute__("states")[v]:
        #             return False

    return True


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
        if param.values.shape == (2,):
            return param.values[1]
        elif param.__getattribute__("state_names")[param.variables[0]][0] == 'True':
            return param.values[0]
        elif param.__getattribute__("state_names")[param.variables[0]][0] == 'False':
            return 0
        else:
            raise RuntimeError("dont know anymore")


def get_sum_up_to_x(cpd: DiscreteFactor, hw_variable, max_prob):
    current_sum = 0
    for index, element in enumerate(cpd.values):
        current_sum += element
        if current_sum >= max_prob:
            return int(cpd.no_to_name[hw_variable][index]) + (0 if hw_variable == "gpu" else 1)

    raise RuntimeError("Why?")


def get_latency_for_devices(d1, d2):
    translate_dict = {'Nano': 0, 'Xavier': 1, 'Orin': 2, 'Laptop': 3, 'PC': 4}
    # TODO: See that this actually makes sense together with the evaluation
    distance = np.array([[1, 3, 5, 10, 20],
                         [3, 1, 2, 7, 17],
                         [5, 2, 1, 5, 15],
                         [10, 7, 5, 1, 10],
                         [20, 17, 15, 10, 1]])

    a = translate_dict[d1]
    b = translate_dict[d2]

    return distance[a, b]


def normalize_to_pods(prob_distribution, num_pods):
    """
    Normalize a probability distribution to a specified number of pods (bins).

    Parameters:
    - prob_distribution: NumPy array representing the probability distribution.
    - num_pods: Number of pods (bins) to normalize the distribution to.

    Returns:
    - Normalized probability distribution.
    """
    hist, bin_edges = np.histogram(prob_distribution, bins=num_pods, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    return hist / np.sum(hist), bin_centers


def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    return intersection_size / union_size


def prepare_samples(samples: pd.DataFrame, remove_device_metrics=False, export_path=None, conversion=True):
    if conversion:
        samples["delta"] = samples["delta"].apply(np.floor).astype(int)
        samples["cpu"] = samples["cpu"].apply(np.floor).astype(int)
        samples["memory"] = samples["memory"].apply(np.floor).astype(int)
        samples['in_time'] = samples['delta'] <= (1000 / samples['fps'])
        samples['energy_saved'] = samples['consumption'] <= 100

        # Idea: The number of buckets or even their distribution could also be a hyperparameter
        samples['cpu'] = pd.cut(samples['cpu'], bins=[0, 25, 50, 75, 100],
                                labels=['Low', 'Mid', 'High', 'Very_High'], include_lowest=True)
        samples['memory'] = pd.cut(samples['memory'], bins=[0, 25, 50, 75, 100],
                                   labels=['Low', 'Mid', 'High', 'Very_High'], include_lowest=True)
        samples['gpu'] = pd.cut(samples['gpu'], bins=[0, 25, 50, 75, 100],
                                labels=['Low', 'Mid', 'High', 'Very_High'], include_lowest=True)

    samples['fps'] = samples['fps'].astype(str)
    samples['pixel'] = samples['pixel'].astype(str)
    samples['in_time'] = samples['in_time'].astype(str)
    samples['energy_saved'] = samples['energy_saved'].astype(str)

    if hasattr(samples, '_id'):
        del samples['_id']
    if hasattr(samples, 'timestamp'):
        del samples['timestamp']
    if hasattr(samples, 'delta'):
        del samples['delta']
    if hasattr(samples, 'consumption'):
        del samples['consumption']

    if remove_device_metrics:
        del samples['cpu']
        del samples['consumption']
        del samples['device_type']

    if export_path is not None:
        samples.to_csv(export_path, index=False)
        print(f"Exported sample file to: {export_path}")

    return samples


def export_samples(samples: pd.DataFrame, export_path):
    samples.to_csv(export_path, index=False)
    print(f"Loaded {export_path} from MongoDB")

    return samples


def train_to_BN(samples, service_name, export_file=None, samples_path=None, dag=None):
    if samples_path is not None:
        samples = pd.read_csv(samples_path)

    if dag is None:
        scoring_method = AICScore(data=samples)  # BDeuScore | AICScore
        estimator = HillClimbSearch(data=samples)

        dag: pgmpy.base.DAG = estimator.estimate(
            scoring_method=scoring_method, max_indegree=5, epsilon=1,
        )

    export_BN_to_graph(dag, vis_ls=['circo'], save=False, name="raw_model", show=True)
    model = BayesianNetwork(ebunch=dag)
    model.name = service_name
    model.fit(data=samples, estimator=MaximumLikelihoodEstimator)

    if export_file is not None:
        export_model_to_path(model, export_file)

    return model


# @print_execution_time
def get_surprise_for_data(model: BayesianNetwork, model_VE: VariableElimination, data: pd.DataFrame, slo_variables):
    bic_sum = 0.0
    try:
        for variable in slo_variables:
            cpd = get_mbs_as_bn(model, [variable]).get_cpds(variable)
            log_likelihood = 0.0
            evidence_variables = ['fps', 'pixel']  # model.get_markov_blanket(variable)

            # if 'consumption' in evidence_variables:
            #     evidence_variables.remove('consumption')

            for _, row in data.iterrows():
                evidence = {col: row[col] for col in evidence_variables}
                query_result = model_VE.query(variables=[variable], evidence=evidence)
                state_index = cpd.__getattribute__("state_names")[variable].index(row[variable])
                p = query_result.values[state_index]
                log_likelihood += np.log(p if p > 0 else 1e-10)

            k = len(cpd.get_values().flatten()) - len(cpd.variables)

            n = len(data)
            bic = -2 * log_likelihood + k * np.log(n)
            bic_sum += bic
    except ValueError or KeyError as e:
        print_in_red(f"Should not happen after safeguard function!!!!" + str(e))

    return bic_sum


def export_model_to_path(model, export_file):
    writer = XMLBIFWriter(model)
    writer.write_xmlbif(filename=export_file)
    log.info(f"L| Model exported as '{export_file}'")


def find_files_with_prefix(directory, prefix, suffix):
    file_list = [f for f in os.listdir(directory) if
                 f.startswith(prefix) and os.path.isfile(os.path.join(directory, f)) and f.endswith(suffix)]
    return file_list


def find_nested_files_with_suffix(root_dir, suffix):
    matching_files = []

    for root, dirs, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, f"*{suffix}"):
            matching_files.append(os.path.join(root, filename))

    return matching_files


def MERGE(service_mb: BayesianNetwork, potential_host_mb: BayesianNetwork):
    service_mb.add_node('cpu')
    service_mb.add_node('device_type')
    service_mb.add_cpds(potential_host_mb.get_cpds('device_type'))
    service_mb.add_cpds(potential_host_mb.get_cpds('cpu'))

    return service_mb


def check_edges_with_service(potential_host_mb: BayesianNetwork):
    hardware_variables = ['cpu', 'device_type', 'memory', 'consumption']
    all_combinations = list(combinations(hardware_variables, 2))

    for (u, v) in all_combinations:
        if potential_host_mb.has_edge(u, v):
            potential_host_mb.remove_edge(u, v)
        elif potential_host_mb.has_edge(v, u):
            potential_host_mb.remove_edge(v, u)

    for variable in hardware_variables:
        if potential_host_mb.degree[variable] != 0:
            return False

    return True


def check_device_present_in_mb(model, device):
    devices_contained = model.get_cpds('device_type').__getattribute__('state_names')['device_type']
    return device in devices_contained


def log_dict(service, device, variable_dict, Consumer_to_Worker_constraints, most_restrictive_consumer_latency):
    with open("../analysis/inference/n_n_assignments.csv", 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [service] + [device] + list(Consumer_to_Worker_constraints.values()) + [most_restrictive_consumer_latency]
            + list(variable_dict.values()))


def print_in_red(text):
    print("\x1b[31m" + text + "\x1b[0m")


def is_jetson_host(host_name):
    return host_name in ["OrinNano", "Xavier", "OrinNano", "Orin"]


def get_service_host_pairs(df):
    unique_pairs_df = df[['service', 'device_type']].drop_duplicates()
    unique_pairs = list(unique_pairs_df.itertuples(index=False, name=None))
    return unique_pairs


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError(f"Cannot parse '{s}' as a boolean")


COLLECTION_NAME = "metrics"
DB_NAME = "vehicle"


def create_model_name(service_name, device_name):
    return f"{service_name}_{device_name}_model.xml"


def get_ENV_PARAM(var, DEFAULT):
    ENV = os.environ.get(var)
    if ENV:
        logging.info(f'Found ENV value for {var}: {ENV}')
    else:
        ENV = DEFAULT
        logging.warning(f"Didn't find ENV value for {var}, default to: {DEFAULT}")
    return ENV


def log_and_return(lg, severity, msg):
    lg.log(severity, msg)
    return msg
