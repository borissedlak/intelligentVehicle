import ast
import os
import threading

import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFReader

import utils
from monitor.DeviceMetricReporter import CyclicArray
from orchestration.http_client import HttpClient
from services.CV.VideoDetector import VideoDetector
from services.VehicleService import VehicleService

HTTP_SERVER = os.environ.get('HTTP_SERVER')
if HTTP_SERVER:
    print(f'Found ENV value for HTTP_SERVER: {HTTP_SERVER}')
else:
    HTTP_SERVER = "127.0.0.1"
    print(f"Didn't find ENV value for HTTP_SERVER, default to: {HTTP_SERVER}")

DEVICE_NAME = os.environ.get('DEVICE_NAME')
if DEVICE_NAME:
    print(f'Found ENV value for DEVICE_NAME: {DEVICE_NAME}')
else:
    DEVICE_NAME = "Unknown"
    print(f"Didn't find ENV value for DEVICE_NAME, default to: {DEVICE_NAME}")

thread_lib = []
http_client = HttpClient(HOST=HTTP_SERVER)

MODEL_DIRECTORY = "./"


# def processing_loop(inf_service: VehicleService, constraints):
class ServiceWrapper:
    def __init__(self, inf_service: VehicleService, description, model: BayesianNetwork):
        self._running = True
        self.inf_service = inf_service
        self.s_description = description
        self.model = model  # TODO: Filter MB with utils.get_mbs_as_bn(model, self.s_description['slo_var'])
        self.model_VE = VariableElimination(self.model)
        self.slo_hist = CyclicArray(100)  # TODO: If at some point I do dynamic adaptations, I must clear this
        self.under_unknown_config = not utils.verify_all_parameters_known(model,
                                                                          pd.DataFrame([self.s_description['constraints']]),
                                                                          list(self.s_description['constraints'].keys()))

    def terminate(self):
        self._running = False

    # TODO: Must update the model once a new version is received
    def update_model(self):
        # utils.verify_all_parameters_known()
        pass

    def run(self):
        while self._running:
            # TODO: Must check the metrics against the SLOs and detect violations
            reality_metrics = self.inf_service.process_one_iteration(self.s_description['constraints'])
            reality_row = utils.prepare_samples(pd.DataFrame([reality_metrics]))

            for var in self.s_description['slo_var']:
                current_slo_f = reality_row[var][0]
                self.slo_hist.add(utils.str_to_bool(current_slo_f))
                rebalanced_slo_f = self.slo_hist.average()

                # print(f'Current avg slo f: {self.slo_hist.average()}')
                # print(f'Current avg slo f: {rebalanced_slo_f}')

            if not self.under_unknown_config:

                # TODO: This assumes some links between the parameters and the in_time
                print(utils.get_surprise_for_data(self.model, self.model_VE, reality_row, self.s_description['slo_var']))

                expectation = utils.get_true(utils.infer_slo_fulfillment(self.model_VE, self.s_description['slo_var'],
                                                                         self.s_description['constraints']))
                print(f"Expectation {round(expectation, 2)} vs. reality {round(rebalanced_slo_f, 2)}")
            else:
                print("Evaluating unknown configuration")

                if self.slo_hist.already_x_values():
                    print("gathered sufficient data")

        print(f"Thread {self.inf_service} exited gracefully")


def start_service(s):
    model_path = f"{s['name']}_{DEVICE_NAME}_model.xml"
    model = XMLBIFReader(model_path).get_model()

    if s['name'] == "XYZ":
        service_wrapper = None  # Other services
    else:
        # service: VehicleService = VideoDetector()
        service_wrapper = ServiceWrapper(VideoDetector(), s, model)

    # slo = utils.get_true(utils.infer_slo_fulfillment(VariableElimination(model), s['slo_var'], s['constraints']))

    # Case 1: If SLO fulfillment looks promising then start immediately
    # Case 1.1: Same if not known, we'll find out during operation

    if 0.0 >= 0.00:  # slo >= X
        background_thread = threading.Thread(target=service_wrapper.run, name=s['name'])
        background_thread.daemon = True  # Set the thread as a daemon, so it exits when the main program exits
        background_thread.start()
        # background_thread.__getattribute__('_args')

        print(f"{service_wrapper.inf_service} started detached for expected SLO fulfillment ")  # '{slo}")
        thread_lib.append((background_thread, service_wrapper))
        # utils.print_current_services(thread_lib)
    else:
        print(f"Skipping service due tu low expected SLO fulfillment")  # {slo}")

    # Case 2: However, if it is below the thresh, try to offload


services = [{"name": 'CV', 'slo_var': ["in_time"], 'constraints': {'pixel': '480', 'fps': '15'}}]

for service_description in services:
    print(f"Starting {service_description['name']} by default")
    start_service(service_description)

app = Flask(__name__)


@app.route("/start_service", methods=['POST'])
def start():
    service_d = ast.literal_eval(request.args.get('service_description'))
    start_service(service_d)
    return "success"


@app.route("/stop_all_services", methods=['POST'])
def stop_all():
    global thread_lib
    if len(thread_lib) <= 0:
        print(f"No service threads running locally")
        return "Stopped all threads"

    print(f"Going to stop {len(thread_lib)} threads")

    for bg_thread, task_object in thread_lib:
        task_object.terminate()
    # service_d = ast.literal_eval(request.args.get('service_description'))
    # start_service(service_d)
    thread_lib = []
    return "Stopped all threads"


@app.route('/model_list', methods=['GET'])
def list_files():
    """Endpoint to list files available for download."""
    files = os.listdir(MODEL_DIRECTORY)
    filtered_files = [f for f in files if f.endswith('model.xml')]
    return jsonify(filtered_files)


@app.route('/model/<model_name>', methods=['GET'])
def download_file(model_name):
    """Endpoint to download a specific file."""
    return send_from_directory(MODEL_DIRECTORY, model_name)


def run_server():
    app.run(host='0.0.0.0', port=8080)


run_server()

# background_thread = threading.Thread(target=run_server)
# background_thread.daemon = True
# background_thread.start()
#
# while True:
#     user_input = input()
#
#     if user_input.startswith("start: "):
#         print(eval(user_input[7:]))
