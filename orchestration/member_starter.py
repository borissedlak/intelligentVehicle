import ast
import os
import threading

from flask import Flask, request
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

import utils
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


def infer_slo_fulfillment(bn_model, slo_variables, constraints=None):
    if constraints is None:
        constraints = {}
    evidence = constraints  # | {'device_type': device_type}
    ve = VariableElimination(bn_model)
    result = ve.query(variables=slo_variables, evidence=evidence)

    return result


def processing_loop(inf_service: VehicleService, constraints):
    while True:
        inf_service.process_one_iteration(constraints)


def start_service(s):
    if s['name'] == "XYZ":
        service: VehicleService = VideoDetector()  # Other services
    else:
        service: VehicleService = VideoDetector()

    model_path = f"CV_{DEVICE_NAME}_model.xml"
    model = XMLBIFReader(model_path).get_model()

    slo = utils.get_true(infer_slo_fulfillment(model, s['slo_var'], s['constraints']))

    # Case 1: If SLO fulfillment looks promising then start immediately
    # Case 1.1: Same if not known, we'll find out during operation

    if slo >= 0.80:
        background_thread = threading.Thread(target=processing_loop, args=(service, s['constraints']))
        background_thread.daemon = True  # Set the thread as a daemon, so it exits when the main program exits
        background_thread.start()
        print(f"{s['name']} started detached for expected SLO fulfillment {slo}")
        thread_lib.append(background_thread)
    else:
        print(f"Skipping service due tu low expected SLO fulfillment {slo}")

    # Case 2: However, if it is below the thresh, try to offload


services = [{"name": 'CV', 'slo_var': ["in_time"], 'constraints': {'pixel': '480', 'fps': '5'}},
            {"name": 'CV', 'slo_var': ["in_time"], 'constraints': {'pixel': '480', 'fps': '10'}}]

for service_description in services:
    print(f"Starting {service_description['name']} by default")
    # start_service(service_description)

app = Flask(__name__)


@app.route("/start_service", methods=['POST'])
def start():
    service_d = ast.literal_eval(request.args.get('service_description'))
    start_service(service_d)
    return "success"


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
