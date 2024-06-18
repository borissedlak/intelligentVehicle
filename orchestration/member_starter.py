import os
import threading
import time

from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

import utils
from services.CV.VideoDetector import VideoDetector
from services.VehicleService import VehicleService

DEVICE_NAME = os.environ.get('DEVICE_NAME')
if DEVICE_NAME:
    print(f'Found ENV value for DEVICE_NAME: {DEVICE_NAME}')
else:
    DEVICE_NAME = "Unknown"
    print(f"Didn't find ENV value for DEVICE_NAME, default to: {DEVICE_NAME}")

thread_lib = []


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
    print(slo)

    # Case 1: If SLO fulfillment looks promising then start immediately
    # Case 1.1: Same if not known, we'll find out during operation

    if slo >= 0.90:
        background_thread = threading.Thread(target=processing_loop, args=(service, s['constraints']))
        background_thread.daemon = True  # Set the thread as a daemon, so it exits when the main program exits
        background_thread.start()
        thread_lib.append(background_thread)

    # Case 2: However, if it is below the thresh, try to offload


services = [{"name": 'CV', 'slo_var': ["in_time"], 'constraints': {'pixel': '480', 'fps': '5'}},
            {"name": 'CV', 'slo_var': ["in_time"], 'constraints': {'pixel': '480', 'fps': '10'}}]

for service_description in services:
    start_service(service_description)

while True:
    for t in thread_lib:
        print(t.is_alive())
        time.sleep(1)
        t.join()
