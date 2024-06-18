import os

from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

import utils
from services.CV.VideoDetector import VideoDetector

DEVICE_NAME = os.environ.get('DEVICE_NAME')
if DEVICE_NAME:
    print(f'Found ENV value for DEVICE_NAME: {DEVICE_NAME}')
else:
    DEVICE_NAME = "Unknown"
    print(f"Didn't find ENV value for DEVICE_NAME, default to: {DEVICE_NAME}")

services = [{"name": 'CV', 'slo_var': ["in_time"], 'constraints': {'pixel': '480', 'fps': '10'}}]


def infer_slo_fulfillment(bn_model, slo_variables, constraints=None):
    if constraints is None:
        constraints = {}
    evidence = constraints  # | {'device_type': device_type}
    ve = VariableElimination(bn_model)
    result = ve.query(variables=slo_variables, evidence=evidence)

    return result


for s in services:
    if s['name'] == "CV":
        # Case 1: If SLO fulfillment looks promising then start immediately
        # Case 1.1: Same if not known, we'll find out during operation

        cv_service = VideoDetector()
        model_path = f"CV_{DEVICE_NAME}_model.xml"
        model = XMLBIFReader(model_path).get_model()

        slo = utils.get_true(infer_slo_fulfillment(model, s['slo_var'], s['constraints']))
        print(slo)

        # Case 2: However, if it is below the thresh, try to offload
