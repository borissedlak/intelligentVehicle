import logging
import threading

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFReader

import utils
from monitor.DeviceMetricReporter import CyclicArray
from orchestration.HttpClient import HttpClient
from services.CV.VideoDetector import VideoDetector
from services.VehicleService import VehicleService

HTTP_SERVER = utils.get_ENV_PARAM('HTTP_SERVER', "127.0.0.1")
DEVICE_NAME = utils.get_ENV_PARAM('DEVICE_NAME', "Unknown")

http_client = HttpClient(HOST=HTTP_SERVER)

MODEL_DIRECTORY = "./"
log = logging.getLogger("vehicle")

RETRAINING_RATE = 0.5  # Idea: This is a hyperparameter
TRAINING_BUFFER_SIZE = 150  # Idea: This is a hyperparameter


class ServiceWrapper:
    def __init__(self, inf_service: VehicleService, description, model: BayesianNetwork):
        self._running = True
        self.inf_service = inf_service
        self.s_description = description
        self.model = model  # TODO: Filter MB with utils.get_mbs_as_bn(model, self.s_description['slo_var'])
        self.model_VE = VariableElimination(self.model)
        self.slo_hist = CyclicArray(100)  # TODO: If at some point I do dynamic adaptations, I must clear this
        self.metrics_buffer = CyclicArray(TRAINING_BUFFER_SIZE)
        self.under_unknown_config = not utils.verify_all_parameters_known(model,
                                                                          pd.DataFrame([self.s_description['constraints']]),
                                                                          list(self.s_description['constraints'].keys()))

    def terminate(self):
        self._running = False

    def update_model(self, model):
        self.model = model
        self.model_VE = VariableElimination(self.model)

    def run(self):
        while self._running:
            reality_metrics = self.inf_service.process_one_iteration(self.s_description['constraints'])
            self.metrics_buffer.append(reality_metrics)
            # reality_row = utils.prepare_samples(pd.DataFrame([reality_metrics]))

            for var in self.s_description['slo_var']:
                current_slo_f = utils.calculate_slo_fulfillment(var, reality_metrics)
                # current_slo_f = reality_row[var][0]
                self.slo_hist.append(current_slo_f)
                rebalanced_slo_f = round(self.slo_hist.average(), 2)
                reality = rebalanced_slo_f  # TODO: Must also support multiple SLOs

            expectation = round(utils.get_true(utils.infer_slo_fulfillment(self.model_VE, self.s_description['slo_var'],
                                                                           self.s_description['constraints'])), 2)
            # surprise = utils.get_surprise_for_data(self.model, self.model_VE, reality_row, self.s_description['slo_var'])
            # print(f"M| Absolute surprise for sample {surprise}")

            evidence_to_retrain = self.metrics_buffer.get_percentage_filled() + np.abs(expectation - reality)
            log.debug(f"Current evidence to retrain {evidence_to_retrain} / {RETRAINING_RATE}")
            log.debug(f"For expectation {expectation} vs {reality}")

            if evidence_to_retrain >= RETRAINING_RATE:
                log.info(f"M| Asking leader to retrain on {self.metrics_buffer.get_number_items()} samples")
                df = pd.DataFrame(self.metrics_buffer.get())  # pd.concat(self.metrics_buffer.get(), ignore_index=True)
                model_file = utils.create_model_name(self.s_description['name'], DEVICE_NAME)
                http_client.push_metrics_retrain(model_file, df)  # Metrics are still raw!
                self.metrics_buffer.clear()

        log.info(f"M| Thread {self.inf_service} exited gracefully")


def start_service(s):
    # TODO: This should pull the latest model before starting
    # TODO: However, the exceptions are that the member is also leader or its already up-to-date

    model_path = utils.create_model_name(s['name'], DEVICE_NAME)
    model = XMLBIFReader(model_path).get_model()

    if s['name'] == "XYZ":
        service_wrapper = None  # Other services
    else:
        service_wrapper = ServiceWrapper(VideoDetector(), s, model)

    # slo = utils.get_true(utils.infer_slo_fulfillment(VariableElimination(model), s['slo_var'], s['constraints']))

    # Case 1: If SLO fulfillment looks promising then start immediately
    # Case 1.1: Same if not known, we'll find out during operation

    if 0.0 >= 0.00:  # slo >= X
        background_thread = threading.Thread(target=service_wrapper.run, name=s['name'])
        background_thread.daemon = True  # Set the thread as a daemon, so it exits when the main program exits
        background_thread.start()

        log.info(f"M| {service_wrapper.inf_service} started detached for expected SLO fulfillment ")  # '{slo}")
        return background_thread, service_wrapper
        # utils.print_current_services(thread_lib)
    else:
        log.info(f"M| Skipping service due tu low expected SLO fulfillment")  # {slo}")

    # Case 2: However, if it is below the thresh, try to offload
