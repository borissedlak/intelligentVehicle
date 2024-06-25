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
from orchestration.SloEstimator import SloEstimator
from services.CV.VideoDetector import VideoDetector
from services.VehicleService import VehicleService

LEADER_HOST = utils.get_ENV_PARAM('LEADER_HOST', "127.0.0.1")
DEVICE_NAME = utils.get_ENV_PARAM('DEVICE_NAME', "Unknown")

http_client = HttpClient(DEFAULT_HOST=LEADER_HOST)

MODEL_DIRECTORY = "./"
logger = logging.getLogger("vehicle")

RETRAINING_RATE = 1.0  # Idea: This is a hyperparameter
OFFLOADING_RATE = 0.3  # Idea: This is a hyperparameter
TRAINING_BUFFER_SIZE = 150  # Idea: This is a hyperparameter
SLO_HISTORY_BUFFER_SIZE = 70  # Idea: This is a hyperparameter


class ServiceWrapper(threading.Thread):
    def __init__(self, inf_service: VehicleService, description, model: BayesianNetwork, platoon_members, isolated=False):
        super().__init__()
        self.daemon = True
        self.name = description['name']
        self._running = True
        self.inf_service = inf_service
        self.s_description = description
        self.model = model  # TODO: Filter MB with utils.get_mbs_as_bn(model, self.s_description['slo_vars'])
        self.model_VE = VariableElimination(self.model)
        self.slo_hist = CyclicArray(SLO_HISTORY_BUFFER_SIZE)
        self.metrics_buffer = CyclicArray(TRAINING_BUFFER_SIZE)
        self.under_unknown_config = not utils.verify_all_parameters_known(model,
                                                                          pd.DataFrame([self.s_description['constraints']]),
                                                                          list(self.s_description['constraints'].keys()))
        self.isolated = isolated
        self.slo_estimator = SloEstimator(self.model, self.s_description)
        self.platoon_members = platoon_members

    def terminate(self):
        self._running = False

    def update_model(self, model):
        self.model = model
        self.model_VE = VariableElimination(self.model)
        self.slo_estimator.reload_source_model(self.model)

    def update_isolation(self, isolated):
        self.isolated = isolated

    def update_platoon(self, platoon):
        self.platoon_members = platoon

    # TODO: Upon error, restart the loop
    def run(self):
        while self._running:
            try:
                reality_metrics = self.inf_service.process_one_iteration(self.s_description['constraints'])
                reality_metrics['isolated'] = self.isolated
                self.inf_service.report_to_mongo(reality_metrics)
                self.metrics_buffer.append(reality_metrics)
                # reality_row = utils.prepare_samples(pd.DataFrame([reality_metrics]))

                for var in self.s_description['slo_vars']:
                    # Idea: This should be able to use a fuzzy classifier if the SLOs are fulfilled
                    current_slo_f = utils.calculate_slo_fulfillment(var, reality_metrics)
                    self.slo_hist.append(current_slo_f)
                    rebalanced_slo_f = round(self.slo_hist.average(), 2)
                    reality = rebalanced_slo_f  # TODO: Must also support multiple SLOs

                expectation = utils.get_true(utils.infer_slo_fulfillment(self.model_VE, self.s_description['slo_vars'],
                                                                         self.s_description['constraints'] | {"isolated": f'{self.isolated}'}))
                # surprise = utils.get_surprise_for_data(self.model, self.model_VE, reality_row, self.s_description['slo_vars'])
                # print(f"M| Absolute surprise for sample {surprise}")

                evidence_to_retrain = self.metrics_buffer.get_percentage_filled() + np.abs(expectation - reality)
                logger.debug(f"Current evidence to retrain {evidence_to_retrain} / {RETRAINING_RATE}")
                logger.debug(f"For expectation {expectation} vs {reality}")

                # if evidence_to_retrain >= RETRAINING_RATE:
                #     logger.info(f"M| Asking leader to retrain on {self.metrics_buffer.get_number_items()} samples")
                #     df = pd.DataFrame(self.metrics_buffer.get())  # pd.concat(self.metrics_buffer.get(), ignore_index=True)
                #     model_file = utils.create_model_name(self.s_description['name'], DEVICE_NAME)
                #     http_client.push_metrics_retrain(model_file, df)  # Metrics are still raw!
                #     self.metrics_buffer.clear()

                evidence_to_load_off = (expectation - reality) + 0  # TODO: Some other factors
                logger.debug(f"Current evidence to load off {evidence_to_load_off} / {OFFLOADING_RATE}")

                if evidence_to_load_off >= OFFLOADING_RATE:
                    for vehicle_address in self.platoon_members:
                        service_name = utils.create_model_name("CV", "Laptop")
                        slo_target_estimated = self.slo_estimator.infer_target_slo_f(service_name, vehicle_address)

                        print(slo_target_estimated)

            except Exception as e:

                error_traceback = traceback.format_exc()
                print("Error Traceback:")
                print(error_traceback)

                util.print_in_red(f"ACI Background thread encountered an exception:{e}")
                return self.start()

        logger.info(f"M| Thread {self.inf_service} exited gracefully")


def start_service(s, platoon_members, isolated=False):
    model_path = utils.create_model_name(s['name'], DEVICE_NAME)
    model = XMLBIFReader(model_path).get_model()

    if s['name'] == "CV":
        service_wrapper = ServiceWrapper(VideoDetector(), s, model, platoon_members, isolated)
    else:
        raise RuntimeError(f"What is this {s['name']}?")

    service_wrapper.start()
    # background_thread = threading.Thread(target=service_wrapper.run, name=s['name'])
    # background_thread.daemon = True  # Set the thread as a daemon, so it exits when the main program exits
    # background_thread.start()

    logger.info(f"M| {service_wrapper.inf_service} started detached for expected SLO fulfillment ")

    slo = utils.get_true(utils.infer_slo_fulfillment(VariableElimination(model), s['slo_vars'], s['constraints']))
    logger.debug(f"M| Expected SLO fulfillment is {slo}")

    return service_wrapper
