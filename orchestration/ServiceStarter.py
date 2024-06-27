import logging
import threading
import time
import traceback

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

RETRAINING_RATE = 999.0  # Idea: This is a hyperparameter
OFFLOADING_RATE = - 999  # Idea: This is a hyperparameter
TRAINING_BUFFER_SIZE = 150  # Idea: This is a hyperparameter
SLO_HISTORY_BUFFER_SIZE = 70  # Idea: This is a hyperparameter
SLO_COLDSTART_DELAY = 15  # Idea: This is a hyperparameter


class ServiceWrapper(threading.Thread):
    def __init__(self, inf_service: VehicleService, description, model: BayesianNetwork, platoon_members, isolated=False):
        super().__init__()
        self.daemon = True
        self.id = description['id']
        self.type = description['type']

        self.reality_metrics = None
        self._running = True
        self.inf_service = inf_service
        self.s_desc = description
        self.model = model  # TODO: Filter MB with utils.get_mbs_as_bn(model, self.s_desc['slo_vars'])
        self.model_VE = VariableElimination(self.model)
        self.slo_hist = CyclicArray(SLO_HISTORY_BUFFER_SIZE)
        self.metrics_buffer = CyclicArray(TRAINING_BUFFER_SIZE)
        self.isolated = isolated
        self.slo_estimator = SloEstimator(self.model, self.s_desc)
        self.platoon_members = platoon_members
        self.service_assignment = {}

    def terminate(self):
        self._running = False

    def update_model(self, model):
        self.model = model
        self.model_VE = VariableElimination(self.model)
        self.slo_estimator.reload_source_model(self.model)

    def update_isolation(self, isolated):
        self.isolated = isolated

    def update_service_assignment(self, service_assignment):
        self.service_assignment = service_assignment

    def update_platoon(self, platoon):
        self.platoon_members = platoon

    def isolated_service(self):
        while self._running:
            self.reality_metrics = self.inf_service.process_one_iteration(self.s_desc['constraints'])
            self.reality_metrics['isolated'] = self.isolated
            self.inf_service.report_to_mongo(self.reality_metrics)
            self.metrics_buffer.append(self.reality_metrics)

        logger.info(f"M| Thread {self.type}-{self.id} exited gracefully")

    def run(self):
        service_thread = threading.Thread(target=self.isolated_service, daemon=True)
        service_thread.start()

        while self._running:
            time.sleep(1)
            if self.reality_metrics is None:
                continue
            try:
                expectation, reality = self.evaluate_slos(self.reality_metrics)
                evidence_to_retrain = self.metrics_buffer.get_percentage_filled() + np.abs(expectation - reality)
                logger.debug(f"Current evidence to retrain {evidence_to_retrain} / {RETRAINING_RATE}")
                logger.debug(f"For expectation {expectation} vs {reality}")

                if evidence_to_retrain >= RETRAINING_RATE:
                    logger.info(f"M| Asking leader to retrain on {self.metrics_buffer.get_number_items()} samples")
                    df = pd.DataFrame(self.metrics_buffer.get())  # pd.concat(self.metrics_buffer.get(), ignore_index=True)
                    model_file = utils.create_model_name(self.s_desc['type'], DEVICE_NAME)
                    http_client.push_metrics_retrain(model_file, df)  # Metrics are still raw!
                    self.metrics_buffer.clear()

                evidence_to_load_off = (expectation - reality) + (1 - reality)
                logger.debug(f"Current evidence to load off {evidence_to_load_off} / {OFFLOADING_RATE}")

                # TODO: I must split this off again, its getting too big
                if evidence_to_load_off >= OFFLOADING_RATE and self.slo_hist.already_x_values(SLO_COLDSTART_DELAY):
                    other_members = utils.get_all_other_members(self.platoon_members)

                    # How would the local SLO-F be changed if we load off
                    if other_members:
                        local_running_services = utils.get_running_services_for_host(self.service_assignment, utils.get_local_ip())
                        # target_model_name = utils.create_model_name(self.type, DEVICE_NAME)
                        slo_local_estimated_initial = self.slo_estimator.infer_local_slo_f(local_running_services, DEVICE_NAME)
                        slo_local_estimated_offload = self.slo_estimator.infer_local_slo_f(local_running_services, DEVICE_NAME, self.s_desc)
                        logger.debug(f"Estimated SLO fulfillment at origin without offload {slo_local_estimated_initial}")
                        logger.debug(f"Estimated SLO fulfillment at origin after offload {slo_local_estimated_offload}")

                        slo_tradeoff_origin_initial = sum([1 - slo for slo in slo_local_estimated_initial])
                        slo_tradeoff_origin_offload = sum([1 - slo for slo in slo_local_estimated_offload])

                    for vehicle_address in other_members:
                        target_running_services = utils.get_running_services_for_host(self.service_assignment, vehicle_address)
                        target_device_type = utils.conv_ip_to_host_type(vehicle_address)
                        target_model_name = utils.create_model_name(self.type, target_device_type)

                        prometheus_instance_name = vehicle_address
                        if vehicle_address == "192.168.31.20":
                            prometheus_instance_name = "host.docker.internal"
                        slo_target_estimated_offload = self.slo_estimator.infer_target_slo_f(target_model_name, target_running_services,
                                                                                             prometheus_instance_name)
                        slo_target_estimated_initial = self.slo_estimator.infer_local_slo_f(target_running_services, target_device_type)
                        logger.debug(f"Estimated SLO fulfillment at target without offload {slo_target_estimated_initial}")
                        logger.debug(f"Estimated SLO fulfillment at target after offload {slo_target_estimated_offload}")

                        slo_tradeoff_target_initial = sum([1 - slo for slo in slo_target_estimated_initial])
                        slo_tradeoff_target_offload = sum([1 - slo for slo in slo_target_estimated_offload[2]])

                        # TODO: I'd expect the target initial/offload to perform better
                        # TODO: What about the live information?
                        if (slo_tradeoff_target_initial + slo_tradeoff_origin_initial) > (slo_tradeoff_origin_offload + slo_tradeoff_target_offload):
                            logger.info(f"M| Thread {self.type} #{self.id} offloaded to "
                                        f"{utils.conv_ip_to_host_type(vehicle_address)} at address {vehicle_address}")
                            http_client.start_service_remotely(self.s_desc, target_route=vehicle_address)
                            self.terminate()
                            return

            except Exception as e:

                error_traceback = traceback.format_exc()
                print("Error Traceback:")
                print(error_traceback)

                utils.print_in_red(f"ACI Background thread encountered an exception:{e}")
                # self.start()

    def evaluate_slos(self, reality_metrics):
        # TODO: Must also support multiple SLOs
        for var in self.s_desc['slo_vars']:
            # Idea: This should be able to use a fuzzy classifier if the SLOs are fulfilled
            current_slo_f = utils.calculate_slo_fulfillment(var, reality_metrics)
            self.slo_hist.append(current_slo_f)
            rebalanced_slo_f = round(self.slo_hist.average(), 2)
            reality = rebalanced_slo_f

        expectation = utils.get_true(utils.infer_slo_fulfillment(self.model_VE, self.s_desc['slo_vars'],
                                                                 self.s_desc['constraints'] | {
                                                                     "isolated": f'{self.isolated}'}))
        # surprise = utils.get_surprise_for_data(self.model, self.model_VE, reality_row, self.s_desc['slo_vars'])
        # print(f"M| Absolute surprise for sample {surprise}")

        return expectation, reality


def start_service(s_desc, platoon_members, isolated=False):
    model_path = utils.create_model_name(s_desc['type'], DEVICE_NAME)
    model = XMLBIFReader(model_path).get_model()

    if s_desc['type'] == "CV":
        service_wrapper = ServiceWrapper(VideoDetector(), s_desc, model, platoon_members, isolated)
    else:
        raise RuntimeError(f"What is this {s_desc['type']}?")

    service_wrapper.start()
    logger.info(f"M| New thread for {s_desc['type']}-{s_desc['id']} started detached")

    return service_wrapper
