import logging
import random
import threading
import time
import traceback

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFReader
from prometheus_client import Gauge, push_to_gateway, CollectorRegistry

import utils
from monitor.DeviceMetricReporter import CyclicArray
from orchestration.HttpClient import HttpClient
from orchestration.SloEstimator import SloEstimator
from services.CV.YoloDetector import YoloDetector
from services.LI.LidarProcessor import LidarProcessor
from services.QR.QrDetector import QrDetector
from services.VehicleService import VehicleService

LEADER_HOST = utils.get_ENV_PARAM('LEADER_HOST', "127.0.0.1")
DEVICE_NAME = utils.get_ENV_PARAM('DEVICE_NAME', "Unknown")

http_client = HttpClient(DEFAULT_HOST=LEADER_HOST)

MODEL_DIRECTORY = "./"
logger = logging.getLogger("vehicle")

RETRAINING_RATE = 1.0  # Idea: This is a hyperparameter
OFFLOADING_RATE = 0.2  # Idea: This is a hyperparameter
TRAINING_BUFFER_SIZE = 120  # Idea: This is a hyperparameter
SLO_HISTORY_BUFFER_SIZE = 75  # Idea: This is a hyperparameter
SLO_COLDSTART_DELAY = 30 + random.randint(0, 9)  # Idea: This is a hyperparameter

registry = CollectorRegistry()
slo_fulfillment_p = Gauge('slo_f', 'Current SLO fulfillment', ['id', 'host', 'device_name'], registry=registry)


class ServiceWrapper(threading.Thread):
    def __init__(self, inf_service: VehicleService, description, model: BayesianNetwork, platoon_members, isolated=False):
        super().__init__()
        self.daemon = True
        self.id = description['id']
        self.type = description['type']
        self.inf_service = inf_service
        self._running = True

        self.reality_metrics = None
        self.s_desc = description
        self.model = utils.get_mbs_as_bn(model, self.s_desc['slo_vars'])  # Write: improves inference time
        self.model_VE = VariableElimination(self.model)
        self.slo_hist = CyclicArray(SLO_HISTORY_BUFFER_SIZE)
        self.metrics_buffer = CyclicArray(TRAINING_BUFFER_SIZE)
        self.isolated = isolated
        self.slo_estimator = SloEstimator(self.model, self.s_desc)
        self.platoon_members = platoon_members
        self.local_ip = utils.get_local_ip()
        self.is_leader = utils.am_I_the_leader(self.platoon_members, self.local_ip)
        self.service_assignment = {}

    def reset_slo_history(self):
        self.slo_hist = CyclicArray(SLO_HISTORY_BUFFER_SIZE)

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
        self.is_leader = utils.am_I_the_leader(self.platoon_members, utils.get_local_ip())

    def isolated_service(self):
        while self._running:
            self.reality_metrics = self.inf_service.process_one_iteration(self.s_desc['constraints'])
            self.reality_metrics['isolated'] = self.isolated
            # Write: This changes the local perception of how well I'm performing or what I'm supposed to do
            self.reality_metrics['is_leader'] = self.is_leader
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

                # service_load.labels(device_name=DEVICE_NAME).set(reality)
                slo_fulfillment_p.labels(id=f"{self.type}-{self.id}", host=self.local_ip, device_name=DEVICE_NAME).set(reality)
                push_to_gateway(f'{self.platoon_members[0]}:9091', job='batch_job', registry=registry)

                evidence_to_retrain = self.metrics_buffer.get_percentage_filled() + np.abs(expectation - reality)
                logger.debug(f"Current evidence to retrain {evidence_to_retrain} / {RETRAINING_RATE}")
                logger.debug(f"For expectation {expectation} vs {reality}")

                if evidence_to_retrain >= RETRAINING_RATE:
                    logger.info(f"M| Asking leader to retrain {self.type}-{self.id} on {self.metrics_buffer.get_number_items()} samples")
                    self.train_remotely()

                evidence_to_load_off = (expectation - reality) + (1 - reality)
                logger.debug(f"Current evidence to load off {evidence_to_load_off} / {OFFLOADING_RATE}")

                if evidence_to_load_off >= OFFLOADING_RATE and self.slo_hist.already_x_values(SLO_COLDSTART_DELAY):
                    other_members = utils.get_all_other_members(self.platoon_members)

                    if len(other_members) == 0:
                        logger.info(f"M| Thread {self.type}-{self.id} would like to offload, but no other members in platoon")
                        continue

                    offload_gain_list = self.estimate_slos_offload(other_members)
                    target, gain = max(offload_gain_list, key=lambda x: x[1])
                    if (expectation - reality) + gain > 0:
                        logger.info(f"M| Push metrics for thread {self.type}-{self.id} before loading off")
                        self.train_remotely(asynchronous=True)
                        logger.info(f"M| Thread {self.type}-{self.id} offloaded to {utils.conv_ip_to_host_type(target)} at {target}")
                        http_client.start_service_remotely(self.s_desc, target_route=target)
                        self.terminate()
                        return

                    logger.info(f"M| Thread {self.type}-{self.id} found no beneficial hosting destination")

            except Exception as e:
                error_traceback = traceback.format_exc()
                print("Error Traceback:")
                print(error_traceback)
                utils.print_in_red(f"ACI Background thread encountered an exception:{e}")

    def train_remotely(self, asynchronous=False):
        df = pd.DataFrame(self.metrics_buffer.get())  # pd.concat(self.metrics_buffer.get(), ignore_index=True)
        model_file = utils.create_model_name(self.s_desc['type'], DEVICE_NAME)
        http_client.push_metrics_retrain(model_file, df, asynchronous=asynchronous)  # Metrics are still raw!
        self.metrics_buffer.clear()

    def evaluate_slos(self, reality_metrics):
        # Idea: This should be able to use a fuzzy classifier if the SLOs are fulfilled
        current_slo_f = utils.check_slos_fulfilled(self.s_desc['slo_vars'], reality_metrics)
        self.slo_hist.append(current_slo_f)
        rebalanced_slo_f = self.slo_hist.average()

        expectation = utils.get_true(utils.infer_slo_fulfillment(self.model_VE, self.s_desc['slo_vars'],
                                                                 self.s_desc['constraints'] | {"isolated": f'{self.isolated}'}))
        # surprise = utils.get_surprise_for_data(self.model, self.model_VE, reality_row, self.s_desc['slo_vars'])
        # print(f"M| Absolute surprise for sample {surprise}")

        return expectation, rebalanced_slo_f

    def estimate_slos_offload(self, other_members):
        # How would the SLO-F at the target device change if we deploy an additional service there
        local_running_services = utils.get_running_services_for_host(self.service_assignment, utils.get_local_ip())

        slo_local_estimated_initial = self.slo_estimator.infer_local_slo_f(local_running_services, DEVICE_NAME,
                                                                           target_is_leader=self.is_leader)
        slo_local_estimated_offload = self.slo_estimator.infer_local_slo_f(local_running_services, DEVICE_NAME, self.s_desc,
                                                                           target_is_leader=self.is_leader)
        logger.debug(f"Estimated SLO fulfillment at origin without offload {slo_local_estimated_initial}")
        logger.debug(f"Estimated SLO fulfillment at origin after offload {slo_local_estimated_offload}")

        slo_tradeoff_origin_initial = sum([1 - slo for slo in slo_local_estimated_initial])
        slo_tradeoff_origin_offload = sum([1 - slo for slo in slo_local_estimated_offload])

        target_slo_f = []
        for vehicle_address in other_members:
            target_is_leader = vehicle_address == self.platoon_members[0]
            target_running_services = utils.get_running_services_for_host(self.service_assignment, vehicle_address)
            target_device_type = utils.conv_ip_to_host_type(vehicle_address)
            target_model_name = utils.create_model_name(self.type, target_device_type)

            prometheus_instance_name = vehicle_address if vehicle_address != "192.168.31.20" else "host.docker.internal"
            slo_target_estimated_offload = self.slo_estimator.infer_target_slo_f(target_model_name, target_running_services,
                                                                                 prometheus_instance_name, target_is_leader)
            slo_target_estimated_initial = self.slo_estimator.infer_local_slo_f(target_running_services, target_device_type,
                                                                                target_is_leader)
            logger.debug(f"Estimated SLO fulfillment at target without offload {slo_target_estimated_initial}")
            logger.debug(f"Estimated SLO fulfillment at target after offload {slo_target_estimated_offload}")

            slo_tradeoff_target_initial = sum([1 - slo for slo in slo_target_estimated_initial])
            slo_tradeoff_target_offload = sum([1 - slo for slo in slo_target_estimated_offload[2]])

            offload_tradeoff_gain = ((slo_tradeoff_target_initial + slo_tradeoff_origin_initial) -
                                     (slo_tradeoff_origin_offload + slo_tradeoff_target_offload))
            target_slo_f.append((vehicle_address, offload_tradeoff_gain))

        return target_slo_f


def start_service(s_desc, platoon_members, isolated=False):
    model_path = utils.create_model_name(s_desc['type'], DEVICE_NAME)
    model = XMLBIFReader("models/" + model_path).get_model()

    if s_desc['type'] == "CV":
        service_wrapper = ServiceWrapper(YoloDetector(), s_desc, model, platoon_members, isolated)
    elif s_desc['type'] == "QR":
        service_wrapper = ServiceWrapper(QrDetector(), s_desc, model, platoon_members, isolated)
    elif s_desc['type'] == "LI":
        service_wrapper = ServiceWrapper(LidarProcessor(), s_desc, model, platoon_members, isolated)
    else:
        raise RuntimeError(f"What is this {s_desc['type']}?")

    service_wrapper.start()
    logger.info(f"M| New thread for {s_desc['type']}-{s_desc['id']} started detached")

    return service_wrapper
