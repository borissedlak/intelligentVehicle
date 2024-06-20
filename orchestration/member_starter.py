import threading
import warnings

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

# thread_lib = []
http_client = HttpClient(HOST=HTTP_SERVER)

MODEL_DIRECTORY = "./"
warnings.filterwarnings("ignore", category=Warning, module='pgmpy')


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

    def update_model(self, model):
        self.model = model
        self.model_VE = VariableElimination(self.model)

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

            # TODO: Metrics should be buffered for x rows before send
            model_file = utils.create_model_name(self.s_description['name'], DEVICE_NAME)
            http_client.push_metrics_retrain(model_file, reality_row)

            if not self.under_unknown_config:

                # TODO: This assumes some links between the parameters and the in_time
                surprise = utils.get_surprise_for_data(self.model, self.model_VE, reality_row, self.s_description['slo_var'])
                print(f"M| Absolute surprise for sample {surprise}")

                expectation = utils.get_true(utils.infer_slo_fulfillment(self.model_VE, self.s_description['slo_var'],
                                                                         self.s_description['constraints']))
                print(f"M| Expectation {round(expectation, 2)} vs. reality {round(rebalanced_slo_f, 2)}")
            else:
                print("Evaluating unknown configuration")

                if self.slo_hist.already_x_values():
                    print("gathered sufficient data")

        print(f"M| Thread {self.inf_service} exited gracefully")


def start_service(s):
    model_path = utils.create_model_name(s['name'], DEVICE_NAME)
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

        print(f"M| {service_wrapper.inf_service} started detached for expected SLO fulfillment ")  # '{slo}")
        return background_thread, service_wrapper
        # utils.print_current_services(thread_lib)
    else:
        print(f"M| Skipping service due tu low expected SLO fulfillment")  # {slo}")

    # Case 2: However, if it is below the thresh, try to offload
