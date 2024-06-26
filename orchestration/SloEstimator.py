import logging

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

import utils
from orchestration import model_trainer

logger = logging.getLogger("vehicle")
logging.getLogger("vehicle").setLevel(logging.DEBUG)


class SloEstimator:
    def __init__(self, source_model, service_desc):
        self.source_model = source_model
        self.s_desc = service_desc
        self.model_VE = VariableElimination(self.source_model)

    def reload_source_model(self, source_model):
        self.source_model = source_model

    @utils.print_execution_time
    def infer_target_slo_f(self, target_model_name, target_host="localhost", target_running_services=None):
        dest_model = XMLBIFReader(target_model_name).get_model()
        dest_device = utils.conv_ip_to_host_type(target_host)
        # Write: I might create a cube representation of the solution space
        hw_load_p, slof_local_isolated = self.get_isolated_hw_predictions()

        # Write: The problem is that the load does not rise linear with more services
        # Idea: So what I can do is take the one that is worse to make a conservative prediction
        prediction_shifted = self.get_shifted_hw_predictions(hw_load_p, dest_model, target_host)
        logger.debug(f"M| Predictions for SLO fulfillment once load shifted {prediction_shifted}")

        if target_running_services is None:
            target_running_services = []
        prediction_conv = self.get_conv_hw_predictions(hw_load_p, dest_model, dest_device, target_running_services)
        logger.debug(f"M| Predictions for SLO fulfillment when conv with existing services {prediction_conv}")

        return slof_local_isolated, prediction_shifted, prediction_conv

    @utils.print_execution_time
    def calc_weighted_slo_f(self, p_dist_hw, dest_model=None, isolated="False", shift=[0, 0, 0]):
        if dest_model is None:
            dest_model = self.source_model

        sum_slo_f = np.zeros((4, 4, 4))
        sum_0_5 = 0
        for i, _ in enumerate(p_dist_hw['cpu']):
            cpu_index = i + shift[0]
            for j, _ in enumerate(p_dist_hw['gpu']):
                gpu_index = (j + shift[1])
                for k, _ in enumerate(p_dist_hw['memory']):
                    mem_index = (k + shift[2])

                    # TODO: df conversion too slow, try to get this working for not df
                    if not utils.verify_all_parameters_known(dest_model, pd.DataFrame([{'cpu': f'{cpu_index}'}]), ['cpu']):
                        cpu_index = len(p_dist_hw['cpu']) - 1
                    if not utils.verify_all_parameters_known(dest_model, pd.DataFrame([{'gpu': f'{gpu_index}'}]), ['gpu']):
                        gpu_index = len(p_dist_hw['gpu']) - 1
                    if not utils.verify_all_parameters_known(dest_model, pd.DataFrame([{'memory': f'{mem_index}'}]), ['memory']):
                        mem_index = len(p_dist_hw['memory']) - 1

                    p_cumm = p_dist_hw['cpu'][i] * p_dist_hw['gpu'][j] * p_dist_hw['memory'][k]
                    # print(i, shift[0], cpu_index)
                    slo_f_i = utils.get_true(
                        utils.infer_slo_fulfillment(self.model_VE, self.s_desc['slo_vars'], self.s_desc['constraints'] |
                                                    {'cpu': f'{cpu_index}', 'gpu': f'{gpu_index}', 'memory': f'{mem_index}',
                                                     'isolated': isolated}))
                    # print(slo_f_i)
                    # print(utils.get_true(
                    #     utils.infer_slo_fulfillment(self.model_VE, self.s_desc['slo_vars'], self.s_desc['constraints'] |
                    #                                 {'cpu': f'{1}', 'gpu': f'{gpu_index}', 'memory': f'{mem_index}',
                    #                                  'isolated': isolated})))
                    # print(utils.get_true(
                    #     utils.infer_slo_fulfillment(self.model_VE, self.s_desc['slo_vars'], self.s_desc['constraints'] |
                    #                                 {'cpu': f'{2}', 'gpu': f'{gpu_index}', 'memory': f'{mem_index}',
                    #                                  'isolated': isolated})))
                    # print(utils.get_true(
                    #     utils.infer_slo_fulfillment(self.model_VE, self.s_desc['slo_vars'], self.s_desc['constraints'] |
                    #                                 {'cpu': f'{3}', 'gpu': f'{gpu_index}', 'memory': f'{mem_index}',
                    #                                  'isolated': isolated})))
                    weighted_p = p_cumm * slo_f_i
                    sum_slo_f[i, j, k] = weighted_p

                    if slo_f_i == 0.5:
                        sum_0_5 += 1

        # print(f"Number of 0.5 is {sum_0_5}")
        return np.sum(sum_slo_f)

    @utils.print_execution_time
    def get_isolated_hw_predictions(self, model_VE=None, s_desc=None):
        if model_VE is None or s_desc is None:  # No values means take the origin description
            model_VE = self.model_VE
            s_desc = self.s_desc
        hw_predictions = {}
        for var in ['cpu', 'gpu', 'memory']:
            hw_expectation_isolated = utils.infer_slo_fulfillment(model_VE, [var], s_desc['constraints'] | {'isolated': 'True'})
            hw_distribution = hw_expectation_isolated.values
            hw_predictions = hw_predictions | {var: hw_distribution}

        logger.debug(f"M| Predictions for isolated hardware consumption {hw_predictions}")
        slof_local_isolated = self.calc_weighted_slo_f(hw_predictions, isolated="True")
        logger.debug(f"M| Expected SLO fulfillment for running {s_description['name']} locally isolated {slof_local_isolated}")
        return hw_predictions, slof_local_isolated

    @utils.print_execution_time
    def get_shifted_hw_predictions(self, origin_load_p, target_model, target_host):
        dest_current_load = model_trainer.get_latest_load(instance=target_host)
        dest_current_load_cat = model_trainer.convert_prometheus_to_category(dest_current_load)
        logger.debug(f"M| Current load for target device classified into {dest_current_load_cat}")
        if target_host == "host.docker.internal" or target_host == "192.168.31.20":
            dest_current_load_cat[1] = -1
        return self.calc_weighted_slo_f(origin_load_p, dest_model=target_model, shift=(dest_current_load_cat + 1), isolated="False")

    # TODO: I must take the isolated load at the origin and convolve with any service that is running there
    #  but for this I must first load all their models and then execute the get_isolated_hw_predictions
    @utils.print_execution_time
    def get_conv_hw_predictions(self, origin_load_p, target_model_is, target_device, target_running_services):
        if not target_running_services:
            return self.calc_weighted_slo_f(origin_load_p, dest_model=target_model_is, isolated="True")

        target_conv_load = origin_load_p
        target_models = []
        for s_desc in target_running_services:
            target_model = XMLBIFReader(utils.create_model_name(s_desc['name'], target_device)).get_model()
            target_models.append(target_model)
            s_load_p, _ = self.get_isolated_hw_predictions(VariableElimination(target_model), s_desc)

            for var in ['cpu', 'gpu', 'memory']:
                var_conv = utils.compress_into_n_bins(np.convolve(target_conv_load[var], s_load_p[var]))
                target_conv_load[var] = var_conv

        return self.calc_weighted_slo_f(target_conv_load, dest_model=target_model_is, isolated="False")


if __name__ == "__main__":
    local_model_name = utils.create_model_name("CV", "Orin")
    local_model = XMLBIFReader(local_model_name).get_model()

    s_description = {"name": 'CV', 'slo_vars': ["in_time"], 'constraints': {'pixel': '480', 'fps': '5'}}
    estimator = SloEstimator(local_model, service_desc=s_description)

    target_running_services = [{"name": 'CV', 'slo_vars': ["in_time"], 'constraints': {'pixel': '480', 'fps': '5'}}]
    print(estimator.infer_target_slo_f(local_model_name, "192.168.31.183", target_running_services))
