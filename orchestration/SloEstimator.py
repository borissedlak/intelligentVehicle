import logging

import numpy as np
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

    # @utils.print_execution_time  # takes 400ms
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

    # @utils.print_execution_time  # takes 120ms
    def calc_weighted_slo_f(self, p_dist_hw, dest_model=None, isolated="False", shift=[0, 0, 0]):
        if dest_model is None:
            dest_model = self.source_model
        dest_model_VE = VariableElimination(dest_model)

        sum_slo_f = np.zeros((4, 4, 4))
        for i, _ in enumerate(p_dist_hw['cpu']):
            cpu_index = (i + shift[0]) if i + shift[0] <= utils.NUMBER_OF_BINS - 1 else utils.NUMBER_OF_BINS - 1
            for j, _ in enumerate(p_dist_hw['gpu']):
                gpu_index = (j + shift[1]) if j + shift[1] <= utils.NUMBER_OF_BINS - 1 else utils.NUMBER_OF_BINS - 1
                for k, _ in enumerate(p_dist_hw['memory']):
                    mem_index = (k + shift[2]) if k + shift[2] <= utils.NUMBER_OF_BINS - 1 else utils.NUMBER_OF_BINS - 1

                    p_cumm = p_dist_hw['cpu'][i] * p_dist_hw['gpu'][j] * p_dist_hw['memory'][k]
                    slo_f_i = utils.get_true(
                        utils.infer_slo_fulfillment(dest_model_VE, self.s_desc['slo_vars'], self.s_desc['constraints'] |
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

        return np.sum(sum_slo_f)

    # @utils.print_execution_time  # takes 120ms
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
        logger.debug(f"M| Expected SLO fulfillment for running {s_desc['name']} locally isolated {slof_local_isolated}")
        return hw_predictions, slof_local_isolated

    # @utils.print_execution_time  # takes 150ms
    def get_shifted_hw_predictions(self, origin_load_p, target_model, target_host):
        dest_current_load = model_trainer.get_latest_load(instance=target_host)
        dest_current_load_cat = model_trainer.convert_prometheus_to_category(dest_current_load)
        logger.debug(f"M| Current load for target device classified into {dest_current_load_cat}")
        if target_host == "host.docker.internal" or target_host == "192.168.31.20":
            dest_current_load_cat[1] = -1
        return self.calc_weighted_slo_f(origin_load_p, dest_model=target_model, shift=(dest_current_load_cat + 1), isolated="False")

    # Write: I might optimize the runtime a bit, but I can also compare the runtime of shifter vs. conv
    # @utils.print_execution_time  # takes 100ms
    def get_conv_hw_predictions(self, origin_load_p, target_model_is, target_device, target_running_services):
        if not target_running_services:
            return [self.calc_weighted_slo_f(origin_load_p, dest_model=target_model_is, isolated="True")]

        target_conv_load = origin_load_p
        target_models = [target_model_is]
        for s_desc in target_running_services:
            target_model = XMLBIFReader(utils.create_model_name(s_desc['name'], target_device)).get_model()
            target_models.append(target_model)
            s_load_p, _ = self.get_isolated_hw_predictions(VariableElimination(target_model), s_desc)

            for var in ['cpu', 'gpu', 'memory']:
                var_conv = utils.compress_into_n_bins(np.convolve(target_conv_load[var], s_load_p[var]))
                target_conv_load[var] = var_conv

        target_slo_f = []
        for model in target_models:
            target_slo_f.append(self.calc_weighted_slo_f(target_conv_load, dest_model=model, isolated="False"))

        return target_slo_f


if __name__ == "__main__":
    logging.getLogger("vehicle").setLevel(logging.DEBUG)

    local_model_name = utils.create_model_name("CV", "Orin")
    local_model = XMLBIFReader(local_model_name).get_model()

    s_description = {"name": 'CV', 'slo_vars': ["in_time"], 'constraints': {'pixel': '480', 'fps': '10'}}
    estimator = SloEstimator(local_model, service_desc=s_description)

    target_running_s = []
    print(estimator.infer_target_slo_f(local_model_name, "192.168.31.183", target_running_s))

    target_running_s = [s_description]
    print(estimator.infer_target_slo_f(local_model_name, "192.168.31.183", target_running_s))

    target_running_s.append(s_description)
    print(estimator.infer_target_slo_f(local_model_name, "192.168.31.183", target_running_s))

    target_running_s.append(s_description)
    print(estimator.infer_target_slo_f(local_model_name, "192.168.31.183", target_running_s))
