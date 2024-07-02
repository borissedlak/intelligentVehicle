import logging

import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

import utils
from orchestration.models import model_trainer

logger = logging.getLogger("vehicle")
logging.getLogger("vehicle").setLevel(logging.DEBUG)


class SloEstimator:
    def __init__(self, source_model, service_desc):
        self.source_model = source_model
        self.s_desc = service_desc
        self.model_VE = VariableElimination(self.source_model)

    def reload_source_model(self, source_model):
        self.source_model = source_model
        self.model_VE = VariableElimination(self.source_model)

    def infer_local_slo_f(self, target_running_services, device_name, origin_s_offload_desc=None, target_is_leader=False):
        remaining_services_o = target_running_services
        if origin_s_offload_desc is not None:
            remaining_services_o = [s for s in target_running_services if s != origin_s_offload_desc]
        # If there is no other service locally, fine, return 1.0
        if len(remaining_services_o) == 0:
            return [1.0]
        # If there is one other service locally, fine, return its isolated expectation
        # Otherwise return the conv of the remaining services
        else:
            base_service = remaining_services_o[0]
            dest_model = XMLBIFReader("models/" + utils.create_model_name(base_service['type'], device_name)).get_model()
            dest_model_VE = VariableElimination(dest_model)
            hw_load_p, slof_local_isolated = self.get_isolated_hw_predictions(model_VE=dest_model_VE, s_desc=base_service)
            prediction_conv = self.get_conv_hw_predictions(hw_load_p, dest_model, device_name, remaining_services_o[1:], target_is_leader)
            logger.debug(f"M| Predictions for SLO fulfillment at origin when conv with existing services {prediction_conv}")
            return prediction_conv

    # @utils.print_execution_time  # takes 400ms
    def infer_target_slo_f(self, target_model_name, target_running_services, prometheus_instance, target_is_leader):
        dest_model = XMLBIFReader("models/" + target_model_name).get_model()
        dest_model_VE = VariableElimination(dest_model)
        dest_device = utils.conv_ip_to_host_type(prometheus_instance)
        # Write: I might create a cube representation of the solution space
        hw_load_p, slof_local_isolated = self.get_isolated_hw_predictions(model_VE=dest_model_VE)

        # Write: The problem is that the load does not rise linear with more services
        # Idea: So what I can do is take the one that is worse to make a conservative prediction
        prediction_shifted = self.get_shifted_hw_predictions(hw_load_p, dest_model_VE, prometheus_instance, target_is_leader)
        logger.debug(f"M| Predictions for SLO fulfillment at target once load shifted {prediction_shifted}")

        # Write: I might need to split up the methods so that I can also evaluate their runtime more closely
        prediction_conv = self.get_conv_hw_predictions(hw_load_p, dest_model, dest_device, target_running_services, target_is_leader)
        logger.debug(f"M| Predictions for SLO fulfillment at origin when conv with existing services {prediction_conv}")

        return slof_local_isolated, prediction_shifted, prediction_conv

    # @utils.print_execution_time  # takes 120ms
    def calc_weighted_slo_f(self, p_dist_hw, dest_model_VE, isolated="False", shift=[0, 0, 0], is_leader=None):
        sum_slo_f = np.zeros((4, 4, 4))
        for i, _ in enumerate(p_dist_hw['cpu']):
            cpu_index = (i + shift[0]) if i + shift[0] <= utils.NUMBER_OF_BINS - 1 else utils.NUMBER_OF_BINS - 1
            for j, _ in enumerate(p_dist_hw['gpu']):
                gpu_index = (j + shift[1]) if j + shift[1] <= utils.NUMBER_OF_BINS - 1 else utils.NUMBER_OF_BINS - 1
                for k, _ in enumerate(p_dist_hw['memory']):
                    mem_index = (k + shift[2]) if k + shift[2] <= utils.NUMBER_OF_BINS - 1 else utils.NUMBER_OF_BINS - 1

                    p_cumm = p_dist_hw['cpu'][i] * p_dist_hw['gpu'][j] * p_dist_hw['memory'][k]

                    pass_is_leader = {'is_leader': f'{is_leader}'} if is_leader is not None else {}
                    slo_f_i = utils.get_true(
                        utils.infer_slo_fulfillment(dest_model_VE, self.s_desc['slo_vars'], self.s_desc['constraints'] |
                                                    {'cpu': f'{cpu_index}', 'gpu': f'{gpu_index}', 'memory': f'{mem_index}',
                                                     'isolated': isolated} | pass_is_leader))
                    weighted_p = p_cumm * slo_f_i
                    sum_slo_f[i, j, k] = weighted_p

        return np.sum(sum_slo_f)

    # @utils.print_execution_time  # takes 120ms
    def get_isolated_hw_predictions(self, model_VE=None, s_desc=None):
        if model_VE is None:
            model_VE = self.model_VE
        if s_desc is None:  # No values means take the origin description
            s_desc = self.s_desc
        hw_predictions = {}
        for var in ['cpu', 'gpu', 'memory']:
            hw_expectation_isolated = utils.infer_slo_fulfillment(model_VE, [var], s_desc['constraints'] | {'isolated': 'True'})
            hw_distribution = hw_expectation_isolated.values
            hw_predictions = hw_predictions | {var: hw_distribution}

        logger.debug(f"M| Predictions for isolated hardware consumption {hw_predictions}")
        slof_local_isolated = self.calc_weighted_slo_f(hw_predictions, dest_model_VE=model_VE, isolated="True")
        logger.debug(f"M| Expected SLO fulfillment for running {s_desc['type']} locally isolated {slof_local_isolated}")
        return hw_predictions, slof_local_isolated

    # @utils.print_execution_time  # takes 150ms
    def get_shifted_hw_predictions(self, origin_load_p, target_model_VE, target_host, target_is_leader):
        dest_current_load = model_trainer.get_latest_load(instance=target_host)
        dest_current_load_cat = utils.convert_prometheus_to_category(dest_current_load)
        logger.debug(f"M| Current load for target device classified into {dest_current_load_cat}")
        if target_host == "host.docker.internal" or target_host == "192.168.31.20":
            dest_current_load_cat[1] = -1
        # TODO: It might very well be that its isolated there, for this add a case like below
        return self.calc_weighted_slo_f(origin_load_p, dest_model_VE=target_model_VE, shift=(dest_current_load_cat + 1), isolated="False",
                                        is_leader=target_is_leader)

    # @utils.print_execution_time  # takes 100ms
    def get_conv_hw_predictions(self, origin_load_p, target_model_is, target_device, target_running_services, target_leader):
        if not target_running_services:
            return [self.calc_weighted_slo_f(origin_load_p, VariableElimination(target_model_is), isolated="True", is_leader=target_leader)]

        target_conv_load = origin_load_p
        target_models = [target_model_is]
        for s_desc in target_running_services:
            target_model = XMLBIFReader("models/" + utils.create_model_name(s_desc['type'], target_device)).get_model()
            target_models.append(target_model)
            s_load_p, _ = self.get_isolated_hw_predictions(VariableElimination(target_model), s_desc)

            for var in ['cpu', 'gpu', 'memory']:
                var_conv = utils.compress_into_n_bins(np.convolve(target_conv_load[var], s_load_p[var]))
                target_conv_load[var] = var_conv

        target_slo_f = []
        for model in target_models:
            target_slo_f.append(self.calc_weighted_slo_f(target_conv_load, VariableElimination(model), isolated="False",
                                                         is_leader=target_leader))

        return target_slo_f


if __name__ == "__main__":
    logging.getLogger("vehicle").setLevel(logging.DEBUG)

    local_model_name = utils.create_model_name("CV", "Laptop")
    local_model = XMLBIFReader("models/" + local_model_name).get_model()
    target_model_name = utils.create_model_name("CV", "Laptop")
    target_model = XMLBIFReader("models/" + local_model_name).get_model()

    s_desc_1 = {"id": 1, "type": 'CV', 'slo_vars': ["rate_60"], 'constraints': {'fps': '5', 'pixel': '480'}}
    s_desc_2 = {"id": 2, "type": 'CV', 'slo_vars': ["rate_60"], 'constraints': {'pixel': '720', 'fps': '5'}}
    s_desc_3 = {"id": 3, "type": 'CV', 'slo_vars': ["rate_60"], 'constraints': {'pixel': '1080', 'fps': '5'}}
    estimator = SloEstimator(local_model, service_desc=s_desc_1)
    estimator_2 = SloEstimator(local_model, service_desc=s_desc_2)
    estimator_3 = SloEstimator(local_model, service_desc=s_desc_3)

    # hw_load_p, slof_local_isolated = estimator.get_isolated_hw_predictions(model_VE=VariableElimination(local_model))
    # prediction_shifted = estimator.get_shifted_hw_predictions(hw_load_p, VariableElimination(target_model),
    #                                                           "host.docker.internal", True)
    # TODO: Stress device to violate SLOs and see reflected here
    # shifted = estimator.calc_weighted_slo_f(hw_load_p, dest_model_VE=VariableElimination(target_model), shift=[2, 0, 3], isolated="True",
    #                                         is_leader=True)
    # print(shifted)

    print(estimator.infer_local_slo_f([s_desc_1], "Laptop", target_is_leader=True))
    print(estimator_2.infer_local_slo_f([s_desc_2], "Laptop", target_is_leader=True))
    print(estimator_3.infer_local_slo_f([s_desc_3], "Laptop", target_is_leader=True))
