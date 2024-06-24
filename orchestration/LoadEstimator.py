import logging

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

import utils
from orchestration import model_trainer

logger = logging.getLogger("vehicle")
logging.getLogger("vehicle").setLevel(logging.DEBUG)


class LoadEstimator:
    def __init__(self, source_mode_name, service_desc):
        self.source_model = XMLBIFReader(source_mode_name).get_model()  # TODO: Must reload once received
        self.s_desc = service_desc
        self.model_VE = VariableElimination(self.source_model)

    def infer_target_slo_f(self, target_model_name):

        # Idea: Get a shifted list for all hw metrics and supply all as evidence
        for var in ['cpu']:
            expectation_isolated = utils.infer_slo_fulfillment(self.model_VE, [var], self.s_desc['constraints'] | {'isolated': 'True'})
            cpu_distribution = expectation_isolated.values
            prediction_isolated = self.calc_weighted_slo_f(cpu_distribution, isolated="True")
            logger.debug(f"M| Expected SLO fulfillment for running locally isolated {prediction_isolated}")

            # Write: The problem is that the load does not rise linear with more services
            # Idea: So what I can do is take the one that is worse to make a conservative prediction

            dest_model = XMLBIFReader(target_model_name).get_model()
            current_load = model_trainer.get_latest_load(metric_type='cpu', device_name='Laptop')[1]
            current_load_category = np.digitize([current_load], utils.split_into_bins(utils.NUMBER_OF_BINS))[0] - 1
            prediction_shifted = self.calc_weighted_slo_f(cpu_distribution, dest_model=dest_model, shift=(current_load_category + 1))

            P_conv = utils.compress_into_n_bins(np.convolve(cpu_distribution, cpu_distribution))
            prediction_conv = self.calc_weighted_slo_f(P_conv, dest_model=dest_model, isolated="False")

            return prediction_shifted, prediction_conv

    def calc_weighted_slo_f(self, p_dist, dest_model=None, isolated="False", shift=0):
        if dest_model is None:
            dest_model = self.source_model

        sum_slo_f = []
        for i, p in enumerate(p_dist):
            hw_index = (i + shift)

            # TODO: df conversion too slow, try to get this working for not df
            if not utils.verify_all_parameters_known(dest_model, pd.DataFrame([{'cpu': f'{hw_index}'}]), ['cpu']):
                hw_index = len(p_dist) - 1
            slo_f_i = utils.get_true(utils.infer_slo_fulfillment(self.model_VE, self.s_desc['slo_vars'], self.s_desc['constraints'] |
                                                                 {'cpu': f'{hw_index}', 'isolated': isolated}))
            weighted_p = p * slo_f_i
            logger.debug(f"{p} * {slo_f_i} = {weighted_p}")
            sum_slo_f.append(weighted_p)
        return sum(sum_slo_f)


laptop_mode_name = utils.create_model_name("CV", "Laptop")

s_description = {"name": 'CV', 'slo_vars': ["in_time"], 'constraints': {'pixel': '480', 'fps': '5'}}
estimator = LoadEstimator(laptop_mode_name, service_desc=s_description)

print(estimator.infer_target_slo_f(target_model_name=laptop_mode_name))
