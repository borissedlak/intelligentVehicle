import logging

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

import utils
from orchestration import model_trainer

logger = logging.getLogger("vehicle")
logging.getLogger("vehicle").setLevel(logging.DEBUG)

laptop_mode_name = utils.create_model_name("CV", "Laptop")
model = XMLBIFReader(laptop_mode_name).get_model()
model_VE = VariableElimination(model)

s_description = {"name": 'CV', 'slo_vars': ["in_time"], 'constraints': {'pixel': '480', 'fps': '5'}}
expectation = utils.infer_slo_fulfillment(model_VE, ['cpu'], s_description['constraints'] | {'isolated': 'True'})

cpu_distribution = expectation.values
print(cpu_distribution)
P_conv = np.convolve(cpu_distribution, cpu_distribution)  # , cpu_distribution)
current_load = model_trainer.get_latest_load(metric_type='cpu', device_name='Laptop')[1]
current_load_category = np.digitize([current_load], utils.split_into_bins(utils.NUMBER_OF_BINS))[0] - 1

# sys.exit()

# Bin the x-element array down to 4 elements
num_bins = utils.NUMBER_OF_BINS
binned_P = np.zeros(num_bins)
bin_size = len(P_conv) / num_bins

for i in range(num_bins):
    start_index = int(i * bin_size)
    end_index = int((i + 1) * bin_size)
    binned_P[i] = np.sum(P_conv[start_index:end_index])

print("Binned convolutional distribution:")
print(binned_P)


def calculate_weighted_slo_f(p_dist, isolated="False", shift=0):
    sum_slo_f = []
    for i, p in enumerate(p_dist):
        hw_index = (i + shift)

        # TODO: df conversion too slow, try to get this working for not df
        if not utils.verify_all_parameters_known(model, pd.DataFrame([{'cpu': f'{hw_index}'}]), ['cpu']):
            hw_index = len(p_dist) - 1
        slo_f_i = utils.get_true(utils.infer_slo_fulfillment(model_VE, s_description['slo_vars'],
                                                             s_description['constraints'] | {'cpu': f'{hw_index}', 'isolated': isolated}))
        weighted_p = p * slo_f_i
        logger.debug(f"{p} * {slo_f_i} = {weighted_p}")
        sum_slo_f.append(weighted_p)
    return sum(sum_slo_f)


# Write: The problem is that the load does not rise linear with more services
# Idea: So what I can do is take the one that is worse to make a conservative prediction

print(calculate_weighted_slo_f(cpu_distribution, isolated="True"))
print(calculate_weighted_slo_f(cpu_distribution, shift=(current_load_category + 1)))
print(calculate_weighted_slo_f(binned_P, isolated="False"))
