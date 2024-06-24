import logging

import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

import utils

logger = logging.getLogger("vehicle")
logging.getLogger("vehicle").setLevel(logging.DEBUG)

laptop_mode_name = utils.create_model_name("CV", "Laptop")
model = XMLBIFReader(laptop_mode_name).get_model()
model_VE = VariableElimination(model)

s_description = {"name": 'CV', 'slo_vars': ["in_time"], 'constraints': {'pixel': '480', 'fps': '5'}}
expectation = utils.infer_slo_fulfillment(model_VE, ['cpu'], s_description['constraints'] | {'isolated': 'True'})

cpu_distribution = expectation.values
print(cpu_distribution)
P_conv = np.convolve(cpu_distribution, cpu_distribution)#, cpu_distribution)
# print(P_conv)

# Bin the x-element array down to 4 elements
num_bins = utils.NUMBER_OF_BINS
binned_P = np.zeros(num_bins)
bin_size = len(P_conv) / num_bins

for i in range(num_bins):
    start_index = int(i * bin_size)
    end_index = int((i + 1) * bin_size)
    binned_P[i] = np.sum(P_conv[start_index:end_index])

# TODO: I have the feeling that this does not shift the distribution enough, but calculates some thing else
print("Binned convolutional distribution:")
print(binned_P)


def calculate_weighted_slo_f(p_dist, isolated="False"):
    sum_slo_f = []
    for i, p in enumerate(p_dist):
        slo_f_i = utils.get_true(utils.infer_slo_fulfillment(model_VE, s_description['slo_vars'],
                                                             s_description['constraints'] | {'cpu': f'{i}', 'isolated': isolated}))
        weighted_p = p * slo_f_i
        logger.debug(f"{p} * {slo_f_i} = {weighted_p}")
        sum_slo_f.append(weighted_p)
    return sum(sum_slo_f)


print(calculate_weighted_slo_f(cpu_distribution, isolated="True"))
print(calculate_weighted_slo_f(binned_P, isolated="False"))

# TODO: I must finally pin the DAG I think
# print(utils.get_true(utils.infer_slo_fulfillment(model_VE, s_description['slo_vars'], s_description['constraints'])))
# print(utils.get_true(utils.infer_slo_fulfillment(model_VE, s_description['slo_vars'], s_description['constraints'] | {'cpu': '0'})))
# print(utils.get_true(utils.infer_slo_fulfillment(model_VE, s_description['slo_vars'], s_description['constraints'] | {'cpu': '1'})))
# print(utils.get_true(utils.infer_slo_fulfillment(model_VE, s_description['slo_vars'], s_description['constraints'] | {'cpu': '2'})))
# print(utils.get_true(utils.infer_slo_fulfillment(model_VE, s_description['slo_vars'], s_description['constraints'] | {'cpu': '3'})))
# print(utils.get_true(
#     utils.infer_slo_fulfillment(model_VE, s_description['slo_vars'], s_description['constraints'] | {'cpu': '3', 'isolated': 'True'})))
# print(utils.get_true(
#     utils.infer_slo_fulfillment(model_VE, s_description['slo_vars'], s_description['constraints'] | {'cpu': '3', 'isolated': 'False'})))
