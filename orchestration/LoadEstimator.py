import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

import utils

laptop_mode_name = utils.create_model_name("CV", "Laptop")
model = XMLBIFReader(laptop_mode_name).get_model()
model_VE = VariableElimination(model)

s_description = {"name": 'CV', 'slo_var': ["in_time"], 'constraints': {'pixel': '480', 'fps': '5'}}
expectation = utils.infer_slo_fulfillment(model_VE, ['cpu'], s_description['constraints'])

cpu_distribution = expectation.values
print(cpu_distribution)

# Binning the load into 4 discrete categories
# P_A = np.array([0.1, 0.2, 0.4, 0.3])  #
# P_B = np.array([0.3, 0.4, 0.2, 0.1])  #

# Compute the convolution of the two distributions
P_A_plus_B = np.convolve(cpu_distribution, cpu_distribution)

# Result of convolution (example output)
print(P_A_plus_B)

# Bin the 7-element array down to 4 elements
num_bins = 4
binned_P = np.zeros(num_bins)

# Determine the size of each bin
bin_size = len(P_A_plus_B) / num_bins

for i in range(num_bins):
    start_index = int(i * bin_size)
    end_index = int((i + 1) * bin_size)
    binned_P[i] = np.sum(P_A_plus_B[start_index:end_index])

print("Binned Probability Distribution:")
print(binned_P)
