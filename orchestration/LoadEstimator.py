import numpy as np

# Example probability distributions
P_A = np.array([0.1, 0.2, 0.4, 0.3])  # Probabilities for workload A
P_B = np.array([0.3, 0.4, 0.2, 0.1])  # Probabilities for workload B

# Compute the convolution of the two distributions
P_A_plus_B = np.convolve(P_A, P_B)

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