import pulp

# Define the number of items and knapsacks
n_items = 6
n_knapsacks = 3

# Define the values of the items for each knapsack
# This is definitely related to the amount of resources consumer from the overall capacity
slo_values = [
    [100, 100, 100],  # Values of item 1 in knapsacks 1, 2, 3
    [50, 90, 100],  # Values of item 2 in knapsacks 1, 2, 3
    [50, 90, 100],  # Values of item 2 in knapsacks 1, 2, 3
    [50, 90, 100],  # Values of item 2 in knapsacks 1, 2, 3
    [20, 30, 100],  # Values of item 3 in knapsacks 1, 2, 3
    [100, 100, 100],  # Values of item 4 in knapsacks 1, 2, 3
]

# Define the weights and volumes of the items
cpu_demand = [80, 50, 50, 50, 200, 50]
gpu_demand = [10, 150, 150, 150, 100, 10]

# Define the weight and volume capacities of the knapsacks
cpu_capacities = [100, 100, 300]
gpu_capacities = [100, 200, 200]

# Define the problem
prob = pulp.LpProblem("MultipleKnapsackProblem", pulp.LpMaximize)

# Define the decision variables
x = pulp.LpVariable.dicts("x", ((i, j) for i in range(n_items) for j in range(n_knapsacks)), cat='Binary')

# Objective function
prob += pulp.lpSum(slo_values[i][j] * x[(i, j)] for i in range(n_items) for j in range(n_knapsacks))

# Constraints
# Each item can be in at most one knapsack
for i in range(n_items):
    prob += pulp.lpSum(x[(i, j)] for j in range(n_knapsacks)) <= 1

# Knapsack weight capacity constraints
for j in range(n_knapsacks):
    prob += pulp.lpSum(cpu_demand[i] * x[(i, j)] for i in range(n_items)) <= cpu_capacities[j]

# Knapsack volume capacity constraints
for j in range(n_knapsacks):
    prob += pulp.lpSum(gpu_demand[i] * x[(i, j)] for i in range(n_items)) <= gpu_capacities[j]

# Solve the problem
prob.solve()

# Print the results
print("Status:", pulp.LpStatus[prob.status])

total_value = 0
for j in range(n_knapsacks):
    knapsack_value = 0
    knapsack_weight = 0
    knapsack_volume = 0
    print(f"Knapsack {j + 1}:")
    for i in range(n_items):
        if x[(i, j)].varValue == 1:
            print(f" - Item {i + 1}: value = {slo_values[i][j]}, weight = {cpu_demand[i]}, volume = {gpu_demand[i]}")
            knapsack_value += slo_values[i][j]
            knapsack_weight += cpu_demand[i]
            knapsack_volume += gpu_demand[i]
    total_value += knapsack_value
    print(f" Total value: {knapsack_value}, Total weight: {knapsack_weight}, Total volume: {knapsack_volume}")

print(f"Total value of packed items: {total_value}")
