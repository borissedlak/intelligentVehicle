import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFReader, XMLBIFWriter

import util_fgcs
from util_fgcs import prepare_samples, print_execution_time

br = str(30 * 420)
stream = '1'

model_Xavier_CPU = XMLBIFReader("model_Xavier_CPU.xml").get_model()
raw_data_Xavier_CPU = pd.read_csv("backup_entire_data_Xavier_CPU.csv")
curated_data_Xavier_CPU = prepare_samples(raw_data_Xavier_CPU, t_distance=50, t_total_bitrate=(420 * 30 * 10))
curated_data_Xavier_CPU['consumption'] = '22'

inference = VariableElimination(model_Xavier_CPU)
ra = util_fgcs.get_true(
    inference.query(variables=["in_time", "network"], evidence={'bitrate': br, 'stream_count': stream}))
print(ra)

model_Laptop = XMLBIFReader("model_Laptop.xml").get_model()
raw_data_Laptop = pd.read_csv("backup_entire_data_Laptop.csv")
curated_data_Laptop = prepare_samples(raw_data_Laptop, t_distance=50, t_total_bitrate=(420 * 30 * 10))
curated_data_Laptop['consumption'] = '7'

inference = VariableElimination(model_Laptop)
ra = util_fgcs.get_true(
    inference.query(variables=["in_time", "network"], evidence={'bitrate': br, 'stream_count': stream}))
print(ra)

# filtered_data = curated_data_Xavier_CPU[
#     ~curated_data_Xavier_CPU['bitrate'].isin(['3120', '4680', '1680', '2160', '4320'])]


filtered_data = curated_data_Laptop[
    ~curated_data_Laptop['stream_count'].isin(['11', '12', '13', '14', '15'])]


def merge_cpts():
    merged_model = BayesianNetwork(ebunch=model_Laptop.edges())

    # Iterate through the CPTs of the first model (model1)
    for node in ['cpu_utilization']:
        cpd1 = model_Laptop.get_cpds(node)

        # Check if the node exists in the second model (model2)
        if node in model_Xavier_CPU.nodes():
            cpd2 = model_Xavier_CPU.get_cpds(node)

            # Merge the two CPTs
            merged_cpd = cpd1.product(cpd2)
            merged_cpd.normalize()

            # Add the merged CPT to the merged_model
            merged_model.add_cpds(merged_cpd)
        else:
            # If the node does not exist in the second model, add the CPT from the first model to the merged_model
            merged_model.add_cpds(cpd1)

    # Iterate through the CPTs of the second model (model2) and add any CPTs not already added
    for node in model_Xavier_CPU.nodes():
        if node not in merged_model.nodes():
            cpd2 = model_Xavier_CPU.get_cpds(node)
            merged_model.add_cpds(cpd2)

    return merged_model


# merged_model = merge_cpts()


@print_execution_time
def merge():
    model_Xavier_CPU.fit_update(filtered_data, n_prev_samples=len(raw_data_Xavier_CPU))


merge()
inference = VariableElimination(model_Xavier_CPU)
ra = util_fgcs.get_true(
    inference.query(variables=["in_time", "network"], evidence={'bitrate': br, 'stream_count': stream}))
print(ra)

model_name = 'model_Xavier_GPU.xml'
XMLBIFWriter(model_Laptop).write_xmlbif(model_name)
print(f"Model exported as '{model_name}'")

model_Laptop.fit(curated_data_Xavier_CPU)
model_name = 'model_Xavier_CPU.xml'
XMLBIFWriter(model_Laptop).write_xmlbif(model_name)
print(f"Model exported as '{model_name}'")

model_Laptop.fit(curated_data_Laptop)
model_name = 'model_Laptop.xml'
XMLBIFWriter(model_Laptop).write_xmlbif(model_name)
print(f"Model exported as '{model_name}'")

raw_data_Nano = pd.read_csv("backup_entire_data_Xavier_CPU.csv")
curated_data_Nano = prepare_samples(raw_data_Nano, t_distance=50, t_total_bitrate=(420 * 30 * 10))
model_Laptop.fit(curated_data_Nano)
model_name = 'model_Nano.xml'
XMLBIFWriter(model_Laptop).write_xmlbif(model_name)
print(f"Model exported as '{model_name}'")

raw_data_Orin = pd.read_csv("backup_entire_data_Orin.csv")
curated_data_Orin = prepare_samples(raw_data_Orin, t_distance=50, t_total_bitrate=(420 * 30 * 10))
model_Laptop.fit(curated_data_Orin)
model_name = 'model_Orin.xml'
XMLBIFWriter(model_Laptop).write_xmlbif(model_name)
print(f"Model exported as '{model_name}'")
