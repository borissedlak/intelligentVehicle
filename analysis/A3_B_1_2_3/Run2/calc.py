from pgmpy.readwrite import XMLBIFReader

import utils
from orchestration.SloEstimator import SloEstimator

local_model_name = utils.create_model_name("CV", "Laptop")
local_model = XMLBIFReader("../../../orchestration/models/" + local_model_name).get_model()

cv_2 = {"id": 2, "type": 'CV', 'slo_vars': ["time"], 'constraints': {'fps': '10', 'pixel': '480'}}
cv_4 = {"id": 4, "type": 'CV', 'slo_vars': ["time"], 'constraints': {'fps': '10', 'pixel': '720'}}
li_3 = {"id": 3, "type": 'LI', 'slo_vars': ["time"], 'constraints': {'fps': '5', 'mode': 'single'}}
estimator = SloEstimator(local_model, service_desc=cv_4, prom_host="localhost")

qr_1 = {"id": 1, "type": 'QR', 'slo_vars': ["time"], 'constraints': {'fps': '5', 'pixel': '480'}}

# hw_load_p, slof_local_isolated = estimator.get_isolated_hw_predictions(model_VE=VariableElimination(local_model))
# prediction_shifted = estimator.get_shifted_hw_predictions(hw_load_p, VariableElimination(target_model),
#                                                           "host.docker.internal", True)

# shifted = estimator.calc_weighted_slo_f(hw_load_p, dest_model_VE=VariableElimination(target_model), shift=[2, 0, 3], isolated="True",
#                                         is_leader=True)
# print(shifted)

print(estimator.infer_local_slo_f([cv_2, cv_4, li_3], "Laptop", target_is_leader=True))
# print(estimator.infer_target_slo_f([s_desc_1], "Laptop", target_is_leader=True))
