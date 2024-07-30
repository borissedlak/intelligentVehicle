import itertools
import os
import warnings

import numpy as np
import pandas as pd
import pgmpy
import pgmpy.base.DAG
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, AICScore
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFReader, XMLBIFWriter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import util_fgcs
import utils
from monitor.DeviceMetricReporter import CyclicArray
from util_fgcs import print_execution_time

ROOT = os.path.dirname(__file__)

warnings.filterwarnings("ignore", category=RuntimeWarning)
DEVICE_NAME = utils.get_ENV_PARAM('DEVICE_NAME', "Unknown")


class ACI_LI:
    pixel_list = [480, 720, 1080]
    fps_list = [5, 10, 15, 20, 25]
    mode_list = ['single', 'double']

    def __init__(self, description, show_img=False, load_model=None):

        self.show_img = show_img
        if load_model:
            print("Loading pretained model")
            self.model = XMLBIFReader(load_model).get_model()
            util_fgcs.export_BN_to_graph(self.model, vis_ls=['circo'], save=True, name="raw_model", show=self.show_img)
        else:
            self.model = None

        self.load_model = True if self.model is not None else False
        self.distance = 0
        self.surprise_history = []
        self.function_time = []

        self.model_VE = VariableElimination(self.model)
        self.slo_hist = CyclicArray(75)
        self.s_desc = description
        self.backup_data = util_fgcs.prepare_samples(pd.read_csv(f"ES_EXT/models/backup/backup_{self.s_desc['type']}_{DEVICE_NAME}.csv"),
                                                     conversion=False)

        self.valid_stream_values_pv = []
        self.stream_regression_model_pv = LinearRegression()
        self.poly_features = PolynomialFeatures(degree=4)

        self.pv_matrix = np.full((2, 5), -1.0)
        self.ig_matrix = np.full((3, 5), 0.05)
        self.ig_matrix[0][0], self.ig_matrix[1][4], self.ig_matrix[1][0], self.ig_matrix[0][4], \
            self.ig_matrix[1][2] = 1.0, 1.0, 1.0, 1.0, 1.0

    def iterate(self, samples):
        current_batch = self.prepare_last_batch(samples)
        c_mode = current_batch.iloc[0]['mode']
        c_fps = int(current_batch.iloc[0]['fps'])

        s = util_fgcs.get_surprise_for_data(self.model, current_batch, self.s_desc['slo_vars'])
        self.surprise_history.append(s)

        mean_surprise_last_10_values = np.median(self.surprise_history[-10:])
        self.backup_data = pd.concat([self.backup_data, current_batch], ignore_index=True)

        if s >= (1.5 * mean_surprise_last_10_values):
            # self.bnl(self.backup_data)
            self.retrain_parameter(current_batch)
        if s >= (1 * mean_surprise_last_10_values):
            self.retrain_parameter(current_batch)

        pv = self.SLOs_fulfilled(current_batch)
        self.calculate_factors(c_mode, c_fps)
        m_next, f_next, pv_est = self.get_best_configuration()

        return m_next, int(f_next), pv_est, (c_mode, c_fps, pv), s

    # @print_execution_time # takes around 10-15ms
    def calculate_factors(self, c_mode, c_fps):

        self.ig_matrix[ACI_LI.mode_list.index(c_mode)][ACI_LI.fps_list.index(c_fps)] = 0.0
        inference = VariableElimination(self.model)

        # Ensure that the current one is processed first to train the regression
        bitrate_list = list(itertools.product([str(i) for i in ACI_LI.mode_list], [str(i) for i in ACI_LI.fps_list]))

        unknown_combinations = []

        for (mode, fps) in bitrate_list:
            evidence = {'mode': mode, 'fps': fps}

            if self.ig_matrix[ACI_LI.mode_list.index(mode)][ACI_LI.fps_list.index(int(fps))] == 0:  # 0.0 indicates that was visited
                pv = util_fgcs.get_true(inference.query(variables=self.s_desc['slo_vars'], evidence=evidence))
                self.valid_stream_values_pv.append((mode, int(fps), pv))
            else:
                pv = -1.0
                unknown_combinations.append((mode, int(fps)))

            self.pv_matrix[ACI_LI.mode_list.index(mode)][ACI_LI.fps_list.index(int(fps))] = pv

    def get_best_configuration(self):
        pv_interpolated = util_fgcs.interpolate_values(self.pv_matrix)
        ig_interpolated = util_fgcs.interpolate_values(self.ig_matrix)

        max_sum = -float('inf')
        best_index = 0, 0
        for i in range(len(ACI_LI.mode_list)):
            for j in range(len(ACI_LI.fps_list)):
                element_sum = (pv_interpolated[i, j] + ig_interpolated[i, j])
                if element_sum > max_sum:
                    max_sum = element_sum
                    best_index = i, j

        p, f = best_index
        return ACI_LI.mode_list[p], ACI_LI.fps_list[f], pv_interpolated[p, f]

    @print_execution_time
    def retrain_parameter(self, current_batch):
        # past_data_length = len(self.past_training_data)
        # if hasattr(self, 'backup_data'):
        #     past_data_length += len(self.backup_data)
        self.model.fit_update(current_batch)  # , n_prev_samples=(past_data_length / 3))

    def export_model(self):
        self.backup_data.to_csv(f"ES_EXT/models/backup/backup_{self.s_desc['type']}_{DEVICE_NAME}.csv", index=False)
        np.savetxt(f"ES_EXT/results/pv/pv_{self.s_desc['type']}_{DEVICE_NAME}.csv", self.pv_matrix, delimiter=',', fmt='%.3f')

        writer = XMLBIFWriter(self.model)
        file_name = utils.create_model_name("CV", DEVICE_NAME)
        writer.write_xmlbif(filename="ES_EXT/models/" + file_name)
        print(f"Model exported as '{file_name}'")

    @print_execution_time
    def bnl(self, samples):

        scoring_method = AICScore(data=samples)  # BDeuScore | AICScore
        estimator = HillClimbSearch(data=samples)

        dag: pgmpy.base.DAG = estimator.estimate(
            scoring_method=scoring_method, max_indegree=4, epsilon=1,
        )

        util_fgcs.export_BN_to_graph(dag, vis_ls=['circo'], save=True, name="raw_model", show=self.show_img)

        self.model = BayesianNetwork(ebunch=dag)
        self.model.fit(data=samples, estimator=MaximumLikelihoodEstimator)

    def prepare_last_batch(self, buffer):
        samples = pd.DataFrame(buffer)
        samples = util_fgcs.prepare_samples(samples)
        del samples['service']
        del samples['device_type']
        return samples

    def SLOs_fulfilled(self, batch: pd.DataFrame):
        pv = 1.0
        for var in self.s_desc['slo_vars']:
            batch[var] = batch[var].map({'True': True, 'False': False})
            ratio = batch[batch[var]].size / batch.size
            pv *= ratio

        self.slo_hist.append(pv)
        rebalanced_slo_f = self.slo_hist.average()

        return rebalanced_slo_f