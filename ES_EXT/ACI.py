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


class ACI:
    pixel_list = [480, 720, 1080]
    fps_list = [5, 10, 15, 20, 25]

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
        self.initial_data = util_fgcs.prepare_samples(pd.read_csv(f"ES_EXT/models/backup/backup_{self.s_desc['type']}_{DEVICE_NAME}.csv"),
                                                      conversion=False)
        self.past_data = None

        self.valid_stream_values_pv = []
        self.stream_regression_model_pv = LinearRegression()
        self.poly_features = PolynomialFeatures(degree=4)

        self.ig_matrix = np.full((3, 5), -1.0)
        self.pv_matrix = np.full((3, 5), -1.0)
        self.visit_matrix = np.full((3, 5), 0.05)
        self.visit_matrix[0][0], self.visit_matrix[2][4], self.visit_matrix[2][0], self.visit_matrix[0][4], \
            self.visit_matrix[1][2] = 1.0, 1.0, 1.0, 1.0, 1.0

    def iterate(self, samples, c_pixel, c_fps):
        current_batch = self.prepare_last_batch(samples)

        s = util_fgcs.get_surprise_for_data(self.model, current_batch, self.s_desc['slo_vars'])
        self.surprise_history.append(((c_pixel, c_fps), s))

        mean_surprise_last_10_values = np.median([t[1] for t in self.surprise_history][-10:])
        # self.backup_data = pd.concat([self.backup_data, current_batch], ignore_index=True)

        if s >= (1.5 * mean_surprise_last_10_values):
            # self.bnl(self.backup_data)
            self.retrain_parameter(current_batch)
        else:  # s >= (1 * mean_surprise_last_10_values):
            self.retrain_parameter(current_batch)

        pv = self.SLOs_fulfilled(current_batch)
        self.calculate_factors(c_pixel, c_fps)
        p_next, f_next, pv_est = self.get_best_configuration()

        return int(p_next), int(f_next), pv_est, (c_pixel, c_fps, pv), s

    # @print_execution_time # takes around 10-15ms
    def calculate_factors(self, c_pixel, c_fps):

        self.visit_matrix[ACI.pixel_list.index(c_pixel)][ACI.fps_list.index(c_fps)] = 0.0
        inference = VariableElimination(self.model)

        # Ensure that the current one is processed first to train the regression
        bitrate_list = list(itertools.product([str(i) for i in ACI.pixel_list], [str(i) for i in ACI.fps_list]))

        unknown_combinations = []

        for (pixel, fps) in bitrate_list:
            evidence = {'pixel': pixel, 'fps': fps}

            if self.visit_matrix[ACI.pixel_list.index(int(pixel))][ACI.fps_list.index(int(fps))] == 0:  # 0.0 indicates that was visited
                pv = util_fgcs.get_true(inference.query(variables=self.s_desc['slo_vars'], evidence=evidence))
                ig = (util_fgcs.get_median_surprise_one_config(self.surprise_history, (int(pixel), int(fps))) /
                      np.median(np.median([t[1] for t in self.surprise_history])))
                self.valid_stream_values_pv.append((int(pixel), int(fps), pv))
            else:
                pv, ig = -1.0, -1.0
                unknown_combinations.append((int(pixel), int(fps)))

            self.pv_matrix[ACI.pixel_list.index(int(pixel))][ACI.fps_list.index(int(fps))] = pv
            self.ig_matrix[ACI.pixel_list.index(int(pixel))][ACI.fps_list.index(int(fps))] = ig

    def get_best_configuration(self):
        pv_interpolated = util_fgcs.interpolate_values(self.pv_matrix)
        ig_interpolated = util_fgcs.interpolate_values(self.ig_matrix)

        max_sum = -float('inf')
        best_index = 0, 0
        for i in range(len(ACI.pixel_list)):
            for j in range(len(ACI.fps_list)):
                element_sum = (pv_interpolated[i, j] + min(ig_interpolated[i, j], 0.15) + self.visit_matrix[i, j])
                if element_sum > max_sum:
                    max_sum = element_sum
                    best_index = i, j

        p, f = best_index
        return ACI.pixel_list[p], ACI.fps_list[f], pv_interpolated[p, f]

    @print_execution_time
    def retrain_parameter(self, current_batch):
        if self.past_data is None:
            self.past_data = current_batch
        else:
            self.past_data = util_fgcs.prepare_samples(pd.concat([self.past_data, current_batch], ignore_index=True), conversion=False)

        # self.model.fit(self.initial_data)
        self.model.fit_update(self.past_data, n_prev_samples=1)
        # self.past_data_length += len(current_batch)

    def export_model(self, mode):
        self.initial_data.to_csv(f"ES_EXT/models/backup/backup_{self.s_desc['type']}_{DEVICE_NAME}.csv", index=False)
        np.savetxt(f"ES_EXT/results/pv/pv_{self.s_desc['type']}_{DEVICE_NAME}_{mode}.csv", self.pv_matrix, delimiter=',', fmt='%.3f')
        np.savetxt(f"ES_EXT/results/pv/ig_{self.s_desc['type']}_{DEVICE_NAME}_{mode}.csv", self.ig_matrix, delimiter=',', fmt='%.3f')

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
