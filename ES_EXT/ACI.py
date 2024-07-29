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
    bitrate_dict = {}

    for pair in itertools.product(pixel_list, fps_list):
        bitrate = pair[0] * pair[1]
        bitrate_dict.update({bitrate: [pair[0], pair[1]]})

    def __init__(self, description, show_img=False, load_model=None, distance_slo=40, network_slo=(420 * 30 * 10)):
        self.c_distance_bar = distance_slo
        self.c_network_bar = network_slo
        self.show_img = show_img
        if load_model:
            print("Loading pretained model")
            self.model = XMLBIFReader(load_model).get_model()
            util_fgcs.export_BN_to_graph(self.model, vis_ls=['circo'], save=True, name="raw_model", show=self.show_img)
            self.foster_bn_retrain = 0.2
            self.backup_data = pd.read_csv(f"models/backup/backup_CV_{DEVICE_NAME}.csv")
        else:
            self.model = None
            self.foster_bn_retrain = 0.5

        self.load_model = True if self.model is not None else False
        self.distance = 0
        self.current_batch = pd.DataFrame()
        self.entire_training_data = pd.DataFrame()
        self.past_training_data = pd.DataFrame()
        self.surprise_history = []
        self.function_time = []

        self.model_VE = VariableElimination(self.model)
        self.slo_hist = CyclicArray(75)
        self.s_desc = description

        self.valid_stream_values_pv = []
        self.stream_regression_model_pv = LinearRegression()
        self.poly_features = PolynomialFeatures(degree=4)

        self.pv_matrix = np.full((3, 5), -1.0)
        self.ig_matrix = np.full((3, 5), -1.0)
        self.ig_matrix[0][0], self.ig_matrix[2][4], self.ig_matrix[2][0], self.ig_matrix[0][4], \
            self.ig_matrix[1][2] = 0.0, 0.05, 0.05, 0.05, 0.05
        self.ig_matrix = util_fgcs.interpolate_values(self.ig_matrix)
        self.ig_matrix[0][0], self.ig_matrix[2][4], self.ig_matrix[2][0], self.ig_matrix[0][4], \
            self.ig_matrix[1][2] = 0.0, 0.3, 0.3, 0.3, 0.3

    def iterate(self, samples, c_stream_count=1):
        self.load_last_batch(samples)
        # self.past_training_data = self.entire_training_data.copy()
        c_pixel = int(self.current_batch.iloc[0]['pixel'])
        c_fps = int(self.current_batch.iloc[0]['fps'])

        # TODO: Must be passed from service description
        s = util_fgcs.get_surprise_for_data(self.model, self.current_batch, self.s_desc['slo_vars'])
        self.surprise_history.append(s)

        mean_surprise_last_10_values = np.median(self.surprise_history[-10:])
        # if s > ((2 - self.foster_bn_retrain) * mean_surprise_last_10_values):
        #     if self.foster_bn_retrain == 0.5:
        #         self.foster_bn_retrain = 0.2
        #     elif self.foster_bn_retrain == 0.2:
        #         self.foster_bn_retrain = 0.0
        #
        #     # self.bnl(self.entire_training_data)
        #     self.retrain_parameter()
        if s >= (1 * mean_surprise_last_10_values):
            self.retrain_parameter()

        pv = self.SLOs_fulfilled(self.current_batch)
        self.calculate_factors(c_pixel, c_fps)
        p_next, f_next, pv_est = self.get_best_configuration()

        return int(p_next), int(f_next), pv_est, (c_pixel, c_fps, pv), s

    def get_best_configuration(self):
        pv_interpolated = util_fgcs.interpolate_values(self.pv_matrix)
        # ra_interpolated = util_fgcs.interpolate_values(self.ra_matrix)
        ig_interpolated = util_fgcs.interpolate_values(self.ig_matrix)

        max_sum = -float('inf')
        best_index = 0, 0
        for i in range(len(ACI.pixel_list)):
            for j in range(len(ACI.fps_list)):
                element_sum = (pv_interpolated[i, j] + ig_interpolated[i, j])
                if element_sum > max_sum:
                    max_sum = element_sum
                    best_index = i, j

        p, f = best_index
        return ACI.pixel_list[p], ACI.fps_list[f], pv_interpolated[p, f]

    # @print_execution_time # takes around 10-15ms
    def calculate_factors(self, c_pixel, c_fps):

        self.ig_matrix[ACI.pixel_list.index(c_pixel)][ACI.fps_list.index(c_fps)] = 0.0
        inference = VariableElimination(self.model)  # util_fgcs.get_mbs_as_bn(self.model, self.s_desc['slo_vars']))

        # Ensure that the current one is processed first to train the regression
        bitrate_list = list(itertools.product([str(i) for i in ACI.pixel_list], [str(i) for i in ACI.fps_list]))
        bitrate_list.remove((str(c_pixel), str(c_fps)))
        bitrate_list.insert(0, (str(c_pixel), str(c_fps)))

        unknown_combinations = []

        for (pixel, fps) in bitrate_list:

            # try:
            evidence = {'pixel': pixel, 'fps': fps}
            pv = util_fgcs.get_true(inference.query(variables=self.s_desc['slo_vars'], evidence=evidence))

            if 0.2 < pv < 0.3:  # Default value indicating empty
                unknown_combinations.append((int(pixel), int(fps)))
            else:
                self.valid_stream_values_pv.append((int(pixel), int(fps), pv))
                self.pv_matrix[ACI.pixel_list.index(int(pixel))][ACI.fps_list.index(int(fps))] = pv
            # except Exception as ex:
            #     unknown_combinations.append((pixel, fps))
            #     if str(ex) != "_nan":
            #         print(ex)

        if len(unknown_combinations) > 0:
            input_data = np.array([(x1, x2) for (x1, x2, y) in self.valid_stream_values_pv])
            input_data = self.poly_features.fit_transform(input_data)
            target_data = np.array([y for (x1, x2, y) in self.valid_stream_values_pv])
            self.stream_regression_model_pv.fit(input_data, target_data)

            for p, f in unknown_combinations:
                input_vector = self.poly_features.fit_transform(np.array([[p, f]]))
                pv_predict = util_fgcs.cap_0_1(self.stream_regression_model_pv.predict(input_vector)[0])
                self.pv_matrix[ACI.pixel_list.index(p)][ACI.fps_list.index(f)] = pv_predict

    @print_execution_time
    def retrain_parameter(self, full_retrain=False):
        if full_retrain:
            raise RuntimeError("Should not happen for retrain params")
            # if self.load_model:
            #     util_fgcs.print_in_red("Should not happen when loading model")
            #     self.entire_training_data = pd.concat([self.entire_training_data, self.backup_data], ignore_index=True)
            #     self.load_model = False
            # self.model.fit(self.entire_training_data)
        else:
            # past_data_length = len(self.past_training_data)
            # if hasattr(self, 'backup_data'):
            #     past_data_length += len(self.backup_data)
            self.model.fit_update(self.current_batch) #, n_prev_samples=(past_data_length / 3))
            # except ValueError as ve:
            #     print(f"Caught a ValueError: {ve}")
            #     self.retrain_parameter(full_retrain=True)

    # def initialize_bn(self):
    #     self.bnl(self.entire_training_data)

    def export_model(self):
        # Define the output file path
        # output_path = f"backup_entire_data_{DEVICE_NAME}.csv"
        #
        # # TODO: Remove the header from the second file
        # with open(f"backup_entire_data_{DEVICE_NAME}.csv", 'r') as file1, open("../data/Performance_History.csv", 'r') as file2:
        #     file1_contents = file1.read()
        #     file2_contents = file2.read()
        #
        # # Concatenate the contents
        # concatenated_contents = file1_contents + file2_contents
        #
        # # Write the concatenated content to the output file
        # with open(output_path, 'w') as output_file:
        #     output_file.write(concatenated_contents)

        writer = XMLBIFWriter(self.model)
        file_name = utils.create_model_name("CV", DEVICE_NAME)
        writer.write_xmlbif(filename=file_name)
        print(f"Model exported as '{file_name}'")

    @print_execution_time
    def bnl(self, samples):

        scoring_method = AICScore(data=samples)  # BDeuScore | AICScore
        estimator = HillClimbSearch(data=samples)

        dag: pgmpy.base.DAG = estimator.estimate(
            scoring_method=scoring_method, max_indegree=4, epsilon=1,
        )

        # if dag.has_edge("bitrate", "distance"):
        #     dag.remove_edge("bitrate", "distance")
        # dag.add_edge("fps", "distance")
        util_fgcs.export_BN_to_graph(dag, vis_ls=['circo'], save=True, name="raw_model", show=self.show_img)

        self.model = BayesianNetwork(ebunch=dag)
        self.model.fit(data=samples, estimator=MaximumLikelihoodEstimator)

    def load_last_batch(self, samples):
        samples = pd.DataFrame(samples.get())
        samples = util_fgcs.prepare_samples(samples)
        del samples['service']
        del samples['device_type']
        self.current_batch = samples

    def SLOs_fulfilled(self, batch: pd.DataFrame):
        batch['in_time'] = batch['in_time'].map({'True': True, 'False': False})
        batch['energy_saved'] = batch['energy_saved'].map({'True': True, 'False': False})
        # batch['success'] = batch['success'].map({'True': True, 'False': False})

        ratio_in_time = batch[batch["in_time"]].size / batch.size
        ratio_energy = batch[batch["energy_saved"]].size / batch.size
        # ratio_success = batch[batch["success"]].size / batch.size

        # TODO: Missing rate SLO
        pv = ratio_energy * ratio_in_time * 1.0
        self.slo_hist.append(pv)
        rebalanced_slo_f = self.slo_hist.average()

        return rebalanced_slo_f
