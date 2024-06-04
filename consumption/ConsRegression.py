import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from detector.utils import print_execution_time

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = os.path.dirname(__file__)


class ConsRegression:

    def __init__(self, device_type):
        self.device_type = device_type
        self.export_model_file_name = ROOT + f'/model_{self.device_type}.sav'
        self.source_data_file_name = ROOT + f'/data_{self.device_type}.csv'

        self.load_device_model()

    def load_device_model(self):
        file_exists = os.path.isfile(self.export_model_file_name)
        if file_exists:
            loaded_model = joblib.load(self.export_model_file_name)
            self.model = loaded_model
        else:
            trained_model = self.train_model()
            self.model = trained_model

    @print_execution_time
    def train_model(self):
        model = LinearRegression()
        df = pd.read_csv(self.source_data_file_name)
        df = df.dropna()
        df['GPU'] = df['GPU'].astype(int)
        X_train = df[['cpu_utilization', 'GPU']]
        # X_train.columns = None
        y_train = df['consumption']
        model.fit(X_train, y_train)
        joblib.dump(model, self.export_model_file_name)
        return model

    def predict(self, cpu_utilization, GPU):
        return int(round(self.model.predict(np.array([[cpu_utilization, GPU]]))[0], 0))
