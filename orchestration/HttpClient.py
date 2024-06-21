import pandas as pd
import requests


class HttpClient:
    def __init__(self, DEFAULT_HOST='localhost'):
        self.HOST = DEFAULT_HOST
        self.PORT = 8080
        self.SESSION = requests.Session()
        self.http_connection = None
        self.START_SERVICE_PATH = "/start_service"
        self.MODEL_UPLOAD_PATH = "/model/upload"
        self.MODEL_UPDATE_PATH = "/model/update"

        print(f"Opening HTTP Connection with {self.HOST} and {self.PORT}")

    def send_system_stats(self, cpu, device_name, disabled_aci, gpu_available):
        query_params = {
            "cpu": cpu,
            "device_name": device_name,
            "disabled_aci": disabled_aci,
            "gpu_available": gpu_available
        }
        response = self.SESSION.get(f"http://{self.HOST}:{self.PORT}{self.START_SERVICE_PATH}", params=query_params)
        response.raise_for_status()  # Raise an exception for non-2xx status codes

    def push_files_to_member(self, model_names, target_route=None):
        if target_route is None:
            target_route = self.HOST
        files = []
        for index, m in enumerate(model_names):
            files.append((f'file{index + 1}', (m, open(m, 'rb'), 'application/xml')))

        # headers = {'Content-Type': 'application/xml'}  # Set the content type
        url = f"http://{target_route}:{self.PORT}{self.MODEL_UPLOAD_PATH}"
        response = self.SESSION.post(url, files=files)
        response.raise_for_status()  # Raise an exception for non-2xx status codes

    def push_metrics_retrain(self, service_name, df: pd.DataFrame):
        csv_string = df.to_csv(index=False)

        headers = {'Content-Type': 'text/csv'}
        url = f"http://{self.HOST}:{self.PORT}{self.MODEL_UPDATE_PATH}/{service_name}"
        response = self.SESSION.post(url, data=csv_string, headers=headers)
        response.raise_for_status()  # Raise an exception for non-2xx status codes
