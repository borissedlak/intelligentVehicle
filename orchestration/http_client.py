import requests


class HttpClient:
    def __init__(self, HOST='localhost'):
        self.HOST = HOST
        self.PORT = 8080
        self.SESSION = requests.Session()
        self.http_connection = None
        self.START_SERVICE_PATH = "/start_service"
        self.MODEL_UPLOAD_PATH = "/model/upload"

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

    def push_files_to_member(self):
        files = [('file1', ('CV_Laptop_model.xml', open('CV_Laptop_model.xml', 'rb'), 'application/xml')),
                 ('file2', ('CV_Laptop_model.xml', open('CV_Laptop_model.xml', 'rb'), 'application/xml'))]

        # headers = {'Content-Type': 'application/xml'}  # Set the content type
        url = f"http://{self.HOST}:{self.PORT}{self.MODEL_UPLOAD_PATH}"
        response = self.SESSION.post(url, files=files)
        response.raise_for_status()  # Raise an exception for non-2xx status codes
