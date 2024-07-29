import http.client
import threading

import requests


class HttpClient:
    def __init__(self, HOST='localhost'):
        self.HOST = HOST
        self.PORT = 8080
        self.SESSION = requests.Session()
        self.http_connection = None
        self.SYSTEM_STATS_PATH = "/system"
        self.APP_STATS_PATH = "/stats"
        self.latest_config = 1
        self.latest_latency = 1
        self.ignore_response = False

        print(f"Opening HTTP Connection with {self.HOST} and {self.PORT}")

    def send_system_stats(self, cpu, device_name, disabled_aci, gpu_available):
        query_params = {
            "cpu": cpu,
            "device_name": device_name,
            "disabled_aci": disabled_aci,
            "gpu_available": gpu_available
        }
        response = self.SESSION.get(f"http://{self.HOST}:{self.PORT}{self.SYSTEM_STATS_PATH}", params=query_params)
        response.raise_for_status()  # Raise an exception for non-2xx status codes

    def send_app_stats(self, pixel, fps, pv, ra, threads, device_name, gpu, surprise):
        query_params = {
            "pixel": pixel,
            "fps": fps,
            "pv": pv,
            "ra": ra,
            "threads": threads,
            "device_name": device_name,
            "gpu": gpu,
            "surprise": surprise
        }
        response = self.SESSION.get(f"http://{self.HOST}:{self.PORT}{self.APP_STATS_PATH}", params=query_params)
        c_threads, c_latency = response.text.split(",")
        if not self.ignore_response:
            self.latest_config = int(c_threads)
            self.latest_latency = int(c_latency)
        # response.raise_for_status()  # Raise an exception for non-2xx status codes

        # background_thread = threading.Thread(target=self._receive_in_other_thread)
        # background_thread.daemon = True
        # background_thread.start()

    # def _receive_in_other_thread(self):
    #     response = self.http_connection.getresponse()
    #     response_text = response.read().decode('utf-8')
    #     c_threads, x = response_text.split(",")
    #     if not self.ignore_response:
    #         self.latest_config = int(c_threads)
    #     # print(f"Received {c_threads}")

    def get_latest_stream_config(self):
        return self.latest_config, self.latest_latency

    def override_stream_config(self, d_threads):
        self.ignore_response = True
        self.latest_config = d_threads
