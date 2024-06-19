import os
import threading
from datetime import datetime

import GPUtil
import psutil
import pymongo

from consumption.ConsRegression import ConsRegression
from utils import is_jetson_host, DB_NAME

DEVICE_NAME = os.environ.get('DEVICE_NAME')
if DEVICE_NAME:
    print(f'Found ENV value for DEVICE_NAME: {DEVICE_NAME}')
else:
    DEVICE_NAME = "Unknown"
    print(f"Didn't find ENV value for DEVICE_NAME, default to: {DEVICE_NAME}")

LEADER_HOST = os.environ.get('LEADER_HOST')
if LEADER_HOST:
    print(f'Found ENV value for LEADER_HOST: {LEADER_HOST}')
else:
    LEADER_HOST = "localhost"
    print(f"Didn't find ENV value for LEADER_HOST, default to: {LEADER_HOST}")


class CyclicArray:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add(self, value):
        if len(self.buffer) < self.max_size:
            self.buffer.append(value)
        else:
            # When buffer is full, remove the oldest value and append the new one
            self.buffer.pop(0)
            self.buffer.append(value)

    def get(self):
        return self.buffer

    def average(self):
        if not self.buffer:
            return 0
        return sum(self.buffer) / len(self.buffer)


# TODO: Needs a device ID additionally if we have multiple devices with the same type
class DeviceMetricReporter:
    def __init__(self, gpu_available=0, gpu_avg_history_n=5):
        self.host = DEVICE_NAME
        self.consumption_regression = ConsRegression(self.host)
        self.mongo_client = pymongo.MongoClient(LEADER_HOST)[DB_NAME]
        self.gpu_available = gpu_available
        self.gpu_avg_history = None

        if gpu_avg_history_n > 0:
            self.gpu_avg_history = CyclicArray(gpu_avg_history_n)

        if is_jetson_host(self.host):
            from jtop.jtop import jtop
            self.jetson_metrics = jtop()
            self.jetson_metrics.start()

    def create_metrics(self):
        mem_buffer = psutil.virtual_memory()
        mem = (mem_buffer.total - mem_buffer.available) / mem_buffer.total * 100
        cpu = psutil.cpu_percent()
        cons = self.consumption_regression.predict(cpu, self.gpu_available)

        gpu = 0
        if is_jetson_host(self.host):  # Has Jetson lib defined
            # print(self.jetson_metrics.stats)

            gpu = self.jetson_metrics.stats['GPU']

            # TODO: The GPU values are way too unstable, I must fix this somehow, or make an average over the last 5 values
            if self.gpu_avg_history is not None:
                self.gpu_avg_history.add(gpu)
                gpu = self.gpu_avg_history.average()

            cons = self.jetson_metrics.stats['Power TOT'] / 1000
            mode = self.jetson_metrics.stats['nvp model']
        elif self.gpu_available:
            if len(GPUtil.getGPUs()) > 0:  # Has Nvidia GPU but is no Jetson
                gpu = int(GPUtil.getGPUs()[0].load * 100)
            else:  # Old workaround
                # frame_gpu_translation = {15: 30, 20: 40, 25: 65, 30: 75, 35: 80}  # Orin
                # frame_gpu_translation = {15: 35, 20: 50, 25: 70, 30: 80, 35: 85}  # Xavier
                # gpu = frame_gpu_translation[source_fps]
                raise RuntimeError("How come?")

        return {"target": self.host,
                "metrics": {"device_type": self.host, "cpu": cpu, "memory": mem, "consumption": cons,
                            "timestamp": datetime.now(), "gpu": gpu}}

    # @utils.print_execution_time
    def report_metrics(self, target, record):
        insert_thread = threading.Thread(target=self.run_detached, args=(target, record))
        insert_thread.start()

    def run_detached(self, target, record):
        self.mongo_client[target].insert_one(record)
