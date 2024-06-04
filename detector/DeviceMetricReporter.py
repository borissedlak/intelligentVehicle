import os
import threading
from datetime import datetime

import GPUtil
import psutil
import pymongo

from consumption.ConsRegression import ConsRegression

DEVICE_NAME = os.environ.get('DEVICE_NAME')
if DEVICE_NAME:
    print(f'Found ENV value for DEVICE_NAME: {DEVICE_NAME}')
else:
    DEVICE_NAME = "Unknown"
    print(f"Didn't find ENV value for DEVICE_NAME, default to: {DEVICE_NAME}")

MONGO_HOST = os.environ.get('MONGO_HOST')
if MONGO_HOST:
    print(f'Found ENV value for MONGO_HOST: {MONGO_HOST}')
else:
    MONGO_HOST = "localhost"
    print(f"Didn't find ENV value for MONGO_HOST, default to: {MONGO_HOST}")


class DeviceMetricReporter:
    def __init__(self, gpu_available=0):
        self.target = DEVICE_NAME
        self.consumption_regression = ConsRegression(self.target)
        self.mongo_client = pymongo.MongoClient(MONGO_HOST)["metrics"]
        self.gpu_available = gpu_available

    def create_metrics(self, source_fps):
        mem_buffer = psutil.virtual_memory()
        mem = (mem_buffer.total - mem_buffer.available) / mem_buffer.total * 100
        cpu = psutil.cpu_percent()
        cons = self.consumption_regression.predict(cpu, self.gpu_available)

        gpu = 0
        if self.gpu_available:
            if len(GPUtil.getGPUs()) > 0:
                gpu = int(GPUtil.getGPUs()[0].load * 100)
            else:
                # frame_gpu_translation = {15: 30, 20: 40, 25: 65, 30: 75, 35: 80}  # Orin
                frame_gpu_translation = {15: 35, 20: 50, 25: 70, 30: 80, 35: 85}  # Xavier
                gpu = frame_gpu_translation[source_fps]
                # Limitation: Initializing jtop takes way too long
                # from jtop.jtop import jtop
                # with jtop() as jetson:
                #     jetson_dict = jetson.stats
                #     gpu = jetson_dict['GPU']
                #     print(gpu)

        return {"target": self.target,
                "metrics": {"device_type": self.target, "cpu": int(cpu), "memory": int(mem), "consumption": cons,
                            "timestamp": datetime.now(), "gpu": gpu}}

    # @utils.print_execution_time
    def report_metrics(self, target, record):
        insert_thread = threading.Thread(target=self.run_detached, args=(target, record))
        insert_thread.start()

    def run_detached(self, target, record):
        # mongo_client = pymongo.MongoClient(MONGO_HOST)["metrics"]
        self.mongo_client[target].insert_one(record)
