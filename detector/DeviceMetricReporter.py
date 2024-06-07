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

        if DEVICE_NAME in ["Orin", "Xavier"]:
            from jtop.jtop import jtop
            self.jetson_metrics = jtop()
            self.jetson_metrics.start()

    def create_metrics(self):
        mem_buffer = psutil.virtual_memory()
        mem = (mem_buffer.total - mem_buffer.available) / mem_buffer.total * 100
        cpu = psutil.cpu_percent()
        cons = self.consumption_regression.predict(cpu, self.gpu_available)

        gpu = 0
        if self.jetson_metrics is not None and DEVICE_NAME in ["Orin", "Xavier", "Nano"]: # Has Jetson lib defined
                #print(self.jetson_metrics.stats)
                gpu = self.jetson_metrics.stats['GPU']
                cons = self.jetson_metrics.stats['Power TOT'] / 1000
                mode = self.jetson_metrics.stats['nvp model']
        elif self.gpu_available:
            if len(GPUtil.getGPUs()) > 0 and DEVICE_NAME not in ["Orin", "Xavier"]: # Has Nvidia GPU but is no Jetson
                gpu = int(GPUtil.getGPUs()[0].load * 100)
            else: # Old workaround
                # frame_gpu_translation = {15: 30, 20: 40, 25: 65, 30: 75, 35: 80}  # Orin
                #frame_gpu_translation = {15: 35, 20: 50, 25: 70, 30: 80, 35: 85}  # Xavier
                #gpu = frame_gpu_translation[source_fps]   
                raise RuntimeError("How come?")     
        
        return {"target": self.target,
                "metrics": {"device_type": self.target, "cpu": cpu, "memory": mem, "consumption": cons,
                            "timestamp": datetime.now(), "gpu": gpu}}

    # @utils.print_execution_time
    def report_metrics(self, target, record):
        insert_thread = threading.Thread(target=self.run_detached, args=(target, record))
        insert_thread.start()

    def run_detached(self, target, record):
        # mongo_client = pymongo.MongoClient(MONGO_HOST)["metrics"]
        self.mongo_client[target].insert_one(record)
