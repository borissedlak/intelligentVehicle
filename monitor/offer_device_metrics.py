import os
import time

from prometheus_client import start_http_server, Gauge

from DeviceMetricReporter import DeviceMetricReporter

DEVICE_NAME = os.environ.get('DEVICE_NAME')
if DEVICE_NAME:
    print(f'Found ENV value for DEVICE_NAME: {DEVICE_NAME}')
else:
    DEVICE_NAME = "Unknown"
    print(f"Didn't find ENV value for DEVICE_NAME, default to: {DEVICE_NAME}")

PORT = 8000
start_http_server(PORT)

device_metric_reporter = DeviceMetricReporter(gpu_available=False)

# Create a Counter metric
# registry = CollectorRegistry()
cpu_load = Gauge('cpu_load', 'Current CPU load', ['device_name'])
gpu_load = Gauge('gpu_load', 'Current GPU load', ['device_name'])
memory_load = Gauge('memory_load', 'Current memory load', ['device_name'])
consumption = Gauge('consumption', 'Current energy consumption', ['device_name'])

if __name__ == '__main__':
    print(f"Started offering metrics on port {PORT}...")
    while True:
        device_metrics = device_metric_reporter.create_metrics()
        print(device_metrics)

        cpu_load.labels(device_name=DEVICE_NAME).set(device_metrics["metrics"]["cpu"])
        gpu_load.labels(device_name=DEVICE_NAME).set(device_metrics["metrics"]["gpu"])
        memory_load.labels(device_name=DEVICE_NAME).set(device_metrics["metrics"]["memory"])
        consumption.labels(device_name=DEVICE_NAME).set(device_metrics["metrics"]["consumption"])
        time.sleep(1)
