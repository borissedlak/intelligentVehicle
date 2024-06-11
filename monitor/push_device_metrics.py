import time

import psutil
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# Create a registry
registry = CollectorRegistry()
metric = Gauge('cpu_load', 'Last time a job successfully finished', registry=registry)

while True:
    # Define a Gauge metric
    metric.set(psutil.cpu_percent())
    push_to_gateway('localhost:9091', job='batch_job', registry=registry)
    print("Metrics pushed to the Pushgateway")
    time.sleep(1)
