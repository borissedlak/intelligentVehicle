from datetime import datetime


# This might actually run as a detached thread, but I think it facilitates the linking of entries
class ServiceMetricReporter:
    def __init__(self, target):
        self.target = target

    def create_metrics(self, time, fps, pixel):
        return {"target": self.target,
                "metrics": {"delta": time, "fps": fps, "pixel": pixel, "timestamp": datetime.now()}}
