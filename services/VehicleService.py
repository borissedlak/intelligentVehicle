import utils


class VehicleService:
    def __init__(self):
        self.device_metric_reporter = None

    def process_one_iteration(self, params):
        pass

    def report_to_mongo(self, metrics):
        self.device_metric_reporter.report_metrics(utils.COLLECTION_NAME, metrics)
