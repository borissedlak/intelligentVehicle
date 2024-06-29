import os
import time

import cv2
from pyzbar.pyzbar import decode

import utils
from monitor.DeviceMetricReporter import DeviceMetricReporter
from monitor.ServiceMetricReporter import ServiceMetricReporter
from services.VehicleService import VehicleService

DEVICE_NAME = utils.get_ENV_PARAM("DEVICE_NAME", "Unknown")


class QrDetector(VehicleService):
    class Parameters:
        def __init__(self, source_pixel, source_fps):
            self.source_pixel = source_pixel
            self.source_fps = source_fps

    def __init__(self, show_results=False):
        super().__init__()
        ROOT = os.path.dirname(__file__)
        self.video_path = ROOT + "/data/pamela_reif_cut.mp4"
        self.simulate_fps = True

        self.device_metric_reporter = DeviceMetricReporter()
        self.service_metric_reporter = ServiceMetricReporter("QR")

        self.show_result = show_results
        self.initialize_video()

        if not self.cap.isOpened():
            print("Error opening video ...")
            return

    def process_one_iteration(self, params):
        source_pixel, source_fps = int(params['pixel']), int(params['fps'])

        # print(f"Now processing: {params.source_pixel} p, {params.source_fps} FPS")
        available_time_frame = (1000 / source_fps)

        ret, original_frame = self.cap.read()
        if not ret:
            self.initialize_video()
            ret, original_frame = self.cap.read()

        start_time = time.time()
        gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        decoded_objects = decode(gray)
        combined_img = utils.highlight_qr_codes(original_frame, decoded_objects)

        if self.show_result:
            cv2.imshow("Detected Objects", combined_img)

        processing_time = (time.time() - start_time) * 1000.0
        pixel = combined_img.shape[0]

        service_blanket = self.service_metric_reporter.create_metrics(processing_time, source_fps, pixel)
        device_blanket = self.device_metric_reporter.create_metrics()
        merged_metrics = utils.merge_single_dicts(service_blanket["metrics"], device_blanket["metrics"])

        if self.simulate_fps:
            if processing_time < available_time_frame:
                time.sleep((available_time_frame - processing_time) / 1000)

        return merged_metrics

    def initialize_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
