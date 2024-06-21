import os
import time

import cv2

import utils
from monitor.DeviceMetricReporter import DeviceMetricReporter
from monitor.ServiceMetricReporter import ServiceMetricReporter
from services.VehicleService import VehicleService
from services.YOLOv8ObjectDetector import YOLOv8ObjectDetector

# Benchmark for road race with 'video.mp4'
# PC CPU --> XX FPS
# PC GPU --> 45 FPS
# Laptop CPU --> 28 FPS
# Orin GPU --> 42 FPS
# Xavier GPU --> 34 FPS
# Xavier CPU --> 4 FPS


DEVICE_NAME = utils.get_ENV_PARAM("DEVICE_NAME", "Unknown")


class VideoDetector(VehicleService):
    class Parameters:
        def __init__(self, source_pixel, source_fps):
            self.source_pixel = source_pixel
            self.source_fps = source_fps

    def __init__(self, show_results=False):
        ROOT = os.path.dirname(__file__)
        self.model_path = ROOT + "/models/yolov8n.onnx"
        self.video_path = ROOT + "/data/pamela_reif_cut.mp4"
        self.detector = YOLOv8ObjectDetector(self.model_path, conf_threshold=0.5, iou_threshold=0.5)
        self.simulate_fps = True

        self.device_metric_reporter = DeviceMetricReporter(self.detector.gpu_available())
        self.service_metric_reporter = ServiceMetricReporter("CV")

        self.show_result = show_results
        self.write_csv = False
        # csv_values = []
        # csv_headers = []
        self.initialize_video()

        if not self.cap.isOpened():
            print("Error opening video ...")
            return

    def process_one_iteration(self, params):
        source_pixel, source_fps = int(params['pixel']), int(params['fps'])

        global csv_values, csv_headers

        # print(f"Now processing: {params.source_pixel} p, {params.source_fps} FPS")
        available_time_frame = (1000 / source_fps)

        ret, original_frame = self.cap.read()
        if not ret:
            self.initialize_video()
            ret, original_frame = self.cap.read()

        original_width, original_height = original_frame.shape[1], original_frame.shape[0]
        ratio = original_height / source_pixel

        frame = cv2.resize(original_frame, (int(original_width / ratio), int(original_height / ratio)))

        start_time = time.time()
        boxes, scores, class_ids = self.detector.detect_objects(frame)
        combined_img = utils.merge_image_with_overlay(frame, boxes, scores, class_ids)

        if self.show_result:
            cv2.imshow("Detected Objects", combined_img)

        processing_time = (time.time() - start_time) * 1000.0
        # print(f"Inference time: {processing_time:.2f} ms")

        pixel = combined_img.shape[0]

        service_blanket = self.service_metric_reporter.create_metrics(processing_time, source_fps, pixel)
        device_blanket = self.device_metric_reporter.create_metrics()

        # intersection_name = utils.get_mb_name(service_blanket["target"], device_blanket["target"])
        merged_metrics = utils.merge_single_dicts(service_blanket["metrics"], device_blanket["metrics"])

        if self.write_csv:
            csv_headers = merged_metrics.keys()
            csv_values.append(merged_metrics)
        else:
            self.device_metric_reporter.report_metrics(utils.COLLECTION_NAME, merged_metrics)

        if self.simulate_fps:
            if processing_time < available_time_frame:
                time.sleep((available_time_frame - processing_time) / 1000)

        return merged_metrics

    def initialize_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
