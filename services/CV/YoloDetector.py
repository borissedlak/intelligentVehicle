import os
import time

import cv2

import utils
from monitor.DeviceMetricReporter import DeviceMetricReporter
from monitor.ServiceMetricReporter import ServiceMetricReporter
from services.CV.YOLOv8ObjectDetector import YOLOv8ObjectDetector
from services.VehicleService import VehicleService

# Benchmark for road race with 'video.mp4'
# PC CPU --> XX FPS
# PC GPU --> 45 FPS
# Laptop CPU --> 28 FPS
# Orin GPU --> 42 FPS
# Xavier GPU --> 34 FPS
# Xavier CPU --> 4 FPS


DEVICE_NAME = utils.get_ENV_PARAM("DEVICE_NAME", "Unknown")


class YoloDetector(VehicleService):
    class Parameters:
        def __init__(self, source_pixel, source_fps):
            self.source_pixel = source_pixel
            self.source_fps = source_fps

    def __init__(self, show_results=False):
        super().__init__()
        ROOT = os.path.dirname(__file__)
        self.video_path = ROOT + "/data/traffic_junction.mp4"
        self.detector_480 = YOLOv8ObjectDetector(ROOT + "/models/yolov8n.onnx", conf_threshold=0.5, iou_threshold=0.5)
        self.detector_720 = YOLOv8ObjectDetector(ROOT + "/models/yolov8s.onnx", conf_threshold=0.5, iou_threshold=0.5)
        self.detector_1080 = YOLOv8ObjectDetector(ROOT + "/models/yolov8m.onnx", conf_threshold=0.5, iou_threshold=0.5)
        self.simulate_fps = True

        self.device_metric_reporter = DeviceMetricReporter(self.detector_480.gpu_available())
        self.service_metric_reporter = ServiceMetricReporter("CV")

        self.show_result = show_results
        self.initialize_video()

        # self.object_count = []
        self.current_pos = 0

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

        original_width, original_height = original_frame.shape[1], original_frame.shape[0]
        ratio = original_height / source_pixel

        frame = cv2.resize(original_frame, (int(original_width / ratio), int(original_height / ratio)))

        start_time = time.time()
        if source_pixel == 480:
            boxes, scores, class_ids = self.detector_480.detect_objects(frame)
        elif source_pixel == 720:
            boxes, scores, class_ids = self.detector_720.detect_objects(frame)
        elif source_pixel == 1080:
            boxes, scores, class_ids = self.detector_1080.detect_objects(frame)
        else:
            raise RuntimeError(f"What is this pixel {source_pixel}?")
        detection_rate = len(boxes) / utils.object_count_1080[self.current_pos]
        # print(f"Found {detection_rate} % of objects, i.e., {len(boxes)} out of {utils.object_count_1080[self.current_pos]}")
        self.current_pos += 1

        combined_img = utils.merge_image_with_overlay(frame, boxes, scores, class_ids)

        if self.show_result:
            cv2.imshow("Detected Objects", combined_img)

        processing_time = (time.time() - start_time) * 1000.0

        service_blanket = self.service_metric_reporter.create_metrics(processing_time, source_fps, pixel=source_pixel, rate=detection_rate)
        device_blanket = self.device_metric_reporter.create_metrics()
        merged_metrics = utils.merge_single_dicts(service_blanket["metrics"], device_blanket["metrics"])

        if self.simulate_fps:
            if processing_time < available_time_frame:
                time.sleep((available_time_frame - processing_time) / 1000)

        return merged_metrics

    def initialize_video(self):
        self.current_pos = 0
        self.cap = cv2.VideoCapture(self.video_path)
