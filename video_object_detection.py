import csv
import itertools
import os
import time

import cv2

from detector import utils
from detector.DeviceMetricReporter import DeviceMetricReporter
from detector.ServiceMetricReporter import ServiceMetricReporter
from detector.YOLOv8ObjectDetector import YOLOv8ObjectDetector
from detector.utils import COLLECTION_NAME

# Benchmark for road race with 'video.mp4'
# PC CPU --> XX FPS
# PC GPU --> 45 FPS
# Laptop CPU --> 28 FPS
# Orin GPU --> 42 FPS
# Xavier GPU --> 34 FPS
# Xavier CPU --> 4 FPS


DEVICE_NAME = os.environ.get('DEVICE_NAME')
if DEVICE_NAME:
    print(f'Found ENV value for DEVICE_NAME: {DEVICE_NAME}')
else:
    DEVICE_NAME = "Unknown"
    print(f"Didn't find ENV value for DEVICE_NAME, default to: {DEVICE_NAME}")

model_path = "models/yolov8n.onnx"
detector = YOLOv8ObjectDetector(model_path, conf_threshold=0.5, iou_threshold=0.5)
simulate_fps = True

device_metric_reporter = DeviceMetricReporter(detector.gpu_available())
service_metric_reporter = ServiceMetricReporter("CV")

# cv2.namedWindow("Detected Objects", cv2.WINDOW_AUTOSIZE)

csv_values = []
csv_headers = []


def process_video(video_info, show_result=False, repeat=1, write_csv=False):
    global csv_values, csv_headers
    for (source_pixel, source_fps) in video_info:
        for x in range(repeat):

            print(f"Now processing: {source_pixel} p, {source_fps} FPS, Round {x + 1}")
            available_time_frame = (1000 / source_fps)
            cap = cv2.VideoCapture("data/pamela_reif_cut.mp4")

            # output_video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1280, 720))

            if not cap.isOpened():
                print("Error opening video ...")
                return

            while cap.isOpened():
                # Press key q to stop
                if cv2.waitKey(1) == ord('q'):
                    break

                try:
                    ret, original_frame = cap.read()
                    if not ret:
                        break

                    original_width, original_height = original_frame.shape[1], original_frame.shape[0]
                    ratio = original_height / source_pixel

                    frame = cv2.resize(original_frame, (int(original_width / ratio), int(original_height / ratio)))

                except Exception as e:
                    print(e)
                    continue

                start_time = time.time()
                boxes, scores, class_ids = detector.detect_objects(frame)
                combined_img = utils.merge_image_with_overlay(frame, boxes, scores, class_ids)

                # output_video.writ(combined_img)

                if show_result:
                    cv2.imshow("Detected Objects", combined_img)

                processing_time = (time.time() - start_time) * 1000.0
                print(f"Inference time: {processing_time:.2f} ms")

                pixel = combined_img.shape[0]

                service_blanket = service_metric_reporter.create_metrics(processing_time, source_fps, pixel)
                device_blanket = device_metric_reporter.create_metrics()

                # intersection_name = utils.get_mb_name(service_blanket["target"], device_blanket["target"])
                merged_metrics = utils.merge_single_dicts(service_blanket["metrics"], device_blanket["metrics"])

                if write_csv:
                    csv_headers = merged_metrics.keys()
                    csv_values.append(merged_metrics)
                else:
                    device_metric_reporter.report_metrics(COLLECTION_NAME, merged_metrics)

                if simulate_fps:
                    if processing_time < available_time_frame:
                        time.sleep((available_time_frame - processing_time) / 1000)

            # output_video.release()
    detector.print_benchmark()


if __name__ == "__main__":
    write_csv = False
    # process_video(video_info=itertools.product([480, 720, 1080], [15, 20, 25, 30, 35]),
    process_video(video_info=itertools.product([480], [25]),
                  show_result=False,
                  write_csv=write_csv,
                  repeat=50)

    if write_csv:
        with open(f"./analysis/performance/{DEVICE_NAME}.csv", 'w', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
            csv_writer.writeheader()
            csv_writer.writerows(csv_values)
