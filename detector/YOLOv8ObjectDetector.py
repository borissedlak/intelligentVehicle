import time

import cv2
import numpy as np
import onnxruntime

from detector.utils import xywh2xyxy, multiclass_nms


class YOLOv8ObjectDetector:
    def __init__(self, path, conf_threshold=0.7, iou_threshold=0.5):
        # Can be assumed to stay constant for simplicity
        self.img_width = None
        self.img_height = None

        self.model_input_height = None
        self.model_input_width = None
        self.session = None
        self.conf_threshold = conf_threshold
        self.hist = []
        self.iou_threshold = iou_threshold

        # Initialize model
        self.initialize_model(path)

    def initialize_model(self, path):
        # onnxruntime.get_available_providers() returns also the TensorrtExecutionProvider, though it is not supported by the lib
        self.session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider'])
        self.set_input_details()
        self.set_output_details()

    def gpu_available(self):
        return 1 if "CUDAExecutionProvider" in self.session.get_providers() else 0

    def detect_objects(self, image):
        input_tensor = self.prepare_model_input(image)

        # Perform inference on the image
        model_outputs = self.inference(input_tensor)
        boxes, scores, class_ids = self.process_output(model_outputs)
        return boxes, scores, class_ids

    def prepare_model_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.model_input_width, self.model_input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        delta = (time.perf_counter() - start) * 1000
        self.hist.append(delta)
        # print(f"Inference time: {delta:.2f} ms")
        return outputs

    # I don't understand this and I don't have the ambition to do so
    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array(
            [self.model_input_width, self.model_input_height, self.model_input_width, self.model_input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def set_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        input_shape = model_inputs[0].shape
        self.model_input_height = input_shape[2]
        self.model_input_width = input_shape[3]

    def set_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def print_benchmark(self):
        m = np.mean(self.hist)
        print(f"Average processing time: {m:.2f} ms")
        print(f"Respective FPS: {1000 / m:.0f}")

# if __name__ == '__main__':
#     from imread_from_url import imread_from_url
#
#     model_path = "../models/yolov8m.onnx"
#
#     # Initialize YOLOv8 object detector
#     yolov8_detector = YOLOv8ObjectDetector(model_path, conf_thres=0.3, iou_thres=0.5)
#
#     img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
#     img = imread_from_url(img_url)
#
#     # Detect Objects
#     yolov8_detector.detect_objects(img)
#
#     # Draw detections
#     combined_img = yolov8_detector.draw_detections(img)
#     cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
#     cv2.imshow("Output", combined_img)
#     cv2.waitKey(0)
