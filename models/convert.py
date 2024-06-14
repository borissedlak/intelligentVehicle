from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # load a pretrained model
path = model.export(format="onnx")  # export the model to ONNX format
