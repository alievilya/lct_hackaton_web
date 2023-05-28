from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
# model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8s-cls.yaml').load('yolov8s-cls.pt')  # build from YAML and transfer weights

model = YOLO('yolov8s-cls.pt')  # load a pretrained model (recommended for training)
# Train the model
model.train(data='/home/i_aliev/PythonProjects/Hack/datasets/Classification_split', epochs=200, imgsz=640)

# metrics = model.val("./datasets/Classification_split/test/bathroom/26.jpg")  # predict on an image
success1 = model.export(format="onnx")  # export the model to ONNX format
success2 = model.export(format="tflite")  # export the model to ONNX format

# Predict with the model
results = model("./datasets/videos/1.mp4")

# yolo predict model=./runs/classify/train5/weights/best.pt source='datasets/videos/1.mp4'
# yolo classify export model=./runs/classify/train5/weights/best.pt format=tflite
