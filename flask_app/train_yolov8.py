from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)
model = YOLO("runs/detect/train8/weights/best.pt")  # load a pretrained model

# Use the model
model.train(data="/home/i_aliev/PythonProjects/Hack/datasets/Detection/Full_aug/data.yaml", epochs=1)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# results = model("/datasets/Detection/test/images/3b734efb-e087-47a4-bb56-1ddf35dcdcd1_jpg.rf.e2da41e8308ae4e5ac359dc1b0422fcf.jpg")  # predict on an image
success = model.export(format="onnx")  # export the model to ONNX format
# /home/i_aliev/PythonProjects/Hack

# yolo predict model=./runs/detect/train8/weights/best.pt source='datasets/videos/2.mp4'
# yolo detect export model=./runs/detect/train8/weights/best.pt format=tflite
