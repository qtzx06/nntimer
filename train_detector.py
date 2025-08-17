from ultralytics import YOLO

# load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# train the model on the cube dataset
results = model.train(
    data='datasets/cube_data/vision/data.yaml',
    epochs=50,
    imgsz=640,
    device=0  # use the first available GPU
)

# evaluate the model's performance on the validation set
results = model.val()

# export the model to ONNX format for easier deployment
path = model.export(format='onnx')
