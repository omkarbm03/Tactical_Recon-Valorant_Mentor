from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model 

# Train the model
results = model.train(data='./datasets/data.yaml', epochs=50, imgsz=640)