from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can choose 'yolov8s.pt' for a larger model

# Train the model
model.train(
    data='datayolo8/data.yaml',  # Replace with your actual dataset.yaml path
    epochs=50,                    # Number of epochs
    imgsz=640,                    # Image size
    batch=16,                     # Batch size
    workers=8                     # Number of workers for data loading
)
