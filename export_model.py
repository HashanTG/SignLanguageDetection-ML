from ultralytics import YOLO

# Use forward slashes for the path
model = YOLO('runs/detect/train6/weights/best.pt')  # Replace with the correct path

# Export the model to ONNX format
model.export(format='onnx')
