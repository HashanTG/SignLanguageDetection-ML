import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('runs/detect/train6/weights/best.pt')  # Replace with your trained model path

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera, change if using another camera

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Perform inference
    results = model.predict(source=frame, save=False)

    # Visualize results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('Real-Time Sign Detection', annotated_frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
