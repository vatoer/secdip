import cv2
from ultralytics import YOLO
import torch

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

import cv2

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model(frame)

    # Initialize a dictionary to count each class
    class_counts = {}

    # Visualize the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0]  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class label

            # Update the class count
            if cls in class_counts:
                class_counts[cls] += 1
            else:
                class_counts[cls] = 1

            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{model.names[cls]}: {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Find the key for the 'person' class
    person_class_id = None
    for key, value in model.names.items():
        if value == 'person':
            person_class_id = key
            break

    # Display the number of persons detected in the top right corner
    num_persons = class_counts.get(person_class_id, 0)
    cv2.putText(frame, f'Persons: {num_persons}', (frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow('YOLOv8 Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()