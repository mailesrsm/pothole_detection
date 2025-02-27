import cv2
import torch
import numpy as np
import pyttsx3
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load the trained YOLOv8 model
model = YOLO(r"C:\Users\nanth\runs\detect\train15\weights\best.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Video writer settings
output_path = r"E:\PROTOTYPE\output\pothole_detection.avi"
if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = 20  # Adjust FPS if needed
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Distance Calculation Parameters
FOCAL_LENGTH = 800  # Adjust based on testing
REAL_WIDTH = 50  # Approx pothole width in cm

def calculate_distance(pothole_width_in_pixels):
    """Estimate distance based on width in pixels."""
    if pothole_width_in_pixels == 0:
        return None
    return (REAL_WIDTH * FOCAL_LENGTH) / pothole_width_in_pixels

def speak_warning(distance):
    """Speak pothole alert only once per detection."""
    text = f"Pothole detected ahead, distance: {distance:.2f} cm."
    engine.say(text)
    engine.runAndWait()

# Variable to track pothole detection
pothole_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    pothole_present = False  # Track if pothole exists in frame

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box coordinates
            pothole_width_pixels = x2 - x1
            distance = calculate_distance(pothole_width_pixels)

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # Display distance & label
            text = f"Pothole: {distance:.2f} cm" if distance else "Pothole"
            cv2.putText(frame, text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # If pothole is detected for the first time, play alert
            pothole_present = True
            if not pothole_detected:
                speak_warning(distance)
                pothole_detected = True  # Set flag to prevent repeated alerts

    # If no pothole is found, reset the detection flag
    if not pothole_present:
        pothole_detected = False

    # Save frame to output video
    out.write(frame)

    # Display the output frame using Matplotlib instead of cv2.imshow
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
