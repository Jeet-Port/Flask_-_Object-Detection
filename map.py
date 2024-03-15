import cv2
import numpy as np
from ultralytics import YOLO
import winsound

frequency = 1000  # Adjust the frequency as needed
duration = 50   # Adjust the duration as needed


# Capture video from the default camera
model = YOLO('Weight/map.pt')

cap = cv2.VideoCapture('video/map.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    if results[0].probs.top1 == 0:
        winsound.Beep(frequency, duration)

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the red color
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Find contours of the isolated red regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Draw the contours
    for contour in contours:
        if cv2.contourArea(contour) > 50: # Filter very small contours
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)  # Draw with green color


    # Display the frame
    cv2.imshow('Red Spot Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
