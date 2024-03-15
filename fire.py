import cv2
from ultralytics import YOLO
# import winsound
import time

frequency = 1000  # Set the frequency (1000 Hz in this example)
duration = 20


# Load the YOLOv8 model
model = YOLO('Weight/firemain.pt')

# Open the video file:
video_path = "video/firevideo.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # winsound.Beep(frequency, duration)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
