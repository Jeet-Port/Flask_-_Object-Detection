import cv2
from ultralytics import YOLO
# import winsound

frequency = 1000  # Adjust the frequency as needed
duration = 100   # Adjust the duration as needed


# Load the YOLOv8 model
model = YOLO('Weight/wind.pt')

# Open the video file
video_path = "video/wind.mp4"
cap = cv2.VideoCapture(video_path)

def sound():

    # winsound.Beep(frequency, duration)
    None


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        if results[0].probs.top1 == 1:
            sound()

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

