from vidgear.gears import CamGear
import cv2
from ultralytics import YOLO
from flask import Flask,render_template,Response
import winsound

frequency = 1000  # Adjust the frequency as needed
duration = 100   # Adjust the duration as needed


app = Flask(__name__)

model = YOLO('Weight/map.pt')

stream = CamGear(source='https://youtu.be/WJi5ploevZ0?si=0wD1e-MtQ4P-D6Df', stream_mode=True,
                 logging=True).start()  # YouTube Video URL as input

# infinite loop
def generate_frame():
  while True:

    frame = stream.read()
    # do something with frame here
    results = model(frame)

    annotated_frame = results[0].plot()

    if results[0].probs.top1 == 0:
        winsound.Beep(frequency, duration)


    ret, buffer = cv2.imencode('.jpg', annotated_frame)
    annotated_frame = buffer.tobytes()

    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + annotated_frame + b'\r\n')


def generate_frame():
  while True:

    frame = stream.read()
    # do something with frame here
    results = model(frame)

    annotated_frame = results[0].plot()

    if results[0].probs.top1 == 0:
        winsound.Beep(frequency, duration)


    ret, buffer = cv2.imencode('.jpg', annotated_frame)
    annotated_frame = buffer.tobytes()

    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + annotated_frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to render the custom index.html file
@app.route('/')
def index():
    return render_template('index.html', video_source='/video_feed')

@app.route('/preparedness')
def preparedness():
    return render_template('preparedness.html')

@app.route('/resource')
def resource():
    return render_template('resource.html')

@app.route('/contect')
def contect():
    return render_template('contect.html')

if __name__ == '__main__':
    app.run(debug=True)