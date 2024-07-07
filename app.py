import os
import logging
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import base64

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

logging.basicConfig(level=logging.DEBUG)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

is_capturing = False
frame_count = 0
gait_metrics = {
    'knee_angles': [],
    'ankle_angles': [],
    'hip_angles': [],
    'stride_lengths': [],
    'step_lengths': [],
    'swing_phases': [],
    'stance_phases': []
}

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_distance(a, b):
    return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

@socketio.on('start_capture')
def start_capture():
    global is_capturing, frame_count, gait_metrics
    is_capturing = True
    frame_count = 0
    gait_metrics = {key: [] for key in gait_metrics}
    logging.info("Capture started")

@socketio.on('stop_capture')
def stop_capture():
    global is_capturing
    is_capturing = False
    generate_pdf()
    logging.info("Capture stopped")

def generate_pdf():
    pdf_path = os.path.join(app.static_folder, 'gait_analysis_report.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica", 12)

    c.drawString(100, 750, "Gait Analysis Report")

    y = 700
    for key, values in gait_metrics.items():
        if values:
            avg_value = sum(values) / len(values)
            c.drawString(100, y, f"Average {key.replace('_', ' ').title()}: {avg_value:.2f}")
            y -= 20

    c.save()

    with open(pdf_path, "rb") as pdf_file:
        encoded_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')

    socketio.emit('pdf_ready', {'pdf_data': encoded_pdf})
    logging.info("PDF generated and sent to client")

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                try:
                    
                    landmarks = results.pose_landmarks.landmark
                    
                    for landmark in landmarks:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

                    connections = [(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                                   (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                                   (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                                   (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                                   (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                                   (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                                   (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                                   (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                                   (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                                   (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                                   (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                                   (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)]

                    for connection in connections:
                        start_landmark = connection[0]
                        end_landmark = connection[1]

                        if landmarks[start_landmark.value].visibility >= 0.5 and landmarks[end_landmark.value].visibility >= 0.5:
                            start_point = (int(landmarks[start_landmark.value].x * frame.shape[1]),
                                           int(landmarks[start_landmark.value].y * frame.shape[0]))
                            end_point = (int(landmarks[end_landmark.value].x * frame.shape[1]),
                                         int(landmarks[end_landmark.value].y * frame.shape[0]))

                            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)


                    knee_angle = calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    )
                    ankle_angle = calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                    )
                    hip_angle = calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    )

                    stride_length = calculate_distance(
                        [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                    )
                    step_length = stride_length / 2

                    if knee_angle > 170:
                        swing_phase = 0
                        stance_phase = 100
                    else:
                        swing_phase = 100
                        stance_phase = 0

                    if is_capturing:
                        gait_metrics['knee_angles'].append(knee_angle)
                        gait_metrics['ankle_angles'].append(ankle_angle)
                        gait_metrics['hip_angles'].append(hip_angle)
                        gait_metrics['stride_lengths'].append(stride_length)
                        gait_metrics['step_lengths'].append(step_length)
                        gait_metrics['swing_phases'].append(swing_phase)
                        gait_metrics['stance_phases'].append(stance_phase)

                        frame_count += 1

                    cv2.putText(frame, f"Knee Angle: {knee_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Ankle Angle: {ankle_angle:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Hip Angle: {hip_angle:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Stride Length: {stride_length:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Step Length: {step_length:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Swing Phase: {swing_phase}%", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Stance Phase: {stance_phase}%", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                except Exception as e:
                    logging.error(f"Error processing landmarks: {e}")

            else:
                # Display error message when full body is not detected
                cv2.putText(frame, "Please show your full body to begin", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.static_folder):
        os.makedirs(app.static_folder)
    socketio.run(app, debug=True)
