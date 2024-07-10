import os
import logging
from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import uuid
from moviepy.editor import VideoFileClip
import subprocess

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['GRAPHS_FOLDER'] = 'static/graphs'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRAPHS_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.DEBUG)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in the request'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        metrics, graphs, output_video_path = process_video(filepath)
        webm_filename = str(uuid.uuid4()) + '.webm'
        webm_filepath = os.path.join(app.config['OUTPUT_FOLDER'], webm_filename)
        convert_to_webm(output_video_path, webm_filepath)

        # Delete the original and processed video files
        os.remove(filepath)
        os.remove(output_video_path)
        return jsonify({'metrics': metrics, 'graphs': graphs, 'output_video': webm_filepath})

def convert_to_webm(input_video_path, output_webm_path):
    command = [
        'ffmpeg',
        '-i', input_video_path,
        '-c:v', 'libvpx-vp9',
        '-b:v', '1M',
        '-c:a', 'libopus',
        output_webm_path
    ]
    subprocess.run(command)

def process_video(filepath):
    cap = cv2.VideoCapture(filepath)
    frame_count = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    output_filename = str(uuid.uuid4()) + '.mp4'
    output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))

    gait_metrics = {
        'knee_angles': [],
        'ankle_angles': [],
        'hip_angles': [],
        'stride_lengths': [],
        'step_lengths': [],
        'swing_phases': [],
        'stance_phases': []
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

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

            gait_metrics['knee_angles'].append(knee_angle)
            gait_metrics['ankle_angles'].append(ankle_angle)
            gait_metrics['hip_angles'].append(hip_angle)
            gait_metrics['stride_lengths'].append(stride_length)
            gait_metrics['step_lengths'].append(step_length)
            gait_metrics['swing_phases'].append(swing_phase)
            gait_metrics['stance_phases'].append(stance_phase)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            frame_count += 1

        out.write(frame)

    cap.release()
    out.release()

    graphs = {}
    for key, values in gait_metrics.items():
        plt.figure()
        plt.plot(values)
        plt.title(f"{key.replace('_', ' ').title()} Over Time")
        plt.xlabel('Frame Number')
        plt.ylabel(key.replace('_', ' ').title())
        graph_path = os.path.join(app.config['GRAPHS_FOLDER'], f"{key}.png")
        plt.savefig(graph_path)
        plt.close()
        graphs[key] = f"/static/graphs/{key}.png"

    metrics = {key: np.mean(values) for key, values in gait_metrics.items()}
    return metrics, graphs, output_filepath

if __name__ == '__main__':
    app.run(debug=True)
