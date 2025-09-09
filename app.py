import os
import threading
import uuid
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, send_from_directory, jsonify
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 100MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Load model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
model = AutoModelForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to(device)

# Global dictionary to track processing status
processing_status = {}

def estimate_depth(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    depth = processor.post_process_depth_estimation(
        outputs, source_sizes=[(image.height, image.width)]
    )[0]["predicted_depth"]
    depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize to 0-1
    depth_map = depth.squeeze().cpu().numpy()
    return depth_map

def create_stereo_pair(frame, depth_map, max_shift=30):
    h, w = depth_map.shape
    left_img = np.zeros_like(frame)
    right_img = np.zeros_like(frame)

    # Vectorized approach for better performance
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    shifts = (depth_map * max_shift).astype(np.int32)

    # Left image pixels shifted right
    new_x_left = np.clip(xs + shifts, 0, w-1)
    new_y = ys
    left_img[new_y, new_x_left] = frame[ys, xs]

    # Right image pixels shifted left
    new_x_right = np.clip(xs - shifts, 0, w-1)
    right_img[new_y, new_x_right] = frame[ys, xs]

    # Fill black pixels simply by inpainting could be added here (skipped for brevity)
    # For now, dilate pixels to fill some black areas
    kernel = np.ones((3,3),np.uint8)
    left_img = cv2.dilate(left_img, kernel, iterations=1)
    right_img = cv2.dilate(right_img, kernel, iterations=1)

    return left_img, right_img

def process_video(task_id, input_path, output_path):
    processing_status[task_id] = {'progress': 0, 'status': 'Extracting frames'}

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    left_frames = []
    right_frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        depth_map = estimate_depth(frame)
        left_img, right_img = create_stereo_pair(frame, depth_map, max_shift=30)

        left_frames.append(left_img)
        right_frames.append(right_img)

        processing_status[task_id]['progress'] = int((i+0.5) / total_frames * 50)
        processing_status[task_id]['status'] = f'Processed frame {i+1} / {total_frames}'

    cap.release()

    # Write left and right eye videos temporarily
    h, w, _ = left_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    left_path = f'{app.config["PROCESSED_FOLDER"]}/{task_id}_left.mp4'
    right_path = f'{app.config["PROCESSED_FOLDER"]}/{task_id}_right.mp4'

    out_left = cv2.VideoWriter(left_path, fourcc, fps, (w, h))
    out_right = cv2.VideoWriter(right_path, fourcc, fps, (w, h))

    processing_status[task_id]['status'] = 'Writing intermediate videos'

    for i in range(len(left_frames)):
        out_left.write(left_frames[i])
        out_right.write(right_frames[i])
        processing_status[task_id]['progress'] = 50 + int((i+1)/total_frames * 30)

    out_left.release()
    out_right.release()

    # Stitch side-by-side video for VR180 using ffmpeg
    processing_status[task_id]['status'] = 'Stitching side-by-side VR180 video'

    # Command: ffmpeg -i left.mp4 -i right.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2" output.mp4
    cmd = [
        'ffmpeg',
        '-y',
        '-i', left_path,
        '-i', right_path,
        '-filter_complex', 'hstack=inputs=2',
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    processing_status[task_id]['progress'] = 100
    processing_status[task_id]['status'] = 'Completed'

    # Clean up temporary videos
    os.remove(left_path)
    os.remove(right_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['video']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file:
            # Save uploaded file
            filename = f'{uuid.uuid4().hex}.mp4'
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Prepare output path
            output_filename = filename.replace('.mp4', '_vr180.mp4')
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

            # Start processing thread
            task_id = uuid.uuid4().hex
            threading.Thread(target=process_video, args=(task_id, upload_path, output_path)).start()

            # Redirect with task id
            return render_template('index.html', task_id=task_id, output_filename=output_filename)

    return render_template('index.html')

@app.route('/status/<task_id>')
def status(task_id):
    status_info = processing_status.get(task_id)
    if not status_info:
        return jsonify({'progress': 0, 'status': 'No task found'})
    return jsonify(status_info)

@app.route('/preview/<filename>')
def preview_video(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/download/<filename>')
def download_video(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
