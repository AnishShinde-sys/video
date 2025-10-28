#!/usr/bin/env python3
"""
Flask Web Application for AI Video Transformer
"""

import os
import json
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import threading
from datetime import datetime
from video_transformer import VideoTransformer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['SECRET_KEY'] = 'your-secret-key-change-this'

# Create necessary directories
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(parents=True, exist_ok=True)

# Store job status in memory (use Redis/database for production)
jobs = {}

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_async(job_id, video_path, prompt, output_dir):
    """Process video in background thread"""
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['step'] = 'Initializing transformer...'

        # Initialize transformer with context manager for proper cleanup
        api_key = os.getenv("GEMINI_API_KEY")
        with VideoTransformer(api_key=api_key, output_dir=output_dir) as transformer:
            # Step 1: Extract frames
            jobs[job_id]['step'] = 'Extracting frames from video...'
            jobs[job_id]['progress'] = 20
            step1_result = transformer.extract_frames(video_path)

            # Step 2: Enhance prompt
            jobs[job_id]['step'] = 'Enhancing prompt with AI...'
            jobs[job_id]['progress'] = 35
            step2_result = transformer.enhance_prompt(
                step1_result['frames_dir'],
                prompt
            )

            # Step 3: Generate image
            jobs[job_id]['step'] = 'Generating transformed image...'
            jobs[job_id]['progress'] = 50
            step3_result = transformer.generate_image(
                step1_result['frames_dir'],
                step2_result['action_prompt']
            )

            # Step 4: Analyze video
            jobs[job_id]['step'] = 'Analyzing original video for speech and movements...'
            jobs[job_id]['progress'] = 65
            step4_result = transformer.analyze_video(video_path)

            # Step 5: Generate final video
            jobs[job_id]['step'] = 'Generating final video (this may take 2-3 minutes)...'
            jobs[job_id]['progress'] = 80
            step5_result = transformer.generate_video(
                video_path,
                step3_result['generated_image'],
                step4_result['analysis'],
                step2_result['action_prompt']
            )

        if step5_result['success']:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['step'] = 'Video generation complete!'
            jobs[job_id]['result'] = {
                'video_path': step5_result['video_path'],
                'video_url': url_for('download_file', filename=Path(step5_result['video_path']).name),
                'generated_image': step3_result['generated_image'],
                'image_url': url_for('download_file', filename=Path(step3_result['generated_image']).name),
                'duration': step5_result.get('generation_time', 0),
                'size_mb': step5_result.get('size_mb', 0)
            }
        else:
            raise Exception(step5_result.get('error', 'Video generation failed'))

    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['step'] = f'Error: {str(e)}'

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start processing"""

    # Check if video file is present
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    prompt = request.form.get('prompt', '')

    if video_file.filename == '':
        return jsonify({'error': 'No video selected'}), 400

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    if not allowed_file(video_file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, mov, avi, mkv, webm'}), 400

    # Save uploaded file
    filename = secure_filename(video_file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{filename}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    video_file.save(video_path)

    # Create job
    job_id = f"job_{timestamp}"
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)

    jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'step': 'Video uploaded, queued for processing...',
        'prompt': prompt,
        'video_path': video_path,
        'created_at': datetime.now().isoformat()
    }

    # Start processing in background thread
    thread = threading.Thread(
        target=process_video_async,
        args=(job_id, video_path, prompt, output_dir)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'success': True,
        'job_id': job_id,
        'message': 'Video uploaded successfully. Processing started...'
    })

@app.route('/status/<job_id>')
def get_status(job_id):
    """Get job status"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify(jobs[job_id])

@app.route('/download/<filename>')
def download_file(filename):
    """Download generated file"""
    # Search in output directories
    for job_id in jobs:
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)

        # Check in videos folder
        video_path = os.path.join(output_dir, 'videos', filename)
        if os.path.exists(video_path):
            return send_file(video_path, as_attachment=True)

        # Check in generated folder
        image_path = os.path.join(output_dir, 'generated')
        for subdir in Path(image_path).glob('*'):
            file_path = subdir / filename
            if file_path.exists():
                return send_file(str(file_path), as_attachment=True)

    return jsonify({'error': 'File not found'}), 404

@app.route('/jobs')
def list_jobs():
    """List all jobs"""
    return jsonify({
        'jobs': [
            {
                'job_id': job_id,
                'status': job['status'],
                'prompt': job['prompt'],
                'created_at': job['created_at']
            }
            for job_id, job in jobs.items()
        ]
    })

if __name__ == '__main__':
    # Check for API keys
    if not os.getenv("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY not set in environment")
    if not os.getenv("EACHLABS_API_KEY"):
        print("WARNING: EACHLABS_API_KEY not set in environment")

    print("Starting AI Video Transformer Web Application...")
    print("Open http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
