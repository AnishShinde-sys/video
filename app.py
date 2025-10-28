#!/usr/bin/env python3
"""
Flask Web Application for AI Video Transformer
"""

import os
import json
import logging
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import threading
from datetime import datetime
from video_transformer import VideoTransformer, setup_logging

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
        # Use Flask application context for url_for
        with app.app_context():
            jobs[job_id]['status'] = 'processing'
            jobs[job_id]['step'] = 'Initializing transformer...'

            # Check if logging is enabled via environment variable
            enable_logging = os.getenv('ENABLE_LOGGING', 'false').lower() == 'true'
            log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO)
            log_file = os.getenv('LOG_FILE', f'logs/job_{job_id}.log') if enable_logging else None

            # Initialize transformer with context manager for proper cleanup
            api_key = os.getenv("GEMINI_API_KEY")
            with VideoTransformer(api_key=api_key, output_dir=output_dir, 
                                enable_logging=enable_logging, log_level=log_level, 
                                log_file=log_file) as transformer:
                # Step 1: Extract frames
                jobs[job_id]['step'] = 'Extracting frames from video...'
                jobs[job_id]['progress'] = 15
                step1_result = transformer.extract_frames(video_path)

                # Step 2: Enhance prompt
                jobs[job_id]['step'] = 'Enhancing prompt with AI...'
                jobs[job_id]['progress'] = 25
                step2_result = transformer.enhance_prompt(
                    step1_result['frames_dir'],
                    prompt
                )

                # Step 3: Generate image
                jobs[job_id]['step'] = 'Generating transformed image...'
                jobs[job_id]['progress'] = 40
                step3_result = transformer.generate_image(
                    step1_result['frames_dir'],
                    step2_result['action_prompt']
                )

                # Step 4: Extract audio from original video
                jobs[job_id]['step'] = 'Extracting audio from original video...'
                jobs[job_id]['progress'] = 50
                step4_result = transformer.extract_audio_from_video(video_path)

                # Step 5: Analyze video
                jobs[job_id]['step'] = 'Analyzing original video for speech and movements...'
                jobs[job_id]['progress'] = 60
                step5_result = transformer.analyze_video(video_path)

                # Step 6: Generate video
                jobs[job_id]['step'] = 'Generating video (this may take 2-3 minutes)...'
                jobs[job_id]['progress'] = 75
                step6_result = transformer.generate_video(
                    video_path,
                    step3_result['generated_image'],
                    step5_result['analysis'],
                    step2_result['action_prompt']
                )

                # Step 7: Merge audio with generated video
                jobs[job_id]['step'] = 'Merging audio with generated video...'
                jobs[job_id]['progress'] = 90
                step7_result = transformer.merge_audio_with_video(
                    step6_result['video_path'],
                    step4_result
                )

            if step6_result['success']:
                jobs[job_id]['status'] = 'completed'
                jobs[job_id]['progress'] = 100
                jobs[job_id]['step'] = 'Video with audio complete!'
                
                # Get filenames for URLs (construct manually to avoid url_for issues in background thread)
                # Use the final video with audio as the main result
                final_video_filename = Path(step7_result).name
                image_filename = Path(step3_result['generated_image']).name
                
                # Log the final video creation
                print(f"✅ FINAL VIDEO CREATED: {final_video_filename}")
                print(f"   📁 Path: {step7_result}")
                print(f"   🔊 Audio: Included from original video")
                print(f"   📏 Size: {Path(step7_result).stat().st_size / (1024 * 1024):.2f} MB")
                print(f"   🌐 Will be served at: /view/{final_video_filename}")
                
                jobs[job_id]['result'] = {
                    'video_path': step7_result,  # Final video with audio
                    'video_filename': final_video_filename,
                    'video_url': f'/download/{final_video_filename}',
                    'video_view_url': f'/view/{final_video_filename}',
                    'generated_image': step3_result['generated_image'],
                    'image_filename': image_filename,
                    'image_url': f'/download/{image_filename}',
                    'image_view_url': f'/view/{image_filename}',
                    'duration': step6_result.get('generation_time', 0),
                    'size_mb': Path(step7_result).stat().st_size / (1024 * 1024) if Path(step7_result).exists() else 0,
                    'has_audio': True  # Flag to indicate this video has audio
                }
            else:
                raise Exception(step6_result.get('error', 'Video generation failed'))

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

@app.route('/view/<filename>')
def view_file(filename):
    """View/stream generated file (for video player)"""
    # Search in output directories
    for job_id in jobs:
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)

        # Check in final folder (videos with audio)
        final_path = os.path.join(output_dir, 'final')
        for subdir in Path(final_path).glob('*'):
            file_path = subdir / filename
            if file_path.exists():
                return send_file(str(file_path), mimetype='video/mp4')

        # Check in videos folder (original generated videos)
        video_path = os.path.join(output_dir, 'videos', filename)
        if os.path.exists(video_path):
            return send_file(video_path, mimetype='video/mp4')

        # Check in generated folder (images)
        image_path = os.path.join(output_dir, 'generated')
        for subdir in Path(image_path).glob('*'):
            file_path = subdir / filename
            if file_path.exists():
                # Determine mimetype based on extension
                ext = file_path.suffix.lower()
                mimetype = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
                return send_file(str(file_path), mimetype=mimetype)

    return jsonify({'error': 'File not found'}), 404

@app.route('/download/<filename>')
def download_file(filename):
    """Download generated file"""
    # Search in output directories
    for job_id in jobs:
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)

        # Check in final folder (videos with audio) - priority
        final_path = os.path.join(output_dir, 'final')
        for subdir in Path(final_path).glob('*'):
            file_path = subdir / filename
            if file_path.exists():
                return send_file(str(file_path), as_attachment=True)

        # Check in videos folder (original generated videos)
        video_path = os.path.join(output_dir, 'videos', filename)
        if os.path.exists(video_path):
            return send_file(video_path, as_attachment=True)

        # Check in generated folder (images)
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
    # Setup logging for the Flask app if enabled
    enable_app_logging = os.getenv('ENABLE_LOGGING', 'false').lower() == 'true'
    if enable_app_logging:
        log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO)
        log_file = os.getenv('LOG_FILE', 'logs/app.log')
        setup_logging(enable_app_logging, log_level, log_file)
        logging.info("Flask application logging enabled")
    
    # Check for API keys
    if not os.getenv("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY not set in environment")
        if enable_app_logging:
            logging.warning("GEMINI_API_KEY not set in environment")
    if not os.getenv("EACHLABS_API_KEY"):
        print("WARNING: EACHLABS_API_KEY not set in environment")
        if enable_app_logging:
            logging.warning("EACHLABS_API_KEY not set in environment")

    print("Starting AI Video Transformer Web Application...")
    print("Open http://localhost:5002 in your browser")
    print("Features: Audio extraction + Video generation + Audio merging")
    
    # Log configuration info
    if enable_app_logging:
        logging.info("Starting AI Video Transformer Web Application")
        logging.info("Features: Audio extraction + Video generation + Audio merging")
        logging.info(f"Logging enabled - Level: {os.getenv('LOG_LEVEL', 'INFO')}")
        logging.info(f"Log file: {os.getenv('LOG_FILE', 'logs/app.log')}")
    else:
        print("Logging disabled. Set ENABLE_LOGGING=true to enable.")
    
    # Disable reloader to prevent file descriptor leaks from multiple processes
    app.run(debug=True, host='0.0.0.0', port=5002, threaded=True, use_reloader=False)
