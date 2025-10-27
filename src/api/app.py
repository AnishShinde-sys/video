"""
Flask Backend API for AI Video Editor
Integrates all 5 steps of the video processing pipeline
"""

import os
import json
import uuid
import threading
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import time

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import get_config

# Import existing step modules
from src.core import step1_frame_extraction as step1
from src.core import step2_gemini_direct as step2
from src.core import step3_nano_banana as step3
from src.core import step4_single_output as step4
from src.core import step5_with_auto_upload as step5

# Initialize Flask app
# Set template and static folders relative to project root
project_root = Path(__file__).parent.parent.parent
app = Flask(__name__,
            template_folder=str(project_root / "templates"),
            static_folder=str(project_root / "static"))

# Load configuration based on environment
config = get_config()
app.config.from_object(config)
config.init_app(app)

# Setup CORS with configuration
CORS(app,
     origins=config.CORS_ORIGINS,
     methods=config.CORS_METHODS,
     allow_headers=config.CORS_ALLOW_HEADERS)

# Setup logging based on configuration
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Job storage (in production, use Redis or database)
jobs = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_job_file(job_id):
    """Get job data from file"""
    job_file = Path(config.JOBS_FOLDER) / f"{job_id}.json"
    if job_file.exists():
        with open(job_file, 'r') as f:
            return json.load(f)
    return None

def save_job(job_id, data):
    """Save job data to file"""
    job_file = Path(config.JOBS_FOLDER) / f"{job_id}.json"
    with open(job_file, 'w') as f:
        json.dump(data, f, indent=2)

def process_video_pipeline(job_id, video_path, prompt):
    """
    Process video through all 5 steps of the pipeline
    Updates job status as it progresses
    """
    job_data = {
        'status': 'processing',
        'created_at': datetime.now().isoformat(),
        'video_path': video_path,
        'prompt': prompt,
        'steps': {
            'step1': {'status': 'pending', 'progress': 0},
            'step2': {'status': 'pending', 'progress': 0},
            'step3': {'status': 'pending', 'progress': 0},
            'step4': {'status': 'pending', 'progress': 0},
            'step5': {'status': 'pending', 'progress': 0}
        },
        'results': {}
    }

    try:
        # Step 1: Frame Extraction
        logger.info(f"Job {job_id}: Starting Step 1 - Frame Extraction")
        job_data['steps']['step1']['status'] = 'processing'
        save_job(job_id, job_data)

        result1 = step1.extract_frames_step1(video_path)
        if not result1 or not result1.get('success'):
            raise Exception(f"Frame extraction failed: {result1.get('error', 'Unknown error')}")

        job_data['steps']['step1']['status'] = 'completed'
        job_data['steps']['step1']['progress'] = 100
        job_data['results']['frames_dir'] = result1['output_dir']

        # Get actual frame files from the directory
        frames_dir = Path(result1['output_dir'])
        frame_files = sorted([f for f in frames_dir.glob('frame_*.jpg') if f.is_file() and f.stat().st_size > 0])
        job_data['results']['frames'] = [str(f) for f in frame_files]
        save_job(job_id, job_data)

        # Step 2: Prompt Enhancement
        logger.info(f"Job {job_id}: Starting Step 2 - Prompt Enhancement")
        job_data['steps']['step2']['status'] = 'processing'
        save_job(job_id, job_data)

        # Get first frame for reference - use the actual first frame file
        if not frame_files:
            raise Exception("No valid frame files found for prompt enhancement")

        first_frame_path = str(frame_files[0])
        if not os.path.exists(first_frame_path) or os.path.getsize(first_frame_path) == 0:
            raise Exception(f"First frame file is invalid or empty: {first_frame_path}")

        gemini_key = config.GEMINI_API_KEY
        if not gemini_key:
            raise Exception("GEMINI_API_KEY not found in environment variables")

        enhancer = step2.PromptEnhancer(gemini_key)
        try:
            result2 = enhancer.enhance_prompt(prompt, first_frame_path)
        finally:
            # Always close the session to prevent file descriptor leaks
            enhancer.close()

        if not result2 or not result2.get('success'):
            raise Exception(f"Prompt enhancement failed: {result2.get('error', 'Unknown error')}")

        job_data['steps']['step2']['status'] = 'completed'
        job_data['steps']['step2']['progress'] = 100
        job_data['results']['prompt_dir'] = result2['output_dir']
        job_data['results']['enhanced_prompt'] = result2.get('enhanced_prompt', prompt)
        save_job(job_id, job_data)

        # Step 3: Image Generation
        logger.info(f"Job {job_id}: Starting Step 3 - Image Generation")
        job_data['steps']['step3']['status'] = 'processing'
        save_job(job_id, job_data)

        # Pass the frames directory and enhanced prompt from previous steps
        frames_dir = job_data['results'].get('frames_dir')
        enhanced_prompt = job_data['results'].get('enhanced_prompt')

        if not frames_dir or not enhanced_prompt:
            raise Exception("Missing frames directory or enhanced prompt from previous steps")

        generator = step3.NanoBanaSingleOutput()
        result3 = generator.generate_output(frames_dir=frames_dir, enhanced_prompt=enhanced_prompt)

        if not result3 or not result3.get('success'):
            raise Exception(f"Image generation failed: {result3.get('error', 'Unknown error')}")

        # Find the generated image
        output_dir = Path(result3.get('output_dir', result3.get('run_dir', '')))

        # First try the expected filename
        nanobana_output = output_dir / "nanobana_output.jpg"
        if nanobana_output.exists():
            job_data['results']['generated_image'] = str(nanobana_output)
        else:
            # Fallback to any jpg in the directory
            output_images = list(output_dir.glob("*.jpg"))
            if output_images:
                job_data['results']['generated_image'] = str(output_images[0])

        job_data['steps']['step3']['status'] = 'completed'
        job_data['steps']['step3']['progress'] = 100
        job_data['results']['image_dir'] = result3.get('output_dir', result3.get('run_dir'))
        save_job(job_id, job_data)

        # Step 4: Video Analysis
        logger.info(f"Job {job_id}: Starting Step 4 - Video Analysis")
        job_data['steps']['step4']['status'] = 'processing'
        save_job(job_id, job_data)

        analyzer = step4.SingleFileVideoAnalyzer()
        result4 = analyzer.analyze_video(video_path)

        if not result4 or not result4.get('success'):
            raise Exception(f"Video analysis failed: {result4.get('error', 'Unknown error')}")

        job_data['steps']['step4']['status'] = 'completed'
        job_data['steps']['step4']['progress'] = 100
        job_data['results']['analysis_dir'] = result4['output_dir']
        job_data['results']['analysis'] = result4.get('analysis', 'No analysis available')
        save_job(job_id, job_data)

        # Step 5: Video Generation
        logger.info(f"Job {job_id}: Starting Step 5 - Video Generation")
        job_data['steps']['step5']['status'] = 'processing'
        save_job(job_id, job_data)

        # Get the generated image path and analysis text from previous steps
        generated_image = job_data['results'].get('generated_image')
        analysis_text = job_data['results'].get('analysis')

        if not generated_image or not analysis_text:
            raise Exception("Missing generated image or analysis text from previous steps")

        video_generator = step5.EachLabsVideoGenerator()
        result5 = video_generator.generate_video(
            image_path=generated_image,
            analysis_text=analysis_text
        )

        if not result5 or not result5.get('success'):
            raise Exception(f"Video generation failed: {result5.get('error', 'Unknown error')}")

        # Find the generated video
        video_dir = Path(result5['output_dir'])
        video_files = list(video_dir.glob("generated_video_*.mp4"))
        if video_files:
            job_data['results']['generated_video'] = str(video_files[0])

        job_data['steps']['step5']['status'] = 'completed'
        job_data['steps']['step5']['progress'] = 100
        job_data['results']['video_dir'] = result5['output_dir']
        save_job(job_id, job_data)

        # Mark job as completed
        job_data['status'] = 'completed'
        job_data['completed_at'] = datetime.now().isoformat()
        save_job(job_id, job_data)

        logger.info(f"Job {job_id}: Processing completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id}: Error during processing - {str(e)}")
        job_data['status'] = 'error'
        job_data['error'] = str(e)
        save_job(job_id, job_data)

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_video():
    """
    Start processing a video
    Accepts video file and prompt
    Returns job ID for status polling
    """
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        prompt = request.form.get('prompt', '')

        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        if not allowed_file(video_file.filename):
            return jsonify({'error': 'Invalid file format. Allowed formats: MP4, AVI, MOV, MKV, WEBM'}), 400

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Save uploaded video
        filename = secure_filename(video_file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{job_id}_{timestamp}_{filename}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)

        # Start processing in background thread
        thread = threading.Thread(
            target=process_video_pipeline,
            args=(job_id, video_path, prompt)
        )
        thread.start()

        # Store initial job info
        jobs[job_id] = {
            'status': 'processing',
            'created_at': datetime.now().isoformat(),
            'video_path': video_path,
            'prompt': prompt
        }

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Processing started successfully'
        }), 200

    except Exception as e:
        logger.error(f"Error starting processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get the current status of a processing job"""
    try:
        job_data = get_job_file(job_id)

        if not job_data:
            return jsonify({'error': 'Job not found'}), 404

        return jsonify(job_data), 200

    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<job_id>', methods=['GET'])
def get_job_results(job_id):
    """Get the results of a completed job"""
    try:
        job_data = get_job_file(job_id)

        if not job_data:
            return jsonify({'error': 'Job not found'}), 404

        if job_data['status'] != 'completed':
            return jsonify({'error': 'Job not completed yet'}), 400

        results = job_data.get('results', {})

        # Prepare URLs for frontend
        response_data = {
            'success': True,
            'enhanced_prompt': results.get('enhanced_prompt', ''),
            'analysis': results.get('analysis', ''),
            'frames': [],
            'video_url': ''
        }

        # Add frame URLs
        if 'frames' in results:
            for frame_path in results['frames']:
                if os.path.exists(frame_path):
                    # Convert to URL path
                    response_data['frames'].append(f'/api/file/{job_id}/frame/{os.path.basename(frame_path)}')

        # Add video URL
        if 'generated_video' in results and os.path.exists(results['generated_video']):
            response_data['video_url'] = f'/api/file/{job_id}/video/{os.path.basename(results["generated_video"])}'

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error getting job results: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/file/<job_id>/<file_type>/<filename>', methods=['GET'])
def serve_file(job_id, file_type, filename):
    """Serve generated files (frames, videos, etc.)"""
    try:
        job_data = get_job_file(job_id)

        if not job_data:
            return jsonify({'error': 'Job not found'}), 404

        results = job_data.get('results', {})

        if file_type == 'frame':
            # Serve frame image
            frames_dir = results.get('frames_dir')
            if frames_dir and os.path.exists(frames_dir):
                file_path = os.path.join(frames_dir, filename)
                if os.path.exists(file_path):
                    return send_file(file_path, mimetype='image/jpeg')

        elif file_type == 'video':
            # Serve generated video
            video_path = results.get('generated_video')
            if video_path and os.path.exists(video_path):
                return send_file(video_path, mimetype='video/mp4', as_attachment=True, download_name=filename)

        return jsonify({'error': 'File not found'}), 404

    except Exception as e:
        logger.error(f"Error serving file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cancel/<job_id>', methods=['DELETE'])
def cancel_job(job_id):
    """Cancel a running job and cleanup files"""
    try:
        job_data = get_job_file(job_id)

        if not job_data:
            return jsonify({'error': 'Job not found'}), 404

        # Update job status
        job_data['status'] = 'cancelled'
        save_job(job_id, job_data)

        # Cleanup uploaded video
        if 'video_path' in job_data and os.path.exists(job_data['video_path']):
            os.remove(job_data['video_path'])

        return jsonify({'success': True, 'message': 'Job cancelled successfully'}), 200

    except Exception as e:
        logger.error(f"Error cancelling job: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200

# Serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    logger.info(f"Starting AI Video Editor server in {config.FLASK_ENV} mode...")

    # Add security headers for production
    if config.FLASK_ENV == 'production':
        @app.after_request
        def set_security_headers(response):
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'SAMEORIGIN'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            if config.SESSION_COOKIE_SECURE:
                response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            return response

    # Log the access URL based on configuration
    if config.HOST == '127.0.0.1':
        logger.info(f"Access the application at: http://localhost:{config.PORT}")
    else:
        logger.info(f"Server listening on: http://{config.HOST}:{config.PORT}")

    # Run the application
    app.run(
        debug=config.DEBUG,
        host=config.HOST,
        port=config.PORT,
        threaded=True
    )