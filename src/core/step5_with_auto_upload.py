#!/usr/bin/env python3
"""
STEP 5: Video Generation with EachLabs API (with automatic image upload)
- Takes the output image from Step 3
- Takes the complete analysis text from Step 4 as the prompt
- Automatically uploads image to get a public URL
- Sends to EachLabs API to generate a video
"""

import os
import sys
import json
import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import base64
from dotenv import load_dotenv
import certifi

# Load environment variables
load_dotenv()

# Colors for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RED = '\033[91m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{CYAN}{BOLD}{text}{RESET}")
    print(f"{CYAN}{'='*70}{RESET}\n")

def print_step(text):
    print(f"{YELLOW}→ {text}{RESET}")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_info(text):
    print(f"{BLUE}  {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_detail(text):
    print(f"{MAGENTA}  » {text}{RESET}")


class EachLabsVideoGenerator:
    def __init__(self, api_key: str = None):
        """Initialize EachLabs video generator"""
        # Use provided API key or get from environment
        if not api_key:
            api_key = os.getenv("EACHLABS_API_KEY")

        if not api_key:
            raise ValueError("EACHLABS_API_KEY not provided or found in environment")

        self.api_key = api_key
        self.base_url = os.getenv("EACHLABS_BASE_URL", "https://api.eachlabs.ai/v1")
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

        # Create a session with retry strategy to handle connection errors
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        print_success("Initialized EachLabs Video Generator")

    def upload_to_tmpfiles(self, image_path: str) -> Optional[str]:
        """
        Upload image to tmpfiles.org for temporary hosting (24 hours)
        This is a free, no-auth-required service
        """
        print_step("Uploading image to temporary hosting...")

        try:
            with open(image_path, 'rb') as f:
                files = {'file': ('image.jpg', f, 'image/jpeg')}
                response = requests.post(
                    'https://tmpfiles.org/api/v1/upload',
                    files=files,
                    timeout=30
                )

            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    # The URL format is: https://tmpfiles.org/12345/filename
                    # We need to convert it to direct link: https://tmpfiles.org/dl/12345/filename
                    temp_url = result['data']['url']
                    direct_url = temp_url.replace('tmpfiles.org/', 'tmpfiles.org/dl/')
                    print_success(f"Image uploaded successfully!")
                    print_info(f"Temporary URL (24 hours): {direct_url}")
                    return direct_url
                else:
                    print_error("Upload failed - service returned error")
                    return None
            else:
                print_error(f"Upload failed with status: {response.status_code}")
                return None

        except Exception as e:
            print_error(f"Failed to upload image: {str(e)}")
            return None

    def upload_to_fileio(self, image_path: str) -> Optional[str]:
        """
        Upload image to file.io as a backup option (expires in 1 day by default)
        """
        print_step("Trying file.io upload...")

        try:
            with open(image_path, 'rb') as f:
                files = {'file': ('image.jpg', f, 'image/jpeg')}
                response = requests.post(
                    'https://file.io',
                    files=files,
                    timeout=30
                )

            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    file_url = result['link']
                    print_success(f"Image uploaded to file.io!")
                    print_info(f"Temporary URL: {file_url}")
                    return file_url
                else:
                    print_error("file.io upload failed")
                    return None
            else:
                print_error(f"file.io upload failed with status: {response.status_code}")
                return None

        except Exception as e:
            print_error(f"Failed to upload to file.io: {str(e)}")
            return None

    def upload_image_automatically(self, image_path: str) -> Optional[str]:
        """
        Automatically upload image to get a public URL
        Tries multiple services in order
        """
        print_header("AUTO-UPLOADING IMAGE")

        # Try tmpfiles.org first
        url = self.upload_to_tmpfiles(image_path)
        if url:
            return url

        # Try file.io as backup
        url = self.upload_to_fileio(image_path)
        if url:
            return url

        print_error("All automatic upload methods failed")
        print_info("Please manually upload the image and provide a URL")
        return None

    def prepare_prompt_from_analysis(self, analysis_text: str, max_length: int = 500) -> str:
        """
        Convert the step 4 analysis into a concise, cinematic prompt for video generation
        Focus on making it suitable for video generation
        """
        print_step("Preparing cinematic prompt from analysis...")

        # Extract the speech content to understand what the person is saying
        lines = analysis_text.split('\n')
        speech_words = []
        movements = []

        for i, line in enumerate(lines):
            # Extract speech
            if line.strip().startswith('[00:') and '"' in line:
                # Extract the word being spoken
                parts = line.split('"')
                if len(parts) >= 2:
                    word = parts[1]
                    if word not in ['uh', 'um', 'ah']:
                        speech_words.append(word)

            # Look for key movements
            if 'Head' in line or 'Eyes' in line or 'expression' in line.lower():
                movements.append(line.strip())
                if len(movements) > 3:
                    break

        # Build context from speech
        speech_context = ' '.join(speech_words[:30])
        if speech_context:
            speech_summary = f"Person discussing: {speech_context}"
        else:
            speech_summary = "Person speaking expressively"

        # Create cinematic video generation prompt
        video_prompt = (
            f"Create a smooth, professional video of a {speech_summary}. "
            f"Natural facial expressions and subtle movements. "
            f"Cinematic quality with realistic motion, proper lip sync. "
            f"Professional lighting, 4K ultra-detailed, smooth transitions."
        )

        # Keep it concise for the API
        if len(video_prompt) > max_length:
            video_prompt = video_prompt[:max_length-3] + "..."

        print_success(f"Prepared cinematic prompt ({len(video_prompt)} chars)")
        print_detail(f"Prompt: {video_prompt[:200]}...")

        return video_prompt

    def create_prediction(self,
                         image_url: str,
                         prompt: str,
                         aspect_ratio: str = "16:9",
                         resolution: str = "1080p",
                         duration: int = 12) -> Dict:
        """
        Create a video generation prediction with EachLabs API
        """
        print_header("CREATING VIDEO PREDICTION")

        # Prepare the request payload
        payload = {
            "model": "seedance-v1-pro-fast-image-to-video",
            "version": "0.0.1",
            "input": {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "duration": duration,
                "image_url": image_url
            },
            "webhook_url": ""
        }

        print_step("Sending prediction request to EachLabs...")
        print_info(f"Model: {payload['model']}")
        print_info(f"Resolution: {resolution}")
        print_info(f"Duration: {duration} seconds (MAXIMUM)")
        print_info(f"Aspect Ratio: {aspect_ratio}")
        print_detail(f"Image URL: {image_url[:50]}...")

        try:
            response = requests.post(
                f"{self.base_url}/prediction/",
                headers=self.headers,
                json=payload,
                timeout=30
            )

            print_info(f"Response status: {response.status_code}")

            if response.status_code in [200, 201, 202]:
                result = response.json()
                # Try different possible field names for prediction ID
                prediction_id = (result.get('id') or
                               result.get('prediction_id') or
                               result.get('predictionID') or
                               result.get('predictionId'))

                if prediction_id:
                    print_success(f"Prediction created successfully!")
                    print_success(f"Prediction ID: {prediction_id}")
                    return {
                        "success": True,
                        "prediction_id": prediction_id,
                        "response": result
                    }
                else:
                    print_info("Response received but no prediction ID found")
                    print_detail(f"Response: {json.dumps(result, indent=2)}")
                    return {
                        "success": False,
                        "error": "No prediction ID in response",
                        "response": result
                    }
            else:
                print_error(f"API returned status: {response.status_code}")
                print_error(f"Response: {response.text[:500]}")
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code
                }

        except requests.exceptions.RequestException as e:
            print_error(f"Request error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            print_error(f"Unexpected error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_prediction_result(self, prediction_id: str, max_wait: int = 300) -> Dict:
        """
        Poll for prediction result with improved error handling
        """
        print_header("WAITING FOR VIDEO GENERATION")
        print_info(f"Prediction ID: {prediction_id}")
        print_info("This may take 1-3 minutes...")

        start_time = time.time()
        poll_interval = 3  # Start with 3 second intervals
        attempt = 0

        while (time.time() - start_time) < max_wait:
            attempt += 1
            elapsed = int(time.time() - start_time)

            try:
                print_step(f"Checking status... (Attempt {attempt}, {elapsed}s elapsed)")

                response = requests.get(
                    f"{self.base_url}/prediction/{prediction_id}",
                    headers={"X-API-Key": self.api_key},
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    status = result.get('status', 'unknown').lower()

                    print_info(f"Current status: {status}")

                    if status in ['succeeded', 'success', 'completed']:
                        print_success("Video generation complete!")

                        # Look for video URL in various possible locations
                        video_url = None
                        if 'output' in result:
                            if isinstance(result['output'], dict):
                                video_url = result['output'].get('video_url') or result['output'].get('url')
                            elif isinstance(result['output'], str):
                                video_url = result['output']

                        if not video_url and 'video_url' in result:
                            video_url = result['video_url']

                        if not video_url and 'url' in result:
                            video_url = result['url']

                        return {
                            "success": True,
                            "result": result,
                            "video_url": video_url,
                            "status": status
                        }
                    elif status == 'failed':
                        error_msg = result.get('error', 'Generation failed without specific error')
                        print_error(f"Generation failed: {error_msg}")
                        return {
                            "success": False,
                            "error": error_msg,
                            "result": result
                        }
                    elif status in ['processing', 'pending', 'starting', 'running']:
                        # Show progress if available
                        if 'progress' in result:
                            print_detail(f"Progress: {result['progress']}%")
                        time.sleep(poll_interval)
                        # Gradually increase interval
                        poll_interval = min(poll_interval + 0.5, 10)
                    else:
                        print_info(f"Received status: {status}")
                        print_detail(f"Full response: {json.dumps(result, indent=2)[:500]}")
                        time.sleep(poll_interval)
                elif response.status_code == 404:
                    print_error("Prediction not found - it may have expired or ID is incorrect")
                    return {
                        "success": False,
                        "error": "Prediction not found"
                    }
                else:
                    print_warning(f"Received status code: {response.status_code}")
                    print_detail(f"Response: {response.text[:200]}")
                    time.sleep(poll_interval)

            except requests.exceptions.Timeout:
                print_warning("Request timed out, retrying...")
                time.sleep(poll_interval)
            except Exception as e:
                print_error(f"Error while polling: {str(e)}")
                time.sleep(poll_interval)

        print_error(f"Timeout after {max_wait} seconds")
        return {
            "success": False,
            "error": f"Timeout after {max_wait} seconds"
        }

    def download_video(self, video_url: str, output_path: str) -> bool:
        """
        Download the generated video with retry logic
        """
        print_step(f"Downloading video...")
        print_detail(f"URL: {video_url[:100]}...")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(video_url, stream=True, timeout=60)

                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))

                    with open(output_path, 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    progress = (downloaded / total_size) * 100
                                    print(f"\r  Downloading: {progress:.1f}%", end='')

                    print()  # New line after progress
                    print_success(f"Video saved to: {output_path}")

                    # Verify file size
                    file_size = os.path.getsize(output_path)
                    print_info(f"File size: {file_size / (1024*1024):.2f} MB")

                    return True
                else:
                    print_error(f"Download failed with status: {response.status_code}")

            except Exception as e:
                print_error(f"Download attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    print_info("Retrying...")
                    time.sleep(2)

        return False

    def generate_video(self,
                      image_path: str,
                      analysis_text: str,
                      output_dir: str = "temp/step5_videos") -> Dict:
        """
        Complete video generation pipeline with automatic image upload
        """
        print_header("STEP 5: EACHLABS VIDEO GENERATION")

        # Create output directory
        output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        print_step(f"Output directory: {run_dir}")

        # Automatically upload image
        image_url = self.upload_image_automatically(image_path)

        if not image_url:
            # Ask user to provide URL if auto-upload failed
            print_info("\nAutomatic upload failed. You have two options:")
            print_info("1. Manually upload the image and provide the URL")
            print_info("2. Exit and try again later")

            manual_url = input("\nEnter image URL (or press Enter to exit): ").strip()
            if not manual_url:
                return {
                    "success": False,
                    "error": "No image URL provided"
                }
            image_url = manual_url

        # Prepare prompt from analysis
        video_prompt = self.prepare_prompt_from_analysis(analysis_text)

        # Save the generation details
        generation_info = {
            "timestamp": timestamp,
            "image_source": str(image_path),
            "image_url": image_url,
            "prompt": video_prompt,
            "settings": {
                "aspect_ratio": "16:9",
                "resolution": "1080p",
                "duration": 12  # MAX duration
            }
        }

        info_file = run_dir / "generation_info.json"
        with open(info_file, 'w') as f:
            json.dump(generation_info, f, indent=2)
        print_success(f"Saved generation info to: {info_file}")

        # Create prediction with MAX duration (12 seconds)
        prediction_result = self.create_prediction(
            image_url=image_url,
            prompt=video_prompt,
            aspect_ratio="16:9",
            resolution="1080p",
            duration=12  # MAX duration for best results
        )

        if not prediction_result["success"]:
            print_error("Failed to create prediction")

            # Save error log
            error_log = run_dir / "error_log.json"
            with open(error_log, 'w') as f:
                json.dump(prediction_result, f, indent=2)

            return prediction_result

        # Check if we got a prediction ID
        prediction_id = prediction_result.get("prediction_id")

        if not prediction_id:
            print_error("No prediction ID received")
            print_info("This might mean the API format has changed")

            # Save the response for debugging
            response_file = run_dir / "api_response.json"
            with open(response_file, 'w') as f:
                json.dump(prediction_result, f, indent=2)
            print_info(f"API response saved to: {response_file}")

            return prediction_result

        # Save prediction info
        prediction_file = run_dir / "prediction_created.json"
        with open(prediction_file, 'w') as f:
            json.dump({
                "prediction_id": prediction_id,
                "timestamp": datetime.now().isoformat(),
                "initial_response": prediction_result
            }, f, indent=2)

        # Wait for result
        print_info("\nVideo generation started...")
        result = self.get_prediction_result(prediction_id)

        # Save final result
        result_file = run_dir / "final_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        if result["success"]:
            video_url = result.get("video_url")

            if video_url:
                print_success(f"Video URL received: {video_url[:50]}...")

                # Download the video
                video_path = run_dir / "generated_video.mp4"
                if self.download_video(video_url, str(video_path)):
                    print_header("SUCCESS!")
                    print_success("Video generation and download complete!")
                    print_success(f"Video saved to: {video_path}")

                    return {
                        "success": True,
                        "video_path": str(video_path),
                        "video_url": video_url,
                        "run_dir": str(run_dir),
                        "output_dir": str(run_dir),  # For compatibility with app.py
                        "prediction_id": prediction_id
                    }
                else:
                    print_warning("Video generated but download failed")
                    print_info(f"You can manually download from: {video_url}")

                    return {
                        "success": True,
                        "video_url": video_url,
                        "run_dir": str(run_dir),
                        "output_dir": str(run_dir),  # For compatibility with app.py
                        "download_failed": True
                    }
            else:
                print_error("No video URL in response")
                print_info("Check the result file for full response")
                return result
        else:
            print_error("Video generation failed")
            print_info(f"Error details saved to: {result_file}")
            return result


def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}⚠ {text}{RESET}")


def main():
    """Main execution"""

    print_header("EACHLABS VIDEO GENERATION - STEP 5")
    print_info("Automatic image upload enabled")

    # Get API key
    api_key = os.getenv("EACHLABS_API_KEY")
    if not api_key:
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("EACHLABS_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break

    if not api_key:
        print_error("No EACHLABS_API_KEY found in .env file!")
        print_info("Please add your API key to the .env file")
        return 1

    print_success("API key loaded")

    # Find latest files from previous steps
    step3_dir = Path("temp/step3_single")
    step4_dir = Path("temp/step4_complete_analysis")

    # Get latest step 3 output image
    if step3_dir.exists():
        latest_runs = sorted(step3_dir.iterdir(), key=lambda x: x.name)
        if latest_runs:
            latest_step3 = latest_runs[-1]
            image_path = latest_step3 / "nanobana_output.jpg"

            if not image_path.exists():
                print_error(f"Output image not found: {image_path}")
                return 1

            print_success(f"Using image from Step 3: {image_path.name}")
            print_detail(f"Full path: {image_path}")
        else:
            print_error("No runs found in Step 3 directory!")
            return 1
    else:
        print_error("Step 3 output directory not found!")
        print_info("Please run step 3 first to generate the image")
        return 1

    # Get latest step 4 analysis
    if step4_dir.exists():
        analysis_files = list(step4_dir.glob("COMPLETE_ANALYSIS_*.txt"))
        if analysis_files:
            latest_analysis = sorted(analysis_files, key=lambda x: x.name)[-1]

            print_success(f"Using analysis from Step 4: {latest_analysis.name}")

            with open(latest_analysis, 'r', encoding='utf-8') as f:
                analysis_text = f.read()

            print_detail(f"Analysis file size: {len(analysis_text)} characters")
        else:
            print_error("No analysis files found in Step 4!")
            print_info("Please run step 4 first to generate the analysis")
            return 1
    else:
        print_error("Step 4 output directory not found!")
        print_info("Please run step 4 first to generate the analysis")
        return 1

    # Initialize generator
    print_step("Initializing video generator...")
    generator = EachLabsVideoGenerator(api_key)

    # Generate video with automatic upload
    print_info("\nStarting automated video generation pipeline...")
    print_info("This will:")
    print_info("1. Automatically upload the image to get a public URL")
    print_info("2. Create a cinematic prompt from the analysis")
    print_info("3. Send to EachLabs for video generation")
    print_info("4. Download the generated video")

    result = generator.generate_video(
        image_path=str(image_path),
        analysis_text=analysis_text
    )

    if result.get("success"):
        print_header("VIDEO GENERATION SUCCESSFUL!")

        if result.get("video_path"):
            print_success(f"Video file: {result['video_path']}")

        if result.get("video_url"):
            print_success(f"Video URL: {result['video_url']}")

        print_info(f"All files saved in: {result.get('run_dir')}")

        return 0
    else:
        print_error("Video generation failed")

        if result.get("error"):
            print_error(f"Error: {result['error']}")

        print_info("Check the logs in the output directory for more details")

        return 1


if __name__ == "__main__":
    sys.exit(main())