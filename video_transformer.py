#!/usr/bin/env python3
"""
AI Video Transformer - All-in-One Pipeline
Complete video transformation with frame extraction, prompt enhancement,
image generation, video analysis, and final video synthesis

Usage:
    python video_transformer.py <video_path> <prompt> [--output OUTPUT_DIR]

Example:
    python video_transformer.py input.mp4 "make the person wear a red hat"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import shutil
import time
import psutil
import gc

# Image processing
from PIL import Image
from io import BytesIO
import numpy as np

# Video processing
import imageio_ffmpeg
import imageio.v3 as iio

# Add path for google-genai if needed
sys.path.append('/Users/anishshinde/Library/Python/3.9/lib/python/site-packages')

# Google Gemini APIs
import google.generativeai as genai_old  # For Steps 2, 4, 5 (vision/text)
from google import genai  # For Step 3 (image generation)
from google.genai import types

# Terminal colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RED = '\033[91m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{CYAN}{BOLD}{text}{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")

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

def print_resource_info(context=""):
    """Print detailed resource usage information"""
    try:
        process = psutil.Process()
        
        # Get file descriptor count
        try:
            num_fds = process.num_fds()
        except AttributeError:
            # Windows doesn't have num_fds, use num_handles instead
            num_fds = process.num_handles() if hasattr(process, 'num_handles') else "N/A"
        
        # Get memory info
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Get open files count
        try:
            open_files = len(process.open_files())
        except (psutil.AccessDenied, AttributeError):
            open_files = "N/A"
        
        # Get network connections count
        try:
            connections = len(process.connections())
        except (psutil.AccessDenied, AttributeError):
            connections = "N/A"
        
        # Check system limits
        try:
            import resource
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            fd_limit_info = f"{soft_limit}/{hard_limit}"
        except (ImportError, AttributeError):
            fd_limit_info = "N/A"
        
        print_detail(f"🔍 RESOURCE MONITOR [{context}]:")
        print_detail(f"   File Descriptors: {num_fds} (limit: {fd_limit_info})")
        print_detail(f"   Open Files: {open_files}")
        print_detail(f"   Network Connections: {connections}")
        print_detail(f"   Memory Usage: {memory_mb:.1f} MB")
        print_detail(f"   PIL Images in memory: {len([obj for obj in gc.get_objects() if isinstance(obj, Image.Image)])}")
        
        # Warning if approaching limits
        if isinstance(num_fds, int) and isinstance(soft_limit, int):
            if num_fds > soft_limit * 0.8:  # 80% of limit
                print_error(f"⚠️  WARNING: File descriptor usage is high ({num_fds}/{soft_limit})")
        
    except Exception as e:
        print_detail(f"⚠️  Resource monitoring failed: {e}")

def log_file_operation(operation, filename, success=True):
    """Log file operations for debugging"""
    status = "✅" if success else "❌"
    print_detail(f"{status} FILE OP: {operation} - {filename}")

def force_cleanup_connections():
    """Force cleanup of any lingering connections"""
    try:
        import requests
        # Close any existing sessions in the requests module
        if hasattr(requests, 'sessions'):
            # Clear the session registry if it exists
            pass
        
        # Force garbage collection
        gc.collect()
        
        # Try to close any open file descriptors that might be lingering
        import os
        try:
            # Get current process info
            process = psutil.Process()
            open_files = process.open_files()
            connections = process.connections()
            
            print_detail(f"🧹 CLEANUP: Found {len(open_files)} open files, {len(connections)} connections")
            
            # Log some details about what's open
            if len(connections) > 10:  # Only log if there are many connections
                for i, conn in enumerate(connections[:5]):  # Show first 5
                    print_detail(f"   Connection {i+1}: {conn.laddr} -> {conn.raddr if conn.raddr else 'N/A'} ({conn.status})")
                if len(connections) > 5:
                    print_detail(f"   ... and {len(connections) - 5} more connections")
                    
        except Exception as e:
            print_detail(f"⚠️  Cleanup inspection failed: {e}")
            
    except Exception as e:
        print_detail(f"⚠️  Force cleanup failed: {e}")


class VideoTransformer:
    """All-in-one AI video transformation pipeline"""

    def __init__(self, api_key: str = None, output_dir: str = "output"):
        """Initialize the video transformer"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not provided or found in environment")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Gemini APIs
        genai_old.configure(api_key=self.api_key)
        self.vision_model = genai_old.GenerativeModel('gemini-2.0-flash')
        self.image_client = None  # Lazy init for image generation

        print_success("Video Transformer initialized")
        print_resource_info("INIT")
        
        # Force cleanup of any lingering connections from previous runs
        force_cleanup_connections()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        print_resource_info("BEFORE_CLEANUP")
        self._close_image_client()
        force_cleanup_connections()  # Force cleanup of connections
        gc.collect()  # Force garbage collection
        print_detail("[DEBUG] VideoTransformer context manager cleanup complete")
        print_resource_info("AFTER_CLEANUP")

    def _get_image_client(self):
        """Lazy initialize image generation client"""
        if self.image_client is None:
            print_resource_info("BEFORE_CLIENT_CREATE")
            print_detail(f"[DEBUG] Creating new genai.Client...")
            # Set API key in environment variable for client to pick up
            os.environ['GOOGLE_API_KEY'] = self.api_key
            print_detail(f"[DEBUG] Set GOOGLE_API_KEY in environment")
            # Initialize without api_key parameter - let it read from env
            self.image_client = genai.Client()
            print_detail(f"[DEBUG] genai.Client created successfully")
            print_resource_info("AFTER_CLIENT_CREATE")
        return self.image_client

    def _close_image_client(self):
        """Safely close image client"""
        if self.image_client is not None:
            print_resource_info("BEFORE_CLIENT_CLOSE")
            print_detail(f"[DEBUG] Closing image client...")
            try:
                if hasattr(self.image_client, '_api_client'):
                    self.image_client.close()
                    print_detail(f"[DEBUG] Client.close() called successfully")
                else:
                    print_detail(f"[DEBUG] Client has no _api_client attribute, skipping close")
            except Exception as e:
                print_detail(f"[DEBUG] Error closing client: {e}")
            finally:
                self.image_client = None
                print_detail(f"[DEBUG] Image client closed")
                print_resource_info("AFTER_CLIENT_CLOSE")

    # ================================================================================
    # STEP 1: FRAME EXTRACTION
    # ================================================================================

    def extract_frames(self, video_path: str, num_frames: int = 7) -> Dict:
        """Extract key frames from video"""
        print_header("STEP 1: FRAME EXTRACTION")
        print_resource_info("EXTRACT_FRAMES_START")
        print_detail(f"[DEBUG] Starting frame extraction for: {video_path}")

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frames_dir = self.output_dir / "frames" / timestamp
        frames_dir.mkdir(parents=True, exist_ok=True)

        print_step(f"Loading video: {video_path.name}")
        print_detail(f"[DEBUG] Output directory: {frames_dir}")

        # Read video metadata
        print_detail(f"[DEBUG] Opening video reader...")
        print_resource_info("BEFORE_VIDEO_READ")
        try:
            log_file_operation("OPEN_VIDEO_READER", str(video_path))
            reader = iio.imiter(str(video_path))
            meta = iio.immeta(str(video_path))
            print_detail(f"[DEBUG] Video metadata retrieved successfully")
            print_resource_info("AFTER_VIDEO_READ")
        except Exception as e:
            print_error(f"[DEBUG] Failed to read video metadata: {e}")
            log_file_operation("OPEN_VIDEO_READER", str(video_path), success=False)
            raise

        fps = meta.get('fps', 30)
        frame_count = meta.get('nframes', 0)
        duration = meta.get('duration', 0)

        # Get first frame for dimensions
        print_detail(f"[DEBUG] Reading first frame...")
        first_frame = next(reader)
        height, width = first_frame.shape[:2]
        print_detail(f"[DEBUG] First frame dimensions: {width}x{height}")

        print_info(f"Resolution: {width}x{height}")
        print_info(f"FPS: {fps:.2f}")
        print_info(f"Total frames: {frame_count}")
        print_info(f"Duration: {duration:.2f} seconds")

        # Calculate frame indices to extract
        if frame_count > 0 and not np.isinf(frame_count):
            indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        else:
            # Fallback for unknown frame count
            indices = list(range(num_frames))

        print_step(f"Extracting {num_frames} frames...")

        # Extract frames
        print_detail(f"[DEBUG] Creating new video reader for extraction...")
        reader = iio.imiter(str(video_path))
        extracted_frames = []
        frame_idx = 0

        print_detail(f"[DEBUG] Starting frame iteration...")
        for frame in reader:
            if frame_idx in indices:
                # Determine section
                total = frame_count if frame_count > 0 and not np.isinf(frame_count) else num_frames
                if frame_idx < total * 0.2:
                    section = "start"
                elif frame_idx > total * 0.8:
                    section = "end"
                else:
                    section = "middle"

                # Save frame
                frame_num = len(extracted_frames) + 1
                filename = f"frame_{frame_num:02d}_{section}_{frame_idx:04d}.jpg"
                filepath = frames_dir / filename

                print_detail(f"[DEBUG] Converting frame {frame_num} to PIL Image...")
                # Convert to PIL Image and save
                img = Image.fromarray(frame)
                print_detail(f"[DEBUG] Saving frame {frame_num} to {filepath}...")
                img.save(filepath, quality=95)
                img.close()  # Explicitly close image
                print_detail(f"[DEBUG] Frame {frame_num} saved and closed")

                extracted_frames.append(str(filepath))
                timestamp_sec = frame_idx / fps if fps > 0 else 0

                print_success(f"Frame {frame_num}/{num_frames}: {section} @ {timestamp_sec:.2f}s")
                print_info(f"Size: {os.path.getsize(filepath) / 1024:.1f} KB")

            frame_idx += 1
            if len(extracted_frames) >= num_frames:
                print_detail(f"[DEBUG] Extracted {len(extracted_frames)} frames, breaking loop")
                break

        print_detail(f"[DEBUG] Frame extraction loop complete")

        # Save metadata
        metadata = {
            "video_path": str(video_path),
            "timestamp": timestamp,
            "frames": extracted_frames,
            "num_frames": len(extracted_frames),
            "video_info": {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration
            }
        }

        metadata_file = frames_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print_header("STEP 1 COMPLETE: FRAME EXTRACTION")
        print_success(f"Extracted {len(extracted_frames)} frames")
        print_info(f"Output: {frames_dir}")

        return {
            "success": True,
            "frames_dir": str(frames_dir),
            "frames": extracted_frames,
            "metadata": metadata
        }

    # ================================================================================
    # STEP 2: ACTION-FOCUSED PROMPT ENHANCEMENT
    # ================================================================================

    def enhance_prompt(self, frames_dir: str, user_prompt: str) -> Dict:
        """
        Enhance user prompt focusing on ACTIONS not visual modifications.
        Based on Gemini Nano Banana documentation - describe the scene you want,
        not how to modify the existing one.
        """
        print_header("STEP 2: ACTION-FOCUSED PROMPT ENHANCEMENT")
        print_detail(f"[DEBUG] Looking for frames in: {frames_dir}")

        frames_dir = Path(frames_dir)
        frame_files = sorted([f for f in frames_dir.glob("frame_*.jpg")])
        print_detail(f"[DEBUG] Found {len(frame_files)} frame files")

        if not frame_files:
            raise ValueError("No frames found")

        # Use first frame as reference
        reference_frame = frame_files[0]
        print_detail(f"[DEBUG] Opening reference frame: {reference_frame}")
        img = Image.open(reference_frame)
        print_detail(f"[DEBUG] Reference frame opened successfully")

        print_step("Analyzing reference frame...")
        print_info(f"Frame: {reference_frame.name}")
        print_info(f"Resolution: {img.size[0]}x{img.size[1]}")

        # NEW: Action-focused system prompt
        system_prompt = f"""You are an expert at creating prompts for Gemini's image generation (Nano Banana).

CRITICAL INSTRUCTION: Based on the Gemini documentation, you must "Describe the scene, don't just list keywords."
The prompt should describe WHAT THE FINAL IMAGE SHOULD SHOW, not HOW to modify the existing image.

User's transformation request: "{user_prompt}"
Reference image: [Provided below]

Analyze the reference image and create an action-focused prompt that describes the FINAL SCENE.

Provide your response in this exact JSON format:
{{
  "image_analysis": "Brief analysis of the current scene (people, objects, setting, lighting, mood)",
  "action_prompt": "A descriptive paragraph that tells the model WHAT TO SHOW in the final image. Focus on the subject's actions, appearance, environment, and atmosphere. DO NOT use phrases like 'add', 'remove', 'change', or 'modify'. Instead, describe the complete scene as it should appear.",
  "key_elements": [
    "List 3-5 key visual elements that must be present in the final image"
  ]
}}

EXAMPLE (for "make person wear red hat"):
GOOD: "A person with curly dark hair wearing a vibrant red baseball cap, looking directly at the camera in a home interior with wooden paneling. The cap sits naturally on their head with hair visible underneath. Soft natural lighting creates a casual, authentic atmosphere."

BAD: "Add a red baseball cap to the person's head. Modify the lighting to match..."

Focus on describing the COMPLETE SCENE, not modifications to the existing image."""

        print_step("Sending request to Gemini API...")
        print_detail(f"[DEBUG] Calling vision_model.generate_content...")

        # Generate enhanced prompt
        try:
            response = self.vision_model.generate_content([system_prompt, img])
            response_text = response.text
            print_detail(f"[DEBUG] Received response from Gemini API")
        except Exception as e:
            print_error(f"[DEBUG] Gemini API call failed: {e}")
            raise
        finally:
            # Close the image to free file handle
            print_detail(f"[DEBUG] Closing reference image")
            img.close()

        # Parse JSON response
        try:
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: use raw response as action prompt
            result = {
                "image_analysis": "Unable to parse structured response",
                "action_prompt": response_text,
                "key_elements": []
            }

        print_success("Received action-focused prompt from Gemini")

        # Display results
        print_header("ENHANCEMENT RESULTS")
        print_step("Scene Analysis:")
        print_info(result.get("image_analysis", "N/A"))
        print_step("Action-Focused Prompt:")
        print_success(result.get("action_prompt", "N/A"))

        if result.get("key_elements"):
            print_step("Key Elements:")
            for element in result['key_elements']:
                print_info(f"  • {element}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_dir = self.output_dir / "prompts" / timestamp
        prompt_dir.mkdir(parents=True, exist_ok=True)

        output_data = {
            "user_prompt": user_prompt,
            "action_prompt": result.get("action_prompt", ""),
            "image_analysis": result.get("image_analysis", ""),
            "key_elements": result.get("key_elements", []),
            "reference_frame": str(reference_frame),
            "timestamp": timestamp
        }

        output_file = prompt_dir / "action_prompt.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Copy reference frame
        shutil.copy(reference_frame, prompt_dir / "reference_frame.jpg")

        print_header("STEP 2 COMPLETE: ACTION PROMPT CREATED")
        print_success("Successfully created action-focused prompt")
        print_info(f"Output: {prompt_dir}")

        return {
            "success": True,
            "action_prompt": result.get("action_prompt", ""),
            "prompt_dir": str(prompt_dir),
            "analysis": result
        }

    # ================================================================================
    # STEP 3: IMAGE GENERATION (NANO BANANA)
    # ================================================================================

    def generate_image(self, frames_dir: str, action_prompt: str) -> Dict:
        """
        Generate transformed image using Gemini image generation.
        Rewritten to follow official docs exactly.
        """
        print_header("STEP 3: IMAGE GENERATION (GEMINI)")
        print_resource_info("GENERATE_IMAGE_START")
        print_detail(f"[DEBUG] Step 3 started - frames_dir: {frames_dir}")

        frames_dir = Path(frames_dir)
        frame_files = sorted([f for f in frames_dir.glob("frame_*.jpg")])
        print_detail(f"[DEBUG] Found {len(frame_files)} frame files")

        if not frame_files:
            raise ValueError("No frames found")

        print_success(f"Found {len(frame_files)} total frames")

        # Select only first frame for testing
        selected_frames = [frame_files[0]]

        # Load frames into PIL Image objects with proper resource management
        reference_images = []
        buffers = []  # Track BytesIO objects for cleanup
        
        try:
            print_detail(f"[DEBUG] Loading {len(selected_frames)} selected frames...")
            for i, frame_path in enumerate(selected_frames):
                print_detail(f"[DEBUG] Opening frame {i+1}/{len(selected_frames)}: {frame_path}")
                img = Image.open(frame_path)
                reference_images.append(img)
                print_info(f"Loaded: {frame_path.name} ({img.size[0]}x{img.size[1]})")

            print_success(f"Loaded {len(reference_images)} reference frames")

            # Build prompt following docs pattern
            prompt = f"""Based on this reference image from a video, create a single photorealistic image showing:

{action_prompt}

Maintain the same composition, perspective, and lighting quality as the reference image."""

            print_step(f"Generating output image from {len(reference_images)} reference frame(s)...")
            print_info("Model: gemini-2.5-flash-image")

            # Use REST API directly to avoid SDK hang
            print_detail(f"[DEBUG] Using REST API instead of SDK...")
            import requests
            import base64

            # Encode images to base64 with proper buffer management
            print_detail(f"[DEBUG] Encoding {len(reference_images)} images to base64...")
            image_parts = []
            for i, img in enumerate(reference_images):
                buffer = BytesIO()
                buffers.append(buffer)  # Track for cleanup
                img.save(buffer, format='JPEG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_base64
                    }
                })
                print_detail(f"[DEBUG] Encoded image {i+1}/{len(reference_images)}")

            # Build request payload
            request_data = {
                "contents": [{
                    "parts": [{"text": prompt}] + image_parts
                }],
                "generationConfig": {
                    "response_modalities": ["Image"],
                    "image_config": {
                        "aspect_ratio": "1:1"
                    }
                }
            }

            # Make REST API call with session for proper connection management
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key={self.api_key}"
            print_detail(f"[DEBUG] Calling Gemini REST API...")
            print_resource_info("BEFORE_GEMINI_API_CALL")

            # Use urllib3 directly with explicit connection management to avoid requests session leaks
            import urllib3
            import ssl
            
            # Create a new pool manager with strict limits
            http = urllib3.PoolManager(
                num_pools=1,
                maxsize=1,
                block=True,
                cert_reqs=ssl.CERT_REQUIRED,
                ca_certs=None,
                timeout=urllib3.Timeout(connect=30.0, read=60.0)
            )
            
            try:
                print_detail(f"[DEBUG] Created urllib3 pool manager")
                print_resource_info("AFTER_POOL_CREATE")
                
                # Encode request data as JSON
                import json as json_module
                body = json_module.dumps(request_data).encode('utf-8')
                
                try:
                    print_detail(f"[DEBUG] Sending POST request...")
                    response = http.request(
                        'POST',
                        api_url,
                        body=body,
                        headers={
                            'Content-Type': 'application/json',
                            'Connection': 'close'
                        },
                        retries=urllib3.Retry(total=1, connect=1, read=1)
                    )
                    
                    print_detail(f"[DEBUG] POST request completed")
                    print_resource_info("AFTER_API_RESPONSE")
                    
                    if response.status != 200:
                        raise Exception(f"API returned status {response.status}: {response.data.decode('utf-8')}")
                    
                    result = json_module.loads(response.data.decode('utf-8'))
                    print_detail(f"[DEBUG] Response received successfully")
                    print_resource_info("AFTER_JSON_PARSE")
                    
                except Exception as e:
                    print_error(f"[DEBUG] API call failed: {e}")
                    print_resource_info("AFTER_API_ERROR")
                    raise
                    
            finally:
                # Ensure pool manager is properly closed
                try:
                    http.clear()
                    print_detail(f"[DEBUG] Cleared connection pool")
                    
                    # Force cleanup
                    del http
                    gc.collect()
                    
                    print_resource_info("AFTER_POOL_CLOSE")
                except Exception as e:
                    print_detail(f"[DEBUG] Error closing pool: {e}")

            # Extract generated image from response
            print_detail(f"[DEBUG] Extracting image from response...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_dir = self.output_dir / "generated" / timestamp
            image_dir.mkdir(parents=True, exist_ok=True)

            generated_image_path = None
            generated_image = None  # Track for cleanup
            
            if 'candidates' in result and len(result['candidates']) > 0:
                for part in result['candidates'][0]['content']['parts']:
                    if 'text' in part:
                        print_detail(f"[DEBUG] Text in response: {part['text'][:100]}...")
                    elif 'inlineData' in part:
                        print_detail(f"[DEBUG] Found image data in response")
                        # Decode base64 image with proper resource management
                        img_data = base64.b64decode(part['inlineData']['data'])
                        img_buffer = BytesIO(img_data)
                        try:
                            generated_image = Image.open(img_buffer)
                            output_path = image_dir / "generated_output.jpg"
                            generated_image.save(output_path)
                            generated_image_path = str(output_path)
                            print_success(f"Generated image saved: {output_path.name}")
                            print_info(f"Size: {generated_image.size[0]}x{generated_image.size[1]}")
                        finally:
                            # Close generated image
                            if generated_image:
                                generated_image.close()
                            img_buffer.close()
                        break

            if not generated_image_path:
                raise ValueError("No image data in response")

            # Save metadata
            metadata = {
                "timestamp": timestamp,
                "action_prompt": action_prompt,
                "frames_dir": str(frames_dir),
                "reference_frames": [str(f) for f in selected_frames],
                "generated_image": generated_image_path,
                "model": "gemini-2.5-flash-image",
                "num_reference_images": len(reference_images)
            }

            metadata_file = image_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Copy reference frames
            for i, frame_path in enumerate(selected_frames):
                dest_path = image_dir / f"reference_frame_{i+1:02d}.jpg"
                shutil.copy(frame_path, dest_path)

            print_header("STEP 3 COMPLETE: IMAGE GENERATION")
            print_success(f"Generated 1 output image from {len(reference_images)} reference frames")
            print_info(f"Output: {image_dir}")

            return {
                "success": True,
                "generated_image": generated_image_path,
                "image_dir": str(image_dir),
                "metadata": metadata
            }

        finally:
            # Clean up ALL resources in finally block
            print_detail(f"[DEBUG] Cleaning up resources...")
            print_resource_info("BEFORE_IMAGE_CLEANUP")
            
            # Close all reference images
            for i, img in enumerate(reference_images):
                try:
                    img.close()
                    log_file_operation("CLOSE_PIL_IMAGE", f"reference_image_{i}")
                except Exception as e:
                    log_file_operation("CLOSE_PIL_IMAGE", f"reference_image_{i}", success=False)
                    print_detail(f"[DEBUG] Error closing image {i}: {e}")
            
            # Close all BytesIO buffers
            for i, buffer in enumerate(buffers):
                try:
                    buffer.close()
                    log_file_operation("CLOSE_BUFFER", f"buffer_{i}")
                except Exception as e:
                    log_file_operation("CLOSE_BUFFER", f"buffer_{i}", success=False)
                    print_detail(f"[DEBUG] Error closing buffer {i}: {e}")
            
            # Force garbage collection
            gc.collect()
            print_resource_info("AFTER_IMAGE_CLEANUP")
                    
            print_detail(f"[DEBUG] Cleanup complete")

    # ================================================================================
    # STEP 4: VIDEO ANALYSIS
    # ================================================================================

    def analyze_video(self, video_path: str) -> Dict:
        """
        Perform ULTRA-DETAILED video analysis with second-by-second tracking:
        - Complete speech transcription with exact timestamps
        - Mouth movements and lip positions every second
        - Eye movements, blinks, gaze direction
        - Head position and movements
        - Body movements, gestures, posture
        - Facial expressions and micro-expressions
        - All actions synchronized by timestamp
        """
        print_header("STEP 4: ULTRA-DETAILED VIDEO ANALYSIS")

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print_step(f"Uploading video: {video_path.name}")

        # Upload video file
        video_file = genai_old.upload_file(str(video_path))
        print_success(f"Uploaded: {video_path.name}")

        # Wait for processing
        print_step("Waiting for video processing...")
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai_old.get_file(video_file.name)

        print_success("Video ready for ultra-detailed analysis")

        # ULTRA-COMPREHENSIVE analysis prompt based on Gemini video understanding docs
        analysis_prompt = """Analyze this video in EXTREME detail. Provide second-by-second tracking of EVERYTHING that happens.

## PART 1: COMPLETE SPEECH TRANSCRIPTION - **CRITICAL REQUIREMENT**
**IMPORTANT: This is the MOST CRITICAL part of the analysis. The video generation will use this transcript.**

Transcribe EVERY SINGLE WORD spoken with EXACT timestamps in [MM:SS.milliseconds] format.
**YOU MUST TRANSCRIBE EXACTLY WHAT IS SAID - DO NOT SKIP OR SUMMARIZE ANY WORDS.**

For each word/phrase, include:
- [Timestamp] "Exact words spoken" - Tone, Volume, Pace
- Tone (happy, sad, excited, angry, sarcastic, neutral, etc.)
- Volume (loud, normal, soft, whisper)
- Pace (fast, normal, slow)
- Emphasis or stress on specific words
- Pauses and their duration

After the detailed transcription, provide:
**FULL TRANSCRIPT (for video generation):**
Provide the complete spoken text as one continuous paragraph with all the exact words that must be said in the generated video.

Example format:
[00:00.234] "Hello" - Tone: excited, Volume: loud, Pace: normal
[00:00.567] [pause 0.5s]
[00:01.123] "everyone" - Tone: friendly, Volume: normal, Pace: normal

**FULL TRANSCRIPT:** "Hello everyone..."

## PART 2: SECOND-BY-SECOND MOUTH MOVEMENTS
For EVERY second of the video, describe:
- Mouth position (open, closed, slightly open, wide open)
- Lip movements (forming specific sounds, smiling, frowning, pursed)
- Tongue visibility
- Teeth showing or not
- Jaw position
- Speaking or silent

Format: [MM:SS] Mouth: [detailed description]

## PART 3: SECOND-BY-SECOND EYE TRACKING
For EVERY second, track:
- Eye direction (looking at camera, looking left/right/up/down, looking away)
- Pupil size if visible
- Blinking (note every blink with timestamp)
- Eye movements (rapid, slow, steady, darting)
- Eyelid position (wide open, squinting, half-closed)
- Eye contact duration

Format: [MM:SS] Eyes: [detailed description]

## PART 4: SECOND-BY-SECOND HEAD MOVEMENTS
For EVERY second, document:
- Head position (straight, tilted left/right, turned left/right, up/down)
- Head movements (nodding, shaking, tilting)
- Speed of movement (rapid, slow, steady)
- Angle relative to camera
- Any rotation or repositioning

Format: [MM:SS] Head: [detailed description]

## PART 5: SECOND-BY-SECOND BODY MOVEMENTS
For EVERY second, track:
- Body posture (upright, slouched, leaning)
- Shoulder position and movements
- Arm positions and gestures
- Hand movements and positions
- Torso movements
- Overall body language
- Distance from camera

Format: [MM:SS] Body: [detailed description]

## PART 6: FACIAL EXPRESSIONS (DETAILED)
Track ALL facial expression changes with timestamps:
- Eyebrow position (raised, furrowed, neutral, one raised)
- Forehead (smooth, wrinkled, tense)
- Nose (flared, scrunched, neutral)
- Cheeks (raised, neutral, hollow)
- Overall expression (happy, sad, angry, surprised, disgusted, fearful, neutral, confused, etc.)
- Micro-expressions (brief flashes of emotion)
- Expression intensity (subtle, moderate, strong)

Format: [MM:SS] Expression: [detailed description]

## PART 7: SYNCHRONIZED TIMELINE
Create a second-by-second timeline that combines ALL above elements:

[00:00]
- Speech: [what's said]
- Mouth: [position/movement]
- Eyes: [direction/state]
- Head: [position/movement]
- Body: [posture/movement]
- Expression: [current emotion]

[00:01]
- Speech: [what's said]
- Mouth: [position/movement]
- Eyes: [direction/state]
- Head: [position/movement]
- Body: [posture/movement]
- Expression: [current emotion]

...continue for EVERY second of the video...

## PART 8: SCENE & ENVIRONMENT
- Video duration and total frame count
- Resolution and quality
- Lighting conditions (natural, artificial, direction, intensity, color temperature)
- Background elements (objects, walls, furniture, decorations)
- Foreground elements
- Camera angle (straight on, from above, from below, side angle)
- Camera movement (static, panning, zooming, shaking)
- Depth of field
- Overall composition

## PART 9: AUDIO ANALYSIS
- Audio quality and clarity
- Background noise or music
- Acoustic environment (echo, reverb, room size)
- Audio levels throughout
- Any non-speech sounds with timestamps
- Silence periods with timestamps

## PART 10: METADATA & SUMMARY
- Total video duration
- Number of words spoken
- Number of distinct actions/movements
- Most prominent emotions
- Key moments with timestamps
- Overall narrative or purpose of the video

BE EXTREMELY THOROUGH. Track EVERYTHING. Don't miss any detail. Analyze every single second."""

        print_step("Performing ultra-detailed frame-by-frame analysis...")
        print_info("This may take 1-2 minutes for comprehensive extraction...")

        response = self.vision_model.generate_content([analysis_prompt, video_file])
        analysis_text = response.text

        # Save analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = self.output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        analysis_file = analysis_dir / f"analysis_{timestamp}.txt"
        with open(analysis_file, 'w') as f:
            f.write(analysis_text)

        # Also save as JSON for easier parsing
        analysis_data = {
            "video_path": str(video_path),
            "timestamp": timestamp,
            "analysis": analysis_text,
            "length": len(analysis_text)
        }

        json_file = analysis_dir / f"analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        print_header("STEP 4 COMPLETE: VIDEO ANALYSIS")
        print_success("Video analysis completed")
        print_info(f"Output: {analysis_file}")
        print_info(f"Length: {len(analysis_text)} characters")

        return {
            "success": True,
            "analysis": analysis_text,
            "analysis_file": str(analysis_file),
            "json_file": str(json_file)
        }

    # ================================================================================
    # STEP 5: VIDEO GENERATION (Using analysis + transformed image)
    # ================================================================================

    def generate_video(self, original_video: str, generated_image: str,
                      video_analysis: str, action_prompt: str) -> Dict:
        """
        Generate final video using Seedance V1 Pro Fast IMAGE-TO-VIDEO API:
        - Uploads the generated image to a temporary storage
        - Sends image URL and detailed prompt to Seedance
        - Creates cinematic video with timing and movements from analysis
        - Polls until video is ready
        """
        print_header("STEP 5: VIDEO GENERATION (SEEDANCE IMAGE-TO-VIDEO)")

        import requests
        import base64

        # Get API key
        seedance_api_key = os.getenv("EACHLABS_API_KEY")
        if not seedance_api_key:
            print_error("EACHLABS_API_KEY not found in environment")
            return {
                "success": False,
                "error": "EACHLABS_API_KEY not set"
            }

        print_step("Preparing video generation prompt...")

        # Extract key timing info from analysis
        # Parse duration from metadata
        duration = 12  # Default to 12 seconds for SeedDance
        try:
            if "Total video duration: Approximately" in video_analysis:
                duration_str = video_analysis.split("Total video duration: Approximately")[1].split("seconds")[0].strip()
                duration = int(float(duration_str))
                duration = min(max(duration, 5), 12)  # Clamp between 5-12 seconds
        except:
            pass

        # Extract transcript from analysis
        transcript = self._extract_transcript(video_analysis)

        # Build comprehensive video generation prompt
        generation_prompt = f"""{action_prompt}

SPEECH TRANSCRIPT (THE PERSON MUST SAY THESE EXACT WORDS):
{transcript}

MOTION & TIMING (from original video):
{self._extract_key_movements(video_analysis)}

VIDEO REQUIREMENTS:
- Duration: {duration} seconds
- Style: Cinematic, photorealistic, high-fidelity
- Motion: Smooth, natural movements matching original timing
- Camera: {self._extract_camera_info(video_analysis)} - Static camera
- Lighting: Realistic lighting matching the reference image
- Audio: The person must speak the transcript above with natural lip sync and voice

Animate this image with natural, lifelike motion. THE PERSON MUST SAY THE EXACT WORDS FROM THE TRANSCRIPT. Maintain the exact appearance from the image while adding subtle movements, facial expressions, natural lip movements that match the speech, and the actions described above. Ensure photorealistic quality with smooth, natural motion and accurate lip synchronization."""

        print_info(f"Video duration: {duration} seconds")
        print_info("Resolution: 1080p, Aspect ratio: 16:9")

        # Upload image to Backblaze B2 (S3-compatible storage)
        print_step("Uploading generated image to Backblaze B2...")

        # Use urllib3 for B2 API calls to avoid connection leaks
        import urllib3
        import base64 as b64
        
        # Create pool manager with longer timeout for file uploads
        b2_http = urllib3.PoolManager(
            num_pools=1,
            maxsize=1,
            timeout=urllib3.Timeout(connect=30.0, read=120.0),  # Longer read timeout for uploads
            retries=urllib3.Retry(total=2, connect=2, read=2, backoff_factor=1)
        )
        
        try:
            # Get B2 credentials from environment
            b2_key_id = os.getenv("keyID")
            b2_key_name = os.getenv("keyName")
            b2_app_key = os.getenv("applicationKey")

            if not all([b2_key_id, b2_key_name, b2_app_key]):
                raise ValueError("B2 credentials not found in .env file")

            # Step 1: Authorize with B2
            print_detail("Authorizing with Backblaze B2...")
            auth_url = "https://api.backblazeb2.com/b2api/v2/b2_authorize_account"
            
            # Create basic auth header
            auth_string = f"{b2_key_id}:{b2_app_key}"
            auth_bytes = auth_string.encode('utf-8')
            auth_b64 = b64.b64encode(auth_bytes).decode('utf-8')
            
            auth_response = b2_http.request(
                'GET',
                auth_url,
                headers={
                    'Authorization': f'Basic {auth_b64}',
                    'Connection': 'close'
                }
            )
            
            if auth_response.status != 200:
                raise Exception(f"B2 auth failed with status {auth_response.status}")
            
            import json as json_mod
            auth_data = json_mod.loads(auth_response.data.decode('utf-8'))

            api_url_b2 = auth_data['apiUrl']
            auth_token = auth_data['authorizationToken']
            download_url = auth_data['downloadUrl']

            print_detail("B2 authorization successful")

            # Step 2: List buckets to get the bucket ID
            print_detail("Listing B2 buckets...")
            buckets_payload = json_mod.dumps({"accountId": auth_data['accountId']}).encode('utf-8')
            
            buckets_response = b2_http.request(
                'POST',
                f"{api_url_b2}/b2api/v2/b2_list_buckets",
                headers={
                    "Authorization": auth_token,
                    "Content-Type": "application/json",
                    "Connection": "close"
                },
                body=buckets_payload
            )
            
            if buckets_response.status != 200:
                raise Exception(f"B2 list buckets failed with status {buckets_response.status}")
            
            buckets_data = json_mod.loads(buckets_response.data.decode('utf-8'))

            # Find bucket by name
            bucket_id = None
            for bucket in buckets_data.get('buckets', []):
                if bucket['bucketName'] == b2_key_name:
                    bucket_id = bucket['bucketId']
                    break

            if not bucket_id:
                # Use first bucket if named bucket not found
                if buckets_data.get('buckets'):
                    bucket_id = buckets_data['buckets'][0]['bucketId']
                    b2_key_name = buckets_data['buckets'][0]['bucketName']
                    print_detail(f"Using bucket: {b2_key_name}")
                else:
                    raise ValueError("No buckets found in account")

            # Step 3: Get upload URL
            print_detail("Getting B2 upload URL...")
            upload_url_payload = json_mod.dumps({"bucketId": bucket_id}).encode('utf-8')
            
            upload_url_response = b2_http.request(
                'POST',
                f"{api_url_b2}/b2api/v2/b2_get_upload_url",
                headers={
                    "Authorization": auth_token,
                    "Content-Type": "application/json",
                    "Connection": "close"
                },
                body=upload_url_payload
            )
            
            if upload_url_response.status != 200:
                raise Exception(f"B2 get upload URL failed with status {upload_url_response.status}")
            
            upload_data = json_mod.loads(upload_url_response.data.decode('utf-8'))
            upload_url_b2 = upload_data['uploadUrl']
            upload_auth_token = upload_data['authorizationToken']

            # Step 4: Upload the file
            print_detail("Uploading image file to B2...")
            import hashlib
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"video_transformer/{timestamp}_generated.jpg"

            with open(generated_image, 'rb') as f:
                file_content = f.read()
                file_sha1 = hashlib.sha1(file_content).hexdigest()
                file_size = len(file_content)
                
                print_detail(f"File size: {file_size / 1024:.1f} KB")

                upload_headers = {
                    "Authorization": upload_auth_token,
                    "X-Bz-File-Name": file_name,
                    "Content-Type": "image/jpeg",
                    "X-Bz-Content-Sha1": file_sha1,
                    "Content-Length": str(file_size),
                    "Connection": "close"
                }

                upload_response = b2_http.request(
                    'POST',
                    upload_url_b2,
                    headers=upload_headers,
                    body=file_content
                )
                
                if upload_response.status != 200:
                    raise Exception(f"B2 upload failed with status {upload_response.status}: {upload_response.data.decode('utf-8')}")
                
                upload_result = json_mod.loads(upload_response.data.decode('utf-8'))

            # Construct public URL
            file_id = upload_result['fileId']
            bucket_name = b2_key_name  # Using keyName as bucket name
            image_url = f"{download_url}/file/{bucket_name}/{file_name}"

            print_success(f"Image uploaded successfully to Backblaze B2")
            print_detail(f"Image URL: {image_url}")
            print_detail(f"File ID: {file_id}")

        except Exception as e:
            print_error(f"B2 upload failed: {str(e)}")
            import traceback
            print_detail(traceback.format_exc())

            # CRITICAL ERROR: Don't use placeholder, raise the error
            raise Exception(f"Failed to upload image to B2: {str(e)}. Cannot proceed with video generation without the generated image.")
            
        finally:
            # Clean up B2 connection pool
            try:
                b2_http.clear()
                del b2_http
                gc.collect()
                print_detail("B2 connection pool cleaned up")
            except Exception as e:
                print_detail(f"Error cleaning up B2 pool: {e}")

        # Prepare API request for IMAGE-TO-VIDEO
        api_url = "https://api.eachlabs.ai/v1/prediction/"
        headers = {
            "X-API-Key": seedance_api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "model": "seedance-v1-pro-fast-image-to-video",  # Changed to image-to-video
            "version": "0.0.1",
            "input": {
                "prompt": generation_prompt,
                "image_url": image_url,  # Added image URL
                "aspect_ratio": "9:16",  # Vertical/portrait format
                "resolution": "1080p",
                "duration": "12"  # Hardcoded to 12 seconds as string
            },
            "webhook_url": ""
        }

        print_step("Sending request to Seedance V1 Pro Fast API...")
        print_detail(f"Prompt length: {len(generation_prompt)} characters")

        # Use session for proper connection management
        with requests.Session() as session:
            try:
                # Create prediction
                response = session.post(api_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()

                prediction_id = result.get("predictionID") or result.get("id")
                if not prediction_id:
                    raise ValueError(f"No prediction ID returned. Response: {result}")

                print_success(f"Video generation started! Prediction ID: {prediction_id}")
                print_info("Estimated time: ~2 minutes")

                # Poll for result
                print_step("Polling for video result...")
                poll_url = f"https://api.eachlabs.ai/v1/prediction/{prediction_id}"

                max_attempts = 180  # 3 minutes with 1-second intervals
                attempt = 0

                while attempt < max_attempts:
                    time.sleep(1)
                    attempt += 1

                    poll_response = session.get(poll_url, headers=headers, timeout=10)
                    poll_response.raise_for_status()
                    poll_result = poll_response.json()

                    status = poll_result.get("status", "unknown")

                    if attempt % 10 == 0:  # Update every 10 seconds
                        print_detail(f"Status: {status} ({attempt}s elapsed)")

                    if status == "succeeded" or status == "success":
                        print_success("Video generation complete!")

                        # Get output video URL - it's directly in "output" field as string
                        video_url = poll_result.get("output", "")

                        if not video_url or video_url == "":
                            raise ValueError(f"No video URL in response. Output: {poll_result.get('output')}")

                        # Download the video
                        print_step("Downloading generated video...")
                        video_response = session.get(video_url, timeout=60)
                        video_response.raise_for_status()

                        # Save video
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        video_dir = self.output_dir / "videos"
                        video_dir.mkdir(parents=True, exist_ok=True)

                        video_path = video_dir / f"generated_video_{timestamp}.mp4"
                        with open(video_path, 'wb') as f:
                            f.write(video_response.content)

                        video_size_mb = len(video_response.content) / (1024 * 1024)
                        print_success(f"Video saved: {video_path.name}")
                        print_info(f"Size: {video_size_mb:.2f} MB")

                        # Save metadata
                        metadata = {
                            "prediction_id": prediction_id,
                            "timestamp": timestamp,
                            "video_path": str(video_path),
                            "video_url": video_url,
                            "duration": duration,
                            "resolution": "1080p",
                            "aspect_ratio": "16:9",
                            "prompt": generation_prompt,
                            "generated_image": generated_image,
                            "original_video": original_video,
                            "generation_time_seconds": attempt
                        }

                        metadata_file = video_dir / f"metadata_{timestamp}.json"
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)

                        print_header("STEP 5 COMPLETE: VIDEO GENERATED")
                        print_success(f"Generated video: {video_path}")
                        print_info(f"Duration: {duration}s, Resolution: 1080p")
                        print_info(f"Generation time: {attempt} seconds")

                        return {
                            "success": True,
                            "video_path": str(video_path),
                            "video_url": video_url,
                            "prediction_id": prediction_id,
                            "metadata_file": str(metadata_file),
                            "generation_time": attempt,
                            "size_mb": video_size_mb
                        }

                    elif status == "failed":
                        error_msg = poll_result.get("error", "Unknown error")
                        raise ValueError(f"Video generation failed: {error_msg}")

                raise TimeoutError(f"Video generation timed out after {max_attempts} seconds")

            except Exception as e:
                print_error(f"Video generation error: {str(e)}")
                return {
                    "success": False,
                    "error": str(e)
                }

    def _extract_transcript(self, analysis: str) -> str:
        """Extract the full transcript from video analysis"""
        # Look for the FULL TRANSCRIPT section
        if "**FULL TRANSCRIPT" in analysis or "FULL TRANSCRIPT:" in analysis:
            try:
                # Try to extract the full transcript section
                if "**FULL TRANSCRIPT" in analysis:
                    transcript_section = analysis.split("**FULL TRANSCRIPT")[1]
                else:
                    transcript_section = analysis.split("FULL TRANSCRIPT:")[1]

                # Get text until next section or double newline
                transcript = transcript_section.split("##")[0].split("\n\n")[0].strip()
                # Remove markdown formatting and colons
                transcript = transcript.replace("**", "").replace(":", "").strip()
                if transcript:
                    return transcript
            except:
                pass

        # Fallback: extract from timestamped speech lines
        speech_lines = []
        for line in analysis.split('\n'):
            if line.strip().startswith('[') and '"' in line:
                try:
                    # Extract text between quotes
                    text = line.split('"')[1]
                    if text and text not in ['pause', '[pause']:
                        speech_lines.append(text)
                except:
                    pass

        if speech_lines:
            return ' '.join(speech_lines)

        return "Speak naturally and expressively"

    def _extract_camera_info(self, analysis: str) -> str:
        """Extract camera angle from analysis"""
        if "Camera angle" in analysis:
            try:
                camera_line = [line for line in analysis.split('\n') if 'Camera angle' in line][0]
                return camera_line.split('Camera angle:')[1].split('\n')[0].strip()
            except:
                pass
        return "straight on, medium close-up"

    def _extract_key_movements(self, analysis: str) -> str:
        """Extract key movements and timing from analysis"""
        movements = []
        for line in analysis.split('\n'):
            if line.strip().startswith('[') and ('Speech:' in line or 'Eyes:' in line or 'Head:' in line):
                movements.append(line.strip())
                if len(movements) >= 10:  # Limit to first 10 key moments
                    break
        return '\n'.join(movements) if movements else "Maintain natural, subtle movements"

    def _extract_scene_info(self, analysis: str) -> str:
        """Extract scene description from analysis"""
        if "## PART 8: SCENE & ENVIRONMENT" in analysis:
            try:
                scene_section = analysis.split("## PART 8: SCENE & ENVIRONMENT")[1].split("## PART 9")[0]
                return scene_section.strip()[:500]  # Limit length
            except:
                pass
        return "Indoor scene with natural lighting"

    # ================================================================================
    # MAIN PIPELINE
    # ================================================================================

    def transform_video(self, video_path: str, prompt: str,
                       skip_analysis: bool = False,
                       skip_video_gen: bool = True) -> Dict:
        """Run the complete transformation pipeline"""
        print_header("AI VIDEO TRANSFORMER - FULL PIPELINE")
        print_info(f"Video: {video_path}")
        print_info(f"Prompt: {prompt}")
        print_info(f"Output: {self.output_dir}")

        results = {}

        try:
            # Step 1: Extract frames
            step1_result = self.extract_frames(video_path)
            results['step1'] = step1_result

            # Step 2: Enhance prompt (ACTION-FOCUSED)
            step2_result = self.enhance_prompt(
                step1_result['frames_dir'],
                prompt
            )
            results['step2'] = step2_result

            # Step 3: Generate image
            step3_result = self.generate_image(
                step1_result['frames_dir'],
                step2_result['action_prompt']
            )
            results['step3'] = step3_result

            # Step 4: Video analysis
            if not skip_analysis:
                step4_result = self.analyze_video(video_path)
                results['step4'] = step4_result

                # Step 5: Generate video (using analysis)
                if not skip_video_gen:
                    step5_result = self.generate_video(
                        video_path,
                        step3_result['generated_image'],
                        step4_result['analysis'],
                        step2_result['action_prompt']
                    )
                    results['step5'] = step5_result

            # Save final results
            final_results = {
                "video_path": video_path,
                "user_prompt": prompt,
                "action_prompt": step2_result['action_prompt'],
                "generated_image": step3_result['generated_image'],
                "frames_dir": step1_result['frames_dir'],
                "image_dir": step3_result['image_dir'],
                "timestamp": datetime.now().isoformat()
            }

            if not skip_analysis:
                final_results['analysis_file'] = results['step4']['analysis_file']
                if not skip_video_gen and 'step5' in results:
                    final_results['video_generation'] = results['step5']

            results_file = self.output_dir / "results.json"
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2)

            print_header("🎉 PIPELINE COMPLETE!")
            print_success(f"Generated image: {step3_result['generated_image']}")
            if not skip_analysis:
                print_success(f"Video analysis: {results['step4']['analysis_file']}")
            print_success(f"Results saved: {results_file}")

            return {
                "success": True,
                "results": final_results,
                "results_file": str(results_file)
            }

        except Exception as e:
            print_error(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "results": results
            }


def main():
    parser = argparse.ArgumentParser(
        description="AI Video Transformer - Transform videos with AI",
        epilog="Example: python video_transformer.py video.mp4 'person wearing sunglasses'"
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("prompt", help="Transformation prompt (describe what you want to see)")
    parser.add_argument("--output", "-o", default="output", help="Output directory (default: output)")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip video analysis (faster)")
    parser.add_argument("--generate-video", action="store_true", help="Attempt video generation (requires Veo)")
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--frames", type=int, default=7, help="Number of frames to extract (default: 7)")

    args = parser.parse_args()

    # Initialize transformer
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print_error("Error: GEMINI_API_KEY not found")
        print_info("Set it via: export GEMINI_API_KEY=your_api_key")
        print_info("Or use: --api-key YOUR_API_KEY")
        sys.exit(1)

    # Use context manager for proper resource cleanup
    with VideoTransformer(api_key=api_key, output_dir=args.output) as transformer:
        # Run pipeline
        result = transformer.transform_video(
            args.video,
            args.prompt,
            skip_analysis=args.skip_analysis,
            skip_video_gen=not args.generate_video
        )

    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
