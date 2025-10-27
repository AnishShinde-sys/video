#!/usr/bin/env python3
"""
STEP 2: Prompt Enhancement with Gemini
- Takes user prompt and first frame as reference
- Sends to Gemini via OpenRouter to enhance the prompt
- Returns detailed instructions on what to change
- Full logging of inputs and outputs
"""

import os
from PIL import Image
import json
import base64
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import ssl
import certifi

# Colors for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RED = '\033[91m'
CYAN = '\033[96m'
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

class PromptEnhancer:
    def __init__(self, api_key: str):
        """Initialize the prompt enhancer with OpenRouter API key"""
        self.api_key = api_key
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        # Get referer from environment or use default
        app_host = os.getenv("APP_HOST", "localhost")
        app_port = os.getenv("APP_PORT", "5001")
        app_referer = os.getenv("APP_REFERER", f"http://{app_host}:{app_port}")

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": app_referer
        }

        # Create a session with retry strategy to handle connection errors
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"]
        )
        # Reduce connection pool size to avoid "too many open files" error
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=2, pool_maxsize=2)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def __del__(self):
        """Cleanup session when object is destroyed"""
        if hasattr(self, 'session'):
            self.session.close()

    def close(self):
        """Explicitly close the session"""
        if hasattr(self, 'session'):
            self.session.close()

    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode an image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def enhance_prompt(self, user_prompt: str, reference_frame_path: str, output_dir: str = "temp/step2_prompt") -> Dict:
        """
        Enhance the user prompt using Gemini with the first frame as reference

        Args:
            user_prompt: The original user prompt (e.g., "make him wear a suit")
            reference_frame_path: Path to the first frame to use as reference
            output_dir: Directory to save logs and outputs

        Returns:
            Dictionary with enhanced prompt and analysis
        """

        print_header("STEP 2: PROMPT ENHANCEMENT WITH GEMINI")

        # Create output directory
        output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        print_step(f"Output directory: {run_dir}")

        # Load and encode the reference frame
        print_step("Loading reference frame...")

        if not Path(reference_frame_path).exists():
            print_error(f"Reference frame not found: {reference_frame_path}")
            return {"success": False, "error": "Reference frame not found"}

        # Read image info using PIL (with proper closing)
        with Image.open(reference_frame_path) as img:
            width, height = img.size
        file_size = Path(reference_frame_path).stat().st_size / 1024  # KB

        print_info(f"Frame path: {reference_frame_path}")
        print_info(f"Resolution: {width}x{height}")
        print_info(f"Size: {file_size:.1f} KB")

        # Encode image to base64
        print_step("Encoding image to base64...")
        img_base64 = self.encode_image_to_base64(reference_frame_path)
        print_success(f"Encoded image: {len(img_base64)} characters")

        # Create the prompt for Gemini
        system_prompt = """You are an expert AI prompt engineer specializing in video and image editing instructions.
Your task is to take a simple user request and a reference frame, then create a detailed, specific prompt that describes exactly what changes need to be made.

Analyze the reference image carefully and provide:
1. A detailed description of what you see in the image
2. A specific, detailed prompt for making the requested changes
3. Technical details about what needs to be modified (colors, positions, styles, etc.)
4. Any challenges or considerations for the edit

Be very specific about colors, positions, styles, and exact modifications needed."""

        user_message = f"""User's request: "{user_prompt}"

Based on this reference frame, create a detailed prompt that explains exactly how to fulfill this request.
Consider the current state of the image and what specific changes are needed.

Provide your response in this JSON format:
{{
    "image_analysis": "Detailed description of what you see in the reference frame",
    "enhanced_prompt": "A detailed, specific prompt for making the requested changes",
    "technical_details": {{
        "current_state": "Description of relevant current elements",
        "required_changes": ["List of specific changes needed"],
        "color_specifications": "Any specific colors or styles to apply",
        "positioning": "Where and how to apply changes"
    }},
    "considerations": ["Any challenges or important notes"]
}}"""

        # Prepare API request
        print_step("Preparing API request to Gemini...")

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ]

        api_payload = {
            "model": "google/gemini-2.5-flash-image",  # Using Gemini 2.5 Flash for images
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }

        # Log the request
        print_step("Sending request to Gemini via OpenRouter...")
        print_info(f"Model: {api_payload['model']}")
        print_info(f"User prompt: \"{user_prompt}\"")
        print_info(f"Image included: Yes ({file_size:.1f} KB)")

        # Save request log
        request_log = {
            "timestamp": timestamp,
            "user_prompt": user_prompt,
            "reference_frame": reference_frame_path,
            "model": api_payload["model"],
            "image_size": f"{width}x{height}",
            "image_size_kb": file_size
        }

        with open(run_dir / "request_log.json", 'w') as f:
            json.dump(request_log, f, indent=2)

        # Make API request with session
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=api_payload,
                timeout=30,
                verify=certifi.where()  # Use certifi for SSL verification
            )

            print_info(f"Response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                # Extract the response
                gemini_response = data["choices"][0]["message"]["content"]

                # Try to parse as JSON
                try:
                    enhanced_data = json.loads(gemini_response)
                except:
                    # If not valid JSON, create structure
                    enhanced_data = {
                        "enhanced_prompt": gemini_response,
                        "raw_response": True
                    }

                print_success("Received enhanced prompt from Gemini")

                # Display the enhancement
                print_header("ENHANCEMENT RESULTS")

                if "image_analysis" in enhanced_data:
                    print_step("Image Analysis:")
                    print_info(enhanced_data["image_analysis"][:200] + "..." if len(enhanced_data.get("image_analysis", "")) > 200 else enhanced_data.get("image_analysis", ""))

                if "enhanced_prompt" in enhanced_data:
                    print_step("Enhanced Prompt:")
                    enhanced_prompt = enhanced_data["enhanced_prompt"]
                    print(f"{GREEN}{enhanced_prompt[:300]}...{RESET}" if len(enhanced_prompt) > 300 else f"{GREEN}{enhanced_prompt}{RESET}")

                if "technical_details" in enhanced_data:
                    print_step("Technical Details:")
                    tech = enhanced_data["technical_details"]
                    if "required_changes" in tech:
                        print_info(f"Changes needed: {len(tech['required_changes'])} modifications")
                        for change in tech["required_changes"][:3]:
                            print_info(f"  • {change}")

                # Save the response
                response_file = run_dir / "enhanced_prompt.json"
                with open(response_file, 'w') as f:
                    json.dump(enhanced_data, f, indent=2)
                print_success(f"Saved enhanced prompt to: {response_file}")

                # Save a simple text version
                text_file = run_dir / "enhanced_prompt.txt"
                with open(text_file, 'w') as f:
                    f.write(f"Original Prompt: {user_prompt}\n")
                    f.write(f"{'='*50}\n\n")
                    f.write(f"Enhanced Prompt:\n{enhanced_data.get('enhanced_prompt', gemini_response)}\n")

                # Copy reference frame to output dir for reference
                import shutil
                ref_copy = run_dir / "reference_frame.jpg"
                shutil.copy(reference_frame_path, ref_copy)
                print_success(f"Copied reference frame to: {ref_copy}")

                print_header("STEP 2 COMPLETE: PROMPT ENHANCEMENT SUMMARY")
                print_success("Successfully enhanced prompt using Gemini")
                print_info(f"Original prompt: \"{user_prompt}\"")
                print_info(f"Enhanced prompt length: {len(enhanced_data.get('enhanced_prompt', ''))} characters")
                print_info(f"Output directory: {run_dir}")

                return {
                    "success": True,
                    "original_prompt": user_prompt,
                    "enhanced_prompt": enhanced_data.get("enhanced_prompt", gemini_response),
                    "analysis": enhanced_data,
                    "output_dir": str(run_dir),
                    "model": api_payload["model"]
                }

            else:
                error_msg = f"API error: {response.status_code}"
                print_error(error_msg)
                print_error(f"Response: {response.text[:500]}")

                # Save error log
                with open(run_dir / "error_log.json", 'w') as f:
                    json.dump({
                        "error": error_msg,
                        "status_code": response.status_code,
                        "response": response.text
                    }, f, indent=2)

                return {
                    "success": False,
                    "error": error_msg
                }

        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            print_error(error_msg)

            # Save error log
            with open(run_dir / "error_log.json", 'w') as f:
                json.dump({
                    "error": error_msg,
                    "exception": str(e)
                }, f, indent=2)

            return {
                "success": False,
                "error": error_msg
            }


if __name__ == "__main__":
    # Test with the first frame from Step 1

    # Get the API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        # Try to read from parent .env file
        env_path = Path("../.env")
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("OPENROUTER_API_KEY="):
                        api_key = line.split("=")[1].strip()
                        break

    if not api_key:
        print_error("No OpenRouter API key found!")
        print_info("Set OPENROUTER_API_KEY environment variable or add to .env file")
    else:
        # Find the most recent frame extraction
        frames_dir = Path("temp/step1_frames")
        if frames_dir.exists():
            # Get the most recent run
            runs = sorted([d for d in frames_dir.iterdir() if d.is_dir()],
                         key=lambda x: x.name, reverse=True)

            if runs:
                latest_run = runs[0]
                # Get the first frame
                first_frame = latest_run / "frame_01_start_0000.jpg"

                if first_frame.exists():
                    print_success(f"Found reference frame: {first_frame}")

                    # Test prompt enhancement
                    enhancer = PromptEnhancer(api_key)

                    # Example user prompt
                    user_prompt = "make him wear a professional business suit with a blue tie"

                    result = enhancer.enhance_prompt(
                        user_prompt=user_prompt,
                        reference_frame_path=str(first_frame)
                    )

                    if result["success"]:
                        print(f"\n{GREEN}Step 2 completed successfully!{RESET}")
                    else:
                        print(f"\n{RED}Step 2 failed: {result.get('error')}{RESET}")
                else:
                    print_error(f"First frame not found: {first_frame}")
            else:
                print_error("No frame extraction runs found. Run step1_frame_extraction.py first!")
        else:
            print_error("No frames directory found. Run step1_frame_extraction.py first!")