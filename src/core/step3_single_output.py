#!/usr/bin/env python3
"""
STEP 3: NanoBana - Send ALL frames, get ONE output image
- Takes ALL frames from Step 1 as input
- Takes enhanced prompt from Step 2
- Sends everything to Gemini to generate ONE final output image
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Add path for google-genai if needed
sys.path.append('/Users/anishshinde/Library/Python/3.9/lib/python/site-packages')

# Load environment variables
load_dotenv()

from google import genai
from google.genai import types

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


class NanoBanaSingleOutput:
    def __init__(self, api_key: str = None):
        """Initialize NanoBana for single output generation"""
        # Use provided API key or get from environment
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not provided or found in environment")

        self.client = genai.Client(api_key=api_key)
        print_success("Initialized NanoBana for SINGLE OUTPUT generation")

    def generate_single_output(self,
                              frames_dir: str,
                              enhanced_prompt: str,
                              output_dir: str = "temp/step3_single") -> dict:
        """
        Send ALL frames as input and get ONE output image

        Args:
            frames_dir: Directory with ALL input frames
            enhanced_prompt: Enhanced prompt from Step 2
            output_dir: Where to save the single output image
        """

        print_header("STEP 3: NANOBANA - ALL FRAMES → ONE OUTPUT")

        # Create output directory
        output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        print_step(f"Output directory: {run_dir}")

        # Load ALL frames as input
        print_step("Loading ALL frames as INPUT...")
        frames_dir = Path(frames_dir)
        frame_files = sorted([f for f in frames_dir.glob("frame_*.jpg")])

        if not frame_files:
            print_error("No frames found!")
            return {"success": False, "error": "No frames found"}

        print_success(f"Found {len(frame_files)} frames to use as INPUT")

        # Load all frames
        input_frames = []
        total_size = 0
        for i, frame_path in enumerate(frame_files):
            img = Image.open(frame_path)
            input_frames.append(img)
            size_kb = frame_path.stat().st_size / 1024
            total_size += size_kb
            print_info(f"Input frame {i+1}: {frame_path.name} ({img.size[0]}x{img.size[1]}, {size_kb:.1f} KB)")

        print_success(f"Loaded {len(input_frames)} frames as input")
        print_info(f"Total input data: {total_size:.1f} KB")

        # Create the generation prompt
        generation_prompt = f"""You are NanoBana, an advanced AI image processor.

        I'm providing you with {len(input_frames)} frames from a video as INPUT.

        Your task: Generate ONE SINGLE OUTPUT IMAGE based on these inputs and this request:
        {enhanced_prompt}

        IMPORTANT:
        - Analyze ALL the input frames to understand the person, movement, and context
        - Generate ONE SINGLE BEST OUTPUT IMAGE that fulfills the request
        - The output should be the best representation of the transformation requested
        - Maintain photorealistic quality
        - This is NOT frame-by-frame editing - create ONE perfect result

        Generate a single high-quality image that best represents the requested transformation."""

        try:
            print_step(f"Sending {len(input_frames)} frames to NanoBana...")
            print_info("Requesting ONE OUTPUT IMAGE...")

            # Build the content list - prompt + all frames
            contents = [generation_prompt]
            contents.extend(input_frames)

            # Send to Gemini for single image generation
            print_info("Processing with Gemini 2.5 Flash Image...")

            response = self.client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=contents
            )

            # Extract the single output image
            output_image = None
            response_text = None

            for part in response.candidates[0].content.parts:
                # Check for text response
                if part.text is not None:
                    response_text = part.text
                    print_detail(f"Response: {part.text[:200]}...")

                # Check for generated image
                elif part.inline_data is not None:
                    print_success("🎨 RECEIVED OUTPUT IMAGE FROM NANOBANA!")

                    # Extract the single output image
                    image_data = part.inline_data.data
                    output_image = Image.open(BytesIO(image_data))

                    # Save the output image
                    output_path = run_dir / "nanobana_output.jpg"
                    output_image.save(output_path, quality=95)
                    print_success(f"✅ Saved output to: {output_path}")

                    # Save image info
                    image_info = {
                        "size": output_image.size,
                        "mode": output_image.mode,
                        "format": output_image.format,
                        "path": str(output_path)
                    }
                    break

            if output_image:
                # Create a comparison grid showing input frames and output
                print_step("Creating input/output comparison...")

                comparison = create_comparison(input_frames[:3], output_image, run_dir)
                if comparison:
                    print_success(f"Created comparison: {comparison}")

                # Save processing log
                log_data = {
                    "timestamp": timestamp,
                    "model": "gemini-2.5-flash-image",
                    "input_frames_count": len(input_frames),
                    "input_frames": [f.name for f in frame_files],
                    "total_input_size_kb": total_size,
                    "enhanced_prompt": enhanced_prompt,
                    "output_generated": True,
                    "output_image": image_info if output_image else None,
                    "response_text": response_text
                }

                log_file = run_dir / "processing_log.json"
                with open(log_file, 'w') as f:
                    json.dump(log_data, f, indent=2)
                print_success(f"Saved log to: {log_file}")

                print_header("✨ SUCCESS! ✨")
                print_success(f"Generated ONE OUTPUT IMAGE from {len(input_frames)} input frames!")
                print_info(f"Output image: {output_path}")
                print_info(f"Output size: {output_image.size}")

                return {
                    "success": True,
                    "input_frames": len(input_frames),
                    "output_path": str(output_path),
                    "output_size": output_image.size,
                    "run_dir": str(run_dir),
                    "output_dir": str(run_dir)  # For compatibility with app.py
                }

            else:
                print_error("No output image generated - only received text response")

                # Save the text response
                if response_text:
                    text_file = run_dir / "text_response.txt"
                    with open(text_file, 'w') as f:
                        f.write(response_text)
                    print_info(f"Saved text response to: {text_file}")

                return {
                    "success": False,
                    "error": "No image generated",
                    "response_text": response_text
                }

        except Exception as e:
            print_error(f"Processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def generate_output(self, frames_dir=None, enhanced_prompt=None):
        """
        Wrapper method for compatibility with app.py
        Can auto-discover inputs or accept them as parameters
        """
        # If parameters not provided, try to auto-discover from filesystem
        if not frames_dir or not enhanced_prompt:
            try:
                # Find latest frames if not provided
                if not frames_dir:
                    frames_base = Path("temp/step1_frames")
                    if frames_base.exists():
                        latest_frames = sorted(frames_base.iterdir(), key=lambda x: x.name)[-1]
                        frames_dir = str(latest_frames)
                    else:
                        return {"success": False, "error": "No frames directory found"}

                # Find latest prompt if not provided
                if not enhanced_prompt:
                    prompt_base = Path("temp/step2_prompt")
                    if prompt_base.exists():
                        latest_prompt_dir = sorted(prompt_base.iterdir(), key=lambda x: x.name)[-1]
                        prompt_file = latest_prompt_dir / "enhanced_prompt.json"
                        if prompt_file.exists():
                            with open(prompt_file) as f:
                                data = json.load(f)
                                enhanced_prompt_raw = data.get("enhanced_prompt", "")

                                # Parse if needed
                                if "```json" in enhanced_prompt_raw:
                                    import re
                                    match = re.search(r'"enhanced_prompt":\s*"([^"]+)"', enhanced_prompt_raw)
                                    if match:
                                        enhanced_prompt = match.group(1)
                                    else:
                                        enhanced_prompt = enhanced_prompt_raw
                                else:
                                    enhanced_prompt = enhanced_prompt_raw
                        else:
                            return {"success": False, "error": "No enhanced prompt found"}
                    else:
                        return {"success": False, "error": "No prompt directory found"}

            except Exception as e:
                return {"success": False, "error": f"Failed to auto-discover inputs: {str(e)}"}

        # Call the actual generation method with discovered or provided parameters
        return self.generate_single_output(frames_dir=frames_dir, enhanced_prompt=enhanced_prompt)


def create_comparison(input_frames, output_image, output_dir):
    """Create a visual comparison of inputs vs output"""
    try:
        # Resize all to same height for comparison
        height = 400

        # Resize input frames
        resized_inputs = []
        for frame in input_frames[:3]:  # Use first 3 frames
            scale = height / frame.height
            new_width = int(frame.width * scale)
            resized = frame.resize((new_width, height), Image.LANCZOS)
            resized_inputs.append(resized)

        # Resize output
        scale = height / output_image.height
        out_width = int(output_image.width * scale)
        resized_output = output_image.resize((out_width, height), Image.LANCZOS)

        # Create comparison image
        total_width = sum(img.width for img in resized_inputs) + 30 + resized_output.width
        comparison = Image.new('RGB', (total_width, height + 60), 'white')

        # Add input frames
        x_offset = 0
        for i, img in enumerate(resized_inputs):
            comparison.paste(img, (x_offset, 30))
            x_offset += img.width + 10

        # Add separator
        x_offset += 10

        # Add output
        comparison.paste(resized_output, (x_offset, 30))

        # Save comparison
        comp_path = output_dir / "input_vs_output.jpg"
        comparison.save(comp_path, quality=95)

        return comp_path

    except Exception as e:
        print_error(f"Could not create comparison: {e}")
        return None


if __name__ == "__main__":
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break

    if not api_key:
        print_error("No GEMINI_API_KEY found!")
    else:
        # Find latest frames and prompt
        frames_dir = Path("temp/step1_frames")
        prompt_dir = Path("temp/step2_prompt")

        if frames_dir.exists() and prompt_dir.exists():
            # Get latest runs
            latest_frames = sorted(frames_dir.iterdir(), key=lambda x: x.name)[-1]
            latest_prompt = sorted(prompt_dir.iterdir(), key=lambda x: x.name)[-1]

            print_success(f"Using frames from: {latest_frames}")
            print_success(f"Using prompt from: {latest_prompt}")

            # Load enhanced prompt
            prompt_file = latest_prompt / "enhanced_prompt.json"
            if prompt_file.exists():
                with open(prompt_file) as f:
                    data = json.load(f)
                    enhanced_prompt_raw = data.get("enhanced_prompt", "")

                    # Parse if needed
                    if "```json" in enhanced_prompt_raw:
                        import re
                        match = re.search(r'"enhanced_prompt":\s*"([^"]+)"', enhanced_prompt_raw)
                        if match:
                            enhanced_prompt = match.group(1)
                        else:
                            enhanced_prompt = enhanced_prompt_raw
                    else:
                        enhanced_prompt = enhanced_prompt_raw

                print_info(f"Prompt: {enhanced_prompt[:100]}...")

                # Generate single output!
                nanobana = NanoBanaSingleOutput(api_key)
                result = nanobana.generate_single_output(
                    frames_dir=str(latest_frames),
                    enhanced_prompt=enhanced_prompt
                )

                if result["success"]:
                    print(f"\n{GREEN}{BOLD}🎉 SUCCESS! NANOBANA GENERATED ONE OUTPUT IMAGE! 🎉{RESET}")
                    print(f"{GREEN}Input: {result['input_frames']} frames → Output: 1 image{RESET}")
                    print(f"{GREEN}Check: {result['output_path']}{RESET}")
                else:
                    print(f"\n{RED}Failed to generate output image{RESET}")
            else:
                print_error("No enhanced prompt found!")
        else:
            print_error("Run Step 1 and Step 2 first!")