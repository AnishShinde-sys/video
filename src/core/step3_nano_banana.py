#!/usr/bin/env python3
"""
STEP 3: Gemini Image Generation (Nano Banana)
Uses Google's Gemini API for actual image generation
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from PIL import Image
from io import BytesIO

# Add path for google-genai if needed
sys.path.append('/Users/anishshinde/Library/Python/3.9/lib/python/site-packages')

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
        """Initialize with Gemini API key"""
        # Use provided key or get from environment
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not provided or found in environment")

        # Initialize the client with the new genai library
        self.client = genai.Client(api_key=api_key)
        print_success("Initialized Nano Banana Image Generator with Gemini")

    def generate_output(self, frames_dir: str, enhanced_prompt: str, output_dir: str = None) -> dict:
        """
        Generate transformed images based on input frames and enhanced prompt
        """
        return self.generate_single_output(frames_dir, enhanced_prompt, output_dir)

    def generate_single_output(self,
                              frames_dir: str,
                              enhanced_prompt: str,
                              output_dir: str = None) -> dict:
        """
        Generate transformed images using Gemini's image generation capability
        Takes reference frames and generates new images based on the enhanced prompt
        """

        print_header("STEP 3: NANO BANANA IMAGE GENERATION (GEMINI)")

        # Create output directory
        if output_dir is None:
            output_dir = "temp/step3_single"
        output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        print_step(f"Output directory: {run_dir}")

        # Load frames
        print_step("Loading reference frames...")
        frames_dir = Path(frames_dir)
        frame_files = sorted([f for f in frames_dir.glob("frame_*.jpg")])

        if not frame_files:
            print_error("No frames found!")
            return {"success": False, "error": "No frames found"}

        print_success(f"Found {len(frame_files)} frames")

        # Select key frames for generation (first, middle, last)
        key_frames = []
        if len(frame_files) >= 3:
            indices = [0, len(frame_files)//2, -1]
            selected_frames = [frame_files[i] for i in indices]
        else:
            selected_frames = frame_files[:min(3, len(frame_files))]

        # Load selected frames
        reference_images = []
        for frame_path in selected_frames:
            img = Image.open(frame_path)
            reference_images.append(img)
            print_info(f"Reference frame: {frame_path.name} ({img.size[0]}x{img.size[1]})")

        print_success(f"Loaded {len(reference_images)} reference frames")

        # Generate images based on the enhanced prompt
        generated_images = []

        try:
            # For each key frame, generate a transformed version
            for i, ref_image in enumerate(reference_images):
                print_step(f"Generating transformed image {i+1}/{len(reference_images)}...")

                # Create a generation prompt that combines the reference and transformation
                generation_prompt = f"""Based on this reference image, {enhanced_prompt}

                Maintain the overall composition and scene structure but apply the requested transformation.
                Ensure photorealistic quality and consistency with the original image's perspective and lighting."""

                # Prepare contents for generation
                contents = [generation_prompt, ref_image]

                try:
                    # Generate the transformed image
                    response = self.client.models.generate_content(
                        model="gemini-2.5-flash-image",
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=['Image'],
                            image_config=types.ImageConfig(
                                aspect_ratio="1:1",  # Maintain square aspect ratio
                            )
                        )
                    )

                    # Extract the generated image
                    for part in response.candidates[0].content.parts:
                        if part.inline_data is not None:
                            # Save the generated image
                            generated_img = Image.open(BytesIO(part.inline_data.data))
                            output_path = run_dir / f"generated_frame_{i+1:02d}.jpg"
                            generated_img.save(output_path, quality=95)
                            generated_images.append(str(output_path))
                            print_success(f"Generated image saved: {output_path.name}")
                            break

                except Exception as e:
                    print_error(f"Error generating image {i+1}: {str(e)}")
                    # Continue with other frames even if one fails
                    continue

            if not generated_images:
                # If generation failed, try text-to-image as fallback
                print_step("Attempting text-to-image generation as fallback...")

                # Create a detailed text-to-image prompt
                text_prompt = f"""Create a high-quality, photorealistic image: {enhanced_prompt}

                The image should be clear, well-composed, and professionally rendered."""

                try:
                    response = self.client.models.generate_content(
                        model="gemini-2.5-flash-image",
                        contents=[text_prompt],
                        config=types.GenerateContentConfig(
                            response_modalities=['Image'],
                            image_config=types.ImageConfig(
                                aspect_ratio="1:1",
                            )
                        )
                    )

                    for part in response.candidates[0].content.parts:
                        if part.inline_data is not None:
                            generated_img = Image.open(BytesIO(part.inline_data.data))
                            output_path = run_dir / "generated_fallback.jpg"
                            generated_img.save(output_path, quality=95)
                            generated_images.append(str(output_path))
                            print_success(f"Fallback image generated: {output_path.name}")
                            break

                except Exception as e:
                    print_error(f"Fallback generation also failed: {str(e)}")

            # Save generation metadata
            metadata = {
                "timestamp": timestamp,
                "enhanced_prompt": enhanced_prompt,
                "frames_dir": str(frames_dir),
                "reference_frames": [str(f) for f in selected_frames],
                "generated_images": generated_images,
                "model": "gemini-2.5-flash-image"
            }

            metadata_file = run_dir / "generation_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Copy reference frames for comparison
            for i, frame_path in enumerate(selected_frames):
                import shutil
                dest_path = run_dir / f"reference_frame_{i+1:02d}.jpg"
                shutil.copy(frame_path, dest_path)

            if generated_images:
                print_header("STEP 3 COMPLETE: IMAGE GENERATION SUCCESS")
                print_success(f"Successfully generated {len(generated_images)} images")
                print_info(f"Output directory: {run_dir}")

                # Use the middle generated image as the primary output
                primary_output = generated_images[len(generated_images)//2] if generated_images else generated_images[0]

                return {
                    "success": True,
                    "output_dir": str(run_dir),
                    "generated_images": generated_images,
                    "generated_image": primary_output,
                    "frame_count": len(frame_files),
                    "model": "gemini-2.5-flash-image"
                }
            else:
                error_msg = "Failed to generate any images"
                print_error(error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }

        except Exception as e:
            error_msg = f"Image generation error: {str(e)}"
            print_error(error_msg)

            # Save error log
            error_log = run_dir / "error_log.json"
            with open(error_log, 'w') as f:
                json.dump({
                    "error": error_msg,
                    "exception": str(e),
                    "enhanced_prompt": enhanced_prompt,
                    "frames_dir": str(frames_dir)
                }, f, indent=2)

            return {
                "success": False,
                "error": error_msg
            }


if __name__ == "__main__":
    # Test the generator
    frames_dir = Path("temp/step1_frames")
    prompt_dir = Path("temp/step2_prompt")

    if frames_dir.exists() and prompt_dir.exists():
        frame_runs = sorted([d for d in frames_dir.iterdir() if d.is_dir()],
                           key=lambda x: x.name, reverse=True)
        prompt_runs = sorted([d for d in prompt_dir.iterdir() if d.is_dir()],
                           key=lambda x: x.name, reverse=True)

        if frame_runs and prompt_runs:
            latest_frames = frame_runs[0]
            latest_prompt_run = prompt_runs[0]

            prompt_file = latest_prompt_run / "enhanced_prompt.json"
            if prompt_file.exists():
                with open(prompt_file) as f:
                    prompt_data = json.load(f)
                    enhanced_prompt = prompt_data.get("enhanced_prompt", "Transform the image")

                print_success(f"Found frames: {latest_frames}")
                print_success(f"Found prompt: {enhanced_prompt[:100]}...")

                generator = NanoBanaSingleOutput()
                result = generator.generate_single_output(
                    frames_dir=str(latest_frames),
                    enhanced_prompt=enhanced_prompt
                )

                if result["success"]:
                    print(f"\n{GREEN}Step 3 completed successfully!{RESET}")
                else:
                    print(f"\n{RED}Step 3 failed: {result.get('error')}{RESET}")