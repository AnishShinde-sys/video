#!/usr/bin/env python3
"""
STEP 2: Direct Gemini API Integration - Prompt Enhancement
Uses Google's Gemini API directly without OpenRouter
"""

import os
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict
from PIL import Image
import google.generativeai as genai

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
    def __init__(self, api_key: str = None):
        """Initialize the prompt enhancer with Gemini API key"""
        # Use provided key or get from environment
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not provided or found in environment")

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Use Gemini 2.0 Flash which supports vision
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def close(self):
        """For compatibility - no session to close with Gemini"""
        pass

    def __enter__(self):
        """Support context manager"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting context manager"""
        self.close()

    def enhance_prompt(self, user_prompt: str, reference_frame_path: str, output_dir: str = None) -> Dict:
        """
        Enhance the user prompt using Gemini with the first frame as reference

        Args:
            user_prompt: The original user prompt
            reference_frame_path: Path to the first frame to use as reference
            output_dir: Directory to save logs and outputs

        Returns:
            Dictionary with enhanced prompt and analysis
        """
        print_header("STEP 2: PROMPT ENHANCEMENT WITH GEMINI (DIRECT)")

        # Create output directory
        if output_dir is None:
            output_dir = "temp/step2_prompt"
        output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        print_step(f"Output directory: {run_dir}")

        # Load and prepare the reference frame
        print_step("Loading reference frame...")

        if not Path(reference_frame_path).exists():
            print_error(f"Reference frame not found: {reference_frame_path}")
            return {"success": False, "error": "Reference frame not found"}

        try:
            # Load image with PIL
            image = Image.open(reference_frame_path)
            width, height = image.size
            file_size = Path(reference_frame_path).stat().st_size / 1024  # KB

            print_info(f"Frame path: {reference_frame_path}")
            print_info(f"Resolution: {width}x{height}")
            print_info(f"Size: {file_size:.1f} KB")

            # Create the prompt for Gemini
            system_prompt = f"""You are an expert AI prompt engineer specializing in video and image editing instructions.

User's request: "{user_prompt}"

Analyze this reference frame and create a detailed prompt for transforming this image according to the user's request.

Provide a JSON response with these exact fields:
{{
    "image_analysis": "Detailed description of what you see in the reference frame",
    "enhanced_prompt": "A detailed, specific prompt for making the requested changes. Be very specific about colors, positions, styles, and exact modifications needed",
    "technical_details": {{
        "current_state": "Description of relevant current elements in the image",
        "required_changes": ["List of specific changes needed"],
        "color_specifications": "Any specific colors or styles to apply",
        "positioning": "Where and how to apply changes"
    }},
    "considerations": ["Any challenges or important notes for the transformation"]
}}

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""

            print_step("Sending request to Gemini API...")
            print_info(f"Model: gemini-2.0-flash")
            print_info(f"User prompt: \"{user_prompt}\"")
            print_info(f"Image included: Yes ({file_size:.1f} KB)")

            # Generate content with Gemini
            try:
                response = self.model.generate_content([system_prompt, image])

                # Extract the response text
                response_text = response.text

                # Try to parse as JSON
                try:
                    # Clean up response if it has markdown formatting
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0]
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0]

                    enhanced_data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    print_info(f"JSON parsing failed, using raw response: {e}")
                    # If not valid JSON, create a simple structure
                    enhanced_data = {
                        "enhanced_prompt": response_text,
                        "image_analysis": "Analysis included in prompt",
                        "raw_response": True
                    }

                print_success("Received enhanced prompt from Gemini")

                # Display the enhancement
                print_header("ENHANCEMENT RESULTS")

                if "image_analysis" in enhanced_data:
                    print_step("Image Analysis:")
                    analysis = enhanced_data["image_analysis"]
                    print_info(analysis[:200] + "..." if len(analysis) > 200 else analysis)

                if "enhanced_prompt" in enhanced_data:
                    print_step("Enhanced Prompt:")
                    enhanced_prompt = enhanced_data["enhanced_prompt"]
                    print(f"{GREEN}{enhanced_prompt[:300]}...{RESET}" if len(enhanced_prompt) > 300 else f"{GREEN}{enhanced_prompt}{RESET}")

                if "technical_details" in enhanced_data:
                    print_step("Technical Details:")
                    tech = enhanced_data["technical_details"]
                    if isinstance(tech, dict) and "required_changes" in tech:
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
                    f.write(f"Enhanced Prompt:\n{enhanced_data.get('enhanced_prompt', response_text)}\n")

                # Copy reference frame to output dir
                import shutil
                ref_copy = run_dir / "reference_frame.jpg"
                shutil.copy(reference_frame_path, ref_copy)
                print_success(f"Copied reference frame to: {ref_copy}")

                print_header("STEP 2 COMPLETE: PROMPT ENHANCEMENT SUMMARY")
                print_success("Successfully enhanced prompt using Gemini Direct API")
                print_info(f"Original prompt: \"{user_prompt}\"")
                print_info(f"Enhanced prompt length: {len(enhanced_data.get('enhanced_prompt', ''))} characters")
                print_info(f"Output directory: {run_dir}")

                return {
                    "success": True,
                    "original_prompt": user_prompt,
                    "enhanced_prompt": enhanced_data.get("enhanced_prompt", response_text),
                    "analysis": enhanced_data,
                    "output_dir": str(run_dir),
                    "model": "gemini-2.0-flash"
                }

            except Exception as e:
                error_msg = f"Gemini API error: {str(e)}"
                print_error(error_msg)

                # Save error log
                with open(run_dir / "error_log.json", 'w') as f:
                    json.dump({
                        "error": error_msg,
                        "exception": str(e),
                        "user_prompt": user_prompt,
                        "reference_frame": reference_frame_path
                    }, f, indent=2)

                return {
                    "success": False,
                    "error": error_msg
                }

        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            print_error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }


if __name__ == "__main__":
    # Test the enhancer

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
                enhancer = PromptEnhancer()

                # Example user prompt
                user_prompt = "make everyone wear sunglasses"

                result = enhancer.enhance_prompt(
                    user_prompt=user_prompt,
                    reference_frame_path=str(first_frame)
                )

                if result["success"]:
                    print(f"\n{GREEN}Step 2 completed successfully!{RESET}")
                    print(f"Enhanced prompt saved to: {result['output_dir']}")
                else:
                    print(f"\n{RED}Step 2 failed: {result.get('error')}{RESET}")
            else:
                print_error(f"First frame not found: {first_frame}")
        else:
            print_error("No frame extraction runs found. Run step1_frame_extraction.py first!")
    else:
        print_error("No frames directory found. Run step1_frame_extraction.py first!")