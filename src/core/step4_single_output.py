#!/usr/bin/env python3
"""
Step 4: Complete Video Analysis - Single Output File
Generates ONE comprehensive file with ALL movements, actions, and speech
"""

import os
import sys
import time
import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SingleFileVideoAnalyzer:
    """Video analyzer that outputs everything in ONE comprehensive file"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the analyzer"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found")

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Use Gemini 2.0 Flash
        self.model = genai.GenerativeModel('gemini-2.0-flash')

        # Create output directory
        self.output_dir = Path("temp/step4_complete_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def upload_video(self, video_path: str):
        """Upload video to Gemini"""
        logger.info(f"Uploading: {video_path}")

        mime_type, _ = mimetypes.guess_type(video_path)
        if not mime_type:
            mime_type = 'video/mp4'

        video_file = genai.upload_file(
            path=video_path,
            mime_type=mime_type,
            display_name=os.path.basename(video_path)
        )

        logger.info(f"Uploaded: {video_file.display_name}")

        # Wait for processing
        while video_file.state.name == "PROCESSING":
            logger.info("Processing...")
            time.sleep(3)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError("Video processing failed")

        logger.info("Ready for analysis")
        return video_file

    def analyze_video_complete(self, video_path: str) -> str:
        """Analyze video and create ONE complete output file"""

        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"COMPLETE_ANALYSIS_{timestamp}.txt"

        try:
            # Upload video
            video_file = self.upload_video(video_path)

            # Single comprehensive prompt for EVERYTHING
            logger.info("Performing complete analysis...")
            complete_prompt = """
Analyze this video and provide EVERYTHING in ONE comprehensive report.
Include EVERY detail, movement, word, and action.

Format your response EXACTLY like this:

================================================================================
VIDEO ANALYSIS - COMPLETE REPORT
================================================================================

VIDEO INFORMATION:
- Duration: [exact duration]
- Resolution: [if visible]
- Setting: [location/environment]
- People: [count and brief description]

================================================================================
COMPLETE SPEECH TRANSCRIPTION
================================================================================

[Provide EVERY word spoken with exact timestamps]
Format: [MM:SS.ms] "Exact words spoken here"

Example:
[00:00.74] "My"
[00:00.94] "long"
[00:01.14] "term"
[00:01.34] "plan"
[Continue for EVERY word]

Also include:
- Tone/emotion for each phrase
- Volume changes
- Pauses and hesitations
- Non-words (uh, um, ah)
- Background speech if any

================================================================================
ALL MOVEMENTS AND ACTIONS (CHRONOLOGICAL)
================================================================================

[List EVERY movement with millisecond precision]
Format: [MM:SS.ms] Body Part | Action | Duration | Details

Include ALL:
- Eye movements (every glance, blink, gaze shift)
- Head movements (tilts, nods, turns)
- Facial expressions (every change)
- Mouth movements (for each word)
- Hand gestures (if any)
- Body shifts
- Breathing visible
- Micro-expressions
- Clothing movements
- Background movements

Example:
[00:00.50] Head | Slight tilt right | 200ms | Unconscious adjustment
[00:00.74] Mouth | Forms "My" | 200ms | Clear articulation
[00:01.06] Eyes | Roll right | 200ms | Thinking gesture
[00:01.61] Eyes | Look up and left | 200ms | Recalling information
[Continue for EVERY movement]

================================================================================
PARALLEL ACTION TIMELINE
================================================================================

[Show what's happening simultaneously at each moment]

[00:00.0-00:00.5]
• Speaking: Starting to speak
• Eyes: Looking at camera
• Mouth: Opening
• Head: Slight tilt beginning
• Body: Static
• Breathing: Normal

[00:00.5-00:01.0]
• Speaking: "My"
• Eyes: Maintaining gaze
• Mouth: Forming 'M' then 'y'
• Head: Tilted slightly right
• Body: Static
• Breathing: Normal

[Continue for entire video in 0.5 second increments]

================================================================================
FACIAL EXPRESSION CHANGES
================================================================================

[Every facial expression change with timestamp]
[00:00.00] Neutral expression
[00:01.61] Slight eyebrow raise - thinking
[00:02.70] Eyes squint slightly - concentration
[Continue for all changes]

================================================================================
BODY LANGUAGE ANALYSIS
================================================================================

- Posture: [description]
- Energy level: [throughout video]
- Confidence indicators: [list all]
- Nervous indicators: [if any]
- Engagement level: [description]
- Open/closed body language: [details]

================================================================================
COMPLETE ACTION COUNT
================================================================================

EXACT COUNTS:
- Total words spoken: [number]
- Total eye movements: [number]
- Total blinks: [number]
- Total head movements: [number]
- Total facial expressions: [number]
- Total gestures: [number]
- Total breaths visible: [number]
- Total lip movements: [number]
- Total unique actions: [number]

TIMING STATISTICS:
- Speaking duration: [seconds]
- Silence duration: [seconds]
- Longest pause: [seconds]
- Most active period: [timestamp range]
- Most static period: [timestamp range]

================================================================================
KEY MOMENTS AND HIGHLIGHTS
================================================================================

1. [Timestamp] - [What happened and why it's significant]
2. [Timestamp] - [What happened and why it's significant]
[Continue for all key moments]

================================================================================
EMOTIONAL JOURNEY
================================================================================

[Track emotional changes throughout]
[00:00-00:02] Neutral, starting to explain
[00:02-00:05] Slightly critical, expressing skepticism
[Continue throughout video]

================================================================================
COMPLETE SCENE DESCRIPTION
================================================================================

ENVIRONMENT:
- Location: [detailed description]
- Lighting: [type, direction, quality]
- Background: [everything visible]
- Colors: [all colors present]
- Objects: [list all visible objects]
- Atmosphere: [mood/feeling]

PERSON DESCRIPTION:
- Age: [estimate]
- Appearance: [detailed physical description]
- Clothing: [every item in detail]
- Accessories: [any visible]
- Distinguishing features: [any notable]

================================================================================
AUDIO ANALYSIS
================================================================================

- Speech clarity: [description]
- Background noise: [any detected]
- Audio quality: [description]
- Echo/reverb: [if present]
- Other sounds: [list all]

================================================================================
SUMMARY
================================================================================

[Brief overview of what happened in the video]

================================================================================
END OF COMPLETE ANALYSIS
================================================================================

IMPORTANT:
- Include EVERY movement, no matter how small
- Include EVERY word spoken
- Include EVERY timestamp
- Miss NOTHING
- Be EXHAUSTIVELY detailed
- Provide EXACT counts
"""

            response = self.model.generate_content([video_file, complete_prompt])

            # Write everything to the single output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response.text)

            logger.info(f"Complete analysis saved to: {output_file}")

            # Clean up
            try:
                genai.delete_file(video_file.name)
            except:
                pass

            return str(output_file)

        except Exception as e:
            logger.error(f"Error: {e}")
            raise

    def analyze_video(self, video_path: str) -> Dict:
        """
        Wrapper method for compatibility with app.py
        Returns a dictionary with analysis results
        """
        try:
            # Call the actual analysis method
            output_file = self.analyze_video_complete(video_path)

            # Read the analysis text from the file
            analysis_text = ""
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    analysis_text = f.read()

            return {
                "success": True,
                "output_dir": str(self.output_dir),
                "output_file": output_file,
                "analysis": analysis_text
            }

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

def main():
    """Main execution"""

    print("=" * 80)
    print(" " * 20 + "COMPLETE VIDEO ANALYSIS - SINGLE FILE OUTPUT")
    print("=" * 80)

    # Find video files
    video_files = list(Path('.').glob('*.mp4')) + list(Path('.').glob('*.MP4'))

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    elif video_files:
        if len(video_files) == 1:
            video_path = str(video_files[0])
            print(f"\nAutomatically selected: {video_files[0].name}")
        else:
            print("\nFound videos:")
            for i, vf in enumerate(video_files, 1):
                print(f"  {i}. {vf.name}")
            choice = input("\nSelect number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(video_files):
                video_path = str(video_files[int(choice) - 1])
            else:
                video_path = choice
    else:
        video_path = input("\nEnter video path: ").strip()

    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return 1

    print(f"\nVideo: {video_path}")
    print("-" * 80)

    try:
        # Initialize
        print("\nInitializing analyzer...")
        analyzer = SingleFileVideoAnalyzer()

        # Analyze
        print("\n" + "=" * 80)
        print("GENERATING COMPLETE ANALYSIS")
        print("=" * 80)
        print("\nThis single file will contain:")
        print("  • Every word spoken with timestamps")
        print("  • Every movement and action")
        print("  • Every facial expression")
        print("  • Every gesture and micro-movement")
        print("  • Complete timeline of events")
        print("  • All statistics and counts")
        print("\nAnalyzing...")
        print("-" * 80 + "\n")

        output_file = analyzer.analyze_video_complete(video_path)

        # Display results
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)

        print(f"\n✅ Single comprehensive file created:")
        print(f"   {output_file}")

        # Show file size
        file_size = os.path.getsize(output_file) / 1024
        print(f"\n📊 File size: {file_size:.1f} KB")

        print("\n" + "=" * 80)
        print("SUCCESS! All analysis in ONE complete file!")
        print("Open the file above to see EVERYTHING that happened in the video.")
        print("=" * 80)

    except Exception as e:
        print(f"\nError: {e}")
        logger.error(f"Fatal: {e}", exc_info=True)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())