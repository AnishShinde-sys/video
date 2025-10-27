#!/usr/bin/env python3
"""
Test script to verify the pipeline works with dynamic inputs
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the step modules
import step1_frame_extraction as step1
import step2_prompt_enhancement as step2
import step3_single_output as step3
import step4_single_output as step4
import step5_with_auto_upload as step5

def test_pipeline():
    """Test the complete pipeline with dynamic inputs"""

    print("\n" + "="*70)
    print("TESTING COMPLETE PIPELINE WITH DYNAMIC INPUTS")
    print("="*70 + "\n")

    # Check for required API keys
    required_keys = {
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "EACHLABS_API_KEY": os.getenv("EACHLABS_API_KEY")
    }

    missing_keys = [k for k, v in required_keys.items() if not v]
    if missing_keys:
        print(f"❌ Missing API keys: {', '.join(missing_keys)}")
        print("Please set these in your .env file")
        return False

    print("✓ All API keys found\n")

    # Test inputs
    if len(sys.argv) >= 3:
        video_path = sys.argv[1]
        user_prompt = sys.argv[2]
    else:
        # Use defaults for testing
        # Look for any MP4 file in the current directory
        mp4_files = list(Path('.').glob('*.mp4')) + list(Path('.').glob('*.MP4'))
        if mp4_files:
            video_path = str(mp4_files[0])
            print(f"Using video: {video_path}")
        else:
            print("❌ No video file found. Please provide a video path as first argument")
            return False

        user_prompt = "make him wear a professional business suit"
        print(f"Using default prompt: {user_prompt}\n")

    # Verify video exists
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return False

    try:
        # Step 1: Frame Extraction
        print("\n--- Step 1: Frame Extraction ---")
        result1 = step1.extract_frames_step1(video_path)
        if not result1 or not result1.get('success'):
            print(f"❌ Step 1 failed: {result1.get('error', 'Unknown error')}")
            return False

        frames_dir = result1['output_dir']
        # Get first frame
        frames_dir_path = Path(frames_dir)
        frame_files = sorted(frames_dir_path.glob('frame_*.jpg'))
        if not frame_files:
            print("❌ No frames extracted")
            return False
        first_frame = str(frame_files[0])

        print(f"✓ Extracted {len(frame_files)} frames")
        print(f"✓ Frames directory: {frames_dir}")
        print(f"✓ First frame: {first_frame}")

        # Step 2: Prompt Enhancement
        print("\n--- Step 2: Prompt Enhancement ---")
        enhancer = step2.PromptEnhancer(required_keys["OPENROUTER_API_KEY"])
        result2 = enhancer.enhance_prompt(user_prompt, first_frame)

        if not result2 or not result2.get('success'):
            print(f"❌ Step 2 failed: {result2.get('error', 'Unknown error')}")
            return False

        enhanced_prompt = result2.get('enhanced_prompt', user_prompt)
        print(f"✓ Original prompt: {user_prompt}")
        print(f"✓ Enhanced prompt: {enhanced_prompt[:100]}...")

        # Step 3: Image Generation
        print("\n--- Step 3: Image Generation ---")
        generator = step3.NanoBanaSingleOutput()
        result3 = generator.generate_output(
            frames_dir=frames_dir,
            enhanced_prompt=enhanced_prompt
        )

        if not result3 or not result3.get('success'):
            print(f"❌ Step 3 failed: {result3.get('error', 'Unknown error')}")
            return False

        # Find generated image
        output_dir = Path(result3.get('output_dir', result3.get('run_dir', '')))
        nanobana_output = output_dir / "nanobana_output.jpg"
        if not nanobana_output.exists():
            print("❌ Generated image not found")
            return False

        print(f"✓ Generated image: {nanobana_output}")

        # Step 4: Video Analysis
        print("\n--- Step 4: Video Analysis ---")
        analyzer = step4.SingleFileVideoAnalyzer()
        result4 = analyzer.analyze_video(video_path)

        if not result4 or not result4.get('success'):
            print(f"❌ Step 4 failed: {result4.get('error', 'Unknown error')}")
            return False

        analysis_text = result4.get('analysis', '')
        print(f"✓ Analysis completed: {len(analysis_text)} characters")

        # Step 5: Video Generation
        print("\n--- Step 5: Video Generation ---")
        video_generator = step5.EachLabsVideoGenerator()
        result5 = video_generator.generate_video(
            image_path=str(nanobana_output),
            analysis_text=analysis_text
        )

        if not result5 or not result5.get('success'):
            print(f"❌ Step 5 failed: {result5.get('error', 'Unknown error')}")
            return False

        video_path = result5.get('video_path')
        if video_path and Path(video_path).exists():
            print(f"✓ Generated video: {video_path}")
        else:
            video_url = result5.get('video_url')
            if video_url:
                print(f"✓ Video generated (download manually): {video_url}")
            else:
                print("❌ No video generated")
                return False

        print("\n" + "="*70)
        print("✅ PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ Pipeline test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)