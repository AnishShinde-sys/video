#!/usr/bin/env python3
"""
Step 1: Frame Extraction - WITHOUT OpenCV!
Uses imageio for video reading and Pillow for image processing
This version doesn't have threading issues with Flask
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont
import imageio.v3 as iio

# ANSI color codes
RESET = '\033[0m'
BOLD = '\033[1m'
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'

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

def extract_frames_step1(video_path, output_dir="temp/step1_frames"):
    """
    Extract representative frames from video WITHOUT using OpenCV
    Uses imageio for video reading and Pillow for image processing

    Args:
        video_path: Path to input video
        output_dir: Directory to save frames

    Returns:
        Dict with success status, frames list, and metadata
    """

    print_header("STEP 1: FRAME EXTRACTION (No OpenCV)")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print_step(f"Output directory: {run_dir}")

    # Load video using imageio-ffmpeg
    print_step("Loading video with imageio-ffmpeg...")
    try:
        # Use imageio with ffmpeg plugin
        import imageio as iio_legacy

        reader = iio_legacy.get_reader(video_path, 'ffmpeg')
        metadata = reader.get_meta_data()
        fps = metadata.get('fps', 30)
        duration = metadata.get('duration', 0)

        # Read frames into list (limit for memory)
        video = []
        for i, frame in enumerate(reader):
            video.append(frame)
            if i > 1000:  # Limit to prevent memory issues
                break
        reader.close()

        total_frames = len(video)
        if total_frames > 0:
            height, width = video[0].shape[:2]
        else:
            raise ValueError("No frames found in video")

        # Calculate actual duration if not provided
        if duration == 0 and fps > 0:
            duration = total_frames / fps

    except Exception as e:
        print_error(f"Failed to load video: {e}")
        return {"success": False, "error": str(e)}

    print_step("Video Information:")
    print_info(f"Path: {video_path}")
    print_info(f"Resolution: {width}x{height}")
    print_info(f"FPS: {fps:.2f}")
    print_info(f"Total frames: {total_frames}")
    print_info(f"Duration: {duration:.2f} seconds")

    # Select frames to extract (same logic as original)
    num_frames = min(8, total_frames)

    # Calculate frame indices
    frame_indices = []
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        # Start frames (2)
        start_frames = [0, min(int(total_frames * 0.05), total_frames - 1)]

        # Middle frames (4)
        middle_start = int(total_frames * 0.3)
        middle_end = int(total_frames * 0.7)
        middle_step = (middle_end - middle_start) // 4
        middle_frames = [middle_start + i * middle_step for i in range(4)]

        # End frames (2)
        end_frames = [
            max(0, int(total_frames * 0.95)),
            total_frames - 1
        ]

        frame_indices = start_frames + middle_frames + end_frames[:1]  # Take 7 frames
        frame_indices = sorted(list(set(frame_indices)))[:num_frames]

    print_step("Frame Selection Plan:")
    print_info(f"Extracting {len(frame_indices)} frames")

    # Extract and save frames
    print_step("Extracting Frames...")
    extracted_frames = []
    frame_metadata = []

    for idx, frame_num in enumerate(frame_indices):
        # Determine section
        if frame_num < total_frames * 0.1:
            section = "start"
        elif frame_num > total_frames * 0.9:
            section = "end"
        else:
            section = "middle"

        # Get frame
        try:
            frame = video[frame_num]

            # Convert to PIL Image
            if isinstance(frame, np.ndarray):
                image = Image.fromarray(frame)
            else:
                image = frame

            # Save frame using Pillow (no threading issues!)
            frame_filename = f"frame_{idx+1:02d}_{section}_{frame_num:04d}.jpg"
            frame_path = run_dir / frame_filename

            image.save(frame_path, "JPEG", quality=95)

            # Verify file was saved
            file_size = frame_path.stat().st_size / 1024  # KB

            timestamp_sec = frame_num / fps if fps > 0 else 0

            print_success(f"Frame {idx+1}/{len(frame_indices)}: {section} section, frame #{frame_num} @ {timestamp_sec:.2f}s → {frame_filename}")
            print_info(f"Shape: {frame.shape}, Size: {file_size:.1f} KB")

            extracted_frames.append(np.array(image))

            frame_metadata.append({
                "index": frame_num,
                "timestamp": timestamp_sec,
                "section": section,
                "filename": frame_filename,
                "shape": list(frame.shape)
            })

        except Exception as e:
            print_error(f"Failed to extract frame {frame_num}: {e}")

    # Save metadata
    metadata_path = run_dir / "frame_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "video_path": str(video_path),
            "total_frames": total_frames,
            "fps": fps,
            "duration": duration,
            "resolution": f"{width}x{height}",
            "extracted_frames": frame_metadata
        }, f, indent=2)
    print_success(f"Saved metadata to: {metadata_path}")

    # Create debug montage using Pillow
    print_step("Creating debug montage...")
    create_montage_pillow(extracted_frames, run_dir)

    # Summary
    print_header("STEP 1 COMPLETE: FRAME EXTRACTION SUMMARY")
    print_success(f"Successfully extracted {len(extracted_frames)} frames")
    print_info(f"Output directory: {run_dir}")

    return {
        "success": True,
        "frames": extracted_frames,
        "metadata": frame_metadata,
        "output_dir": str(run_dir),
        "video_info": {
            "resolution": f"{width}x{height}",
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration
        }
    }

def create_montage_pillow(frames, output_dir):
    """Create a montage of frames using Pillow instead of OpenCV"""

    if not frames:
        return

    # Resize frames for montage
    montage_size = (160, 120)  # Small size for montage

    # Create montage grid (2 rows, 4 columns)
    cols = 4
    rows = 2

    # Create blank montage image
    montage_width = montage_size[0] * cols
    montage_height = montage_size[1] * rows
    montage = Image.new('RGB', (montage_width, montage_height), 'black')

    # Place frames in montage
    for idx, frame in enumerate(frames[:8]):
        if idx >= cols * rows:
            break

        # Convert numpy array to PIL Image
        if isinstance(frame, np.ndarray):
            img = Image.fromarray(frame)
        else:
            img = frame

        # Resize
        img_resized = img.resize(montage_size, Image.Resampling.LANCZOS)

        # Calculate position
        row = idx // cols
        col = idx % cols
        x = col * montage_size[0]
        y = row * montage_size[1]

        # Paste into montage
        montage.paste(img_resized, (x, y))

        # Add label
        draw = ImageDraw.Draw(montage)
        label = f"Frame {idx + 1}"
        # Use default font (no need for OpenCV font)
        draw.text((x + 5, y + 5), label, fill=(0, 255, 0))

    # Save montage
    montage_path = output_dir / "debug_montage.jpg"
    montage.save(montage_path, "JPEG", quality=95)
    print_success(f"Created debug montage: {montage_path}")

if __name__ == "__main__":
    # Test with e10 video
    video_path = "/Users/anishshinde/Downloads/e10ca805-88e2-474d-85cb-aaf8837357d4 (1).MP4"

    if Path(video_path).exists():
        result = extract_frames_step1(video_path)

        if result["success"]:
            print(f"\n{GREEN}✅ Frame extraction successful (NO OPENCV!){RESET}")
            print(f"{BLUE}This version works perfectly with Flask threading!{RESET}")
    else:
        print_error(f"Video not found: {video_path}")