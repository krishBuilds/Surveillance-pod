#!/usr/bin/env python3
"""
InternVideo2.5 Main Application
Entry point for video analysis
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.model_manager import ModelManager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="InternVideo2.5 Video Analysis")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/InternVideo2_5",
        help="Path to the model directory"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file to analyze"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file to analyze"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe what you see in detail.",
        help="Prompt for the model"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to sample from video"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model precision"
    )
    parser.add_argument(
        "--fps-sampling",
        type=float,
        help="Custom FPS sampling rate (e.g., 2.0 for 2 FPS)"
    )
    parser.add_argument(
        "--time-bound",
        type=float,
        help="Limit analysis to first N seconds of video"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.video and not args.image:
        logger.error("Please provide either --video or --image path")
        sys.exit(1)
    
    if args.video and not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)
    
    if args.image and not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        sys.exit(1)
    
    # Initialize model manager (singleton pattern ensures only one instance)
    # Uses config.yaml for 8-bit quantization and optimal frame limits
    logger.info("Initializing InternVideo2.5...")
    manager = ModelManager(
        model_path=args.model_path,
        device=args.device,
        precision=args.precision,
        config_path="config.yaml"
    )
    
    # Load model
    if not manager.load_model():
        logger.error("Failed to load model")
        sys.exit(1)
    
    # Process input
    if args.video:
        logger.info(f"Processing video: {args.video}")
        result = manager.process_video(
            video_path=args.video,
            prompt=args.prompt,
            num_frames=args.num_frames,
            fps_sampling=args.fps_sampling,
            time_bound=args.time_bound
        )
    else:
        logger.info(f"Processing image: {args.image}")
        result = manager.process_image(
            image_path=args.image,
            prompt=args.prompt
        )
    
    # Display results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Prompt: {result['prompt']}")
    print(f"Response: {result['response']}")
    
    # Display timing information if available
    if result.get('timing'):
        timing = result['timing']
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        print(f"Video Loading Time: {timing['video_loading_time']:.2f} seconds")
        print(f"AI Inference Time: {timing['inference_time']:.2f} seconds") 
        print(f"Total Processing Time: {timing['total_processing_time']:.2f} seconds")
        print(f"Processing Rate: {timing['frames_per_second']:.2f} frames/second")
        if 'num_frames' in result:
            print(f"Frames Processed: {result['num_frames']} frames")
        print(f"Response Length: {len(result['response'])} characters")
    
    if result.get('error'):
        print(f"Error: {result['error']}")
    
    # Cleanup instance references (shared model stays loaded for reuse)
    manager.cleanup()
    logger.info("Done!")

if __name__ == "__main__":
    main()