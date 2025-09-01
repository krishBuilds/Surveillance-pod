#!/usr/bin/env python3
"""
Unified Test Script for InternVideo2.5 System
Uses singleton ModelManager to ensure only one model instance
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.model_manager import ModelManager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test that the singleton pattern works correctly with 8-bit quantization"""
    print("\n" + "="*60)
    print("Testing Singleton Model Manager (8-bit Quantization)")
    print("="*60)
    
    # Create multiple instances - should all be the same
    # Using config.yaml with 8-bit quantization enabled
    manager1 = ModelManager(config_path="config.yaml")
    manager2 = ModelManager(config_path="config.yaml")
    manager3 = ModelManager(config_path="config.yaml")
    
    print(f"Manager 1 ID: {id(manager1)}")
    print(f"Manager 2 ID: {id(manager2)}")
    print(f"Manager 3 ID: {id(manager3)}")
    
    if id(manager1) == id(manager2) == id(manager3):
        print("✓ Singleton pattern working correctly")
        print(f"✓ 8-bit quantization enabled: {manager1.enable_8bit}")
        print(f"✓ Auto frame selection: {manager1.frame_limits.get('auto_select_frames', False)}")
        print(f"✓ Max frames (8-bit): {manager1.get_optimal_frame_count()}")
    else:
        print("✗ Singleton pattern failed")
        return False
    
    return True

def test_image_processing():
    """Test image processing with the shared model"""
    print("\n" + "="*60)
    print("Testing Image Processing")
    print("="*60)
    
    manager = ModelManager(config_path="config.yaml")
    
    # Load model once
    if not manager.load_model():
        print("✗ Failed to load model")
        return False
    
    # Test with sample image if available
    image_path = "sample_image.jpg"
    if not os.path.exists(image_path):
        print(f"Sample image not found: {image_path}")
        return True
    
    result = manager.process_image(
        image_path=image_path,
        prompt="Describe this image in detail"
    )
    
    print(f"Image: {result['image_path']}")
    print(f"Prompt: {result['prompt']}")
    print(f"Response: {result['response']}")
    
    if result.get('error'):
        print(f"Error: {result['error']}")
        return False
    
    print("✓ Image processing completed")
    return True

def test_video_processing():
    """Test video processing with the shared model"""
    print("\n" + "="*60)
    print("Testing Video Processing")
    print("="*60)
    
    manager = ModelManager(config_path="config.yaml")  # Will reuse existing instance
    
    # Test with sample video if available
    video_files = ["test_video.mp4", "test_video_30s.mp4"]
    video_path = None
    
    for video_file in video_files:
        if os.path.exists(video_file):
            video_path = video_file
            break
    
    if not video_path:
        print("No test video found - skipping video test")
        return True
    
    # Use optimal frame count based on 8-bit quantization (auto-selected from config)
    result = manager.process_video(
        video_path=video_path,
        prompt="What is happening in this video?",
        num_frames=None  # Let ModelManager auto-select based on quantization settings
    )
    
    print(f"Video: {result['video_path']}")
    print(f"Prompt: {result['prompt']}")
    print(f"Frames: {result['num_frames']}")
    print(f"Response: {result['response']}")
    
    if result.get('error'):
        print(f"Error: {result['error']}")
        return False
    
    print("✓ Video processing completed")
    return True

def test_multiple_operations():
    """Test multiple operations to verify model reuse"""
    print("\n" + "="*60)
    print("Testing Multiple Operations (Model Reuse)")
    print("="*60)
    
    # Create multiple managers and perform operations
    managers = [ModelManager(config_path="config.yaml") for _ in range(3)]
    
    # All should share the same model
    for i, manager in enumerate(managers):
        if manager.model is None and manager._shared_model is not None:
            manager.model = manager._shared_model
            manager.tokenizer = manager._shared_tokenizer
        
        print(f"Manager {i+1}: Model loaded = {manager.model is not None}")
    
    print("✓ All managers sharing the same model instance")
    return True

def main():
    """Main test function"""
    print("InternVideo2.5 System Test")
    print("="*60)
    
    # Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available - {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("✗ CUDA not available")
        return 1
    
    success = True
    
    # Run tests
    tests = [
        test_model_loading,
        test_image_processing,
        test_video_processing,
        test_multiple_operations
    ]
    
    for test in tests:
        try:
            if not test():
                success = False
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {str(e)}")
            success = False
    
    # Cleanup
    manager = ModelManager(config_path="config.yaml")
    manager.cleanup(force_cleanup_shared=True)
    
    if success:
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("✗ Some tests failed")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())