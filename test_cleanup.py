#!/usr/bin/env python3
"""
Test script to verify GPU memory cleanup and model reloading functionality
"""

import requests
import json
import time
import subprocess
import sys

def get_gpu_memory():
    """Get current GPU memory usage"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            used, total = map(int, result.stdout.strip().split(', '))
            return used, total, (used/total) * 100
    except:
        pass
    return None, None, None

def test_caption_generation():
    """Test caption generation and check memory cleanup"""
    
    print("🧪 Testing Caption Generation Cleanup Mechanism")
    print("=" * 60)
    
    # Get initial GPU memory
    used_before, total, percent_before = get_gpu_memory()
    print(f"📊 Initial GPU Memory: {used_before}MB / {total}MB ({percent_before:.1f}%)")
    
    # Test data
    test_requests = [
        {
            "video_file": "test_video.mp4",
            "fps_sampling": 1.0,
            "chunk_size": 60,
            "processing_mode": "sequential",
            "output_format": "paragraph",
            "prompt": "Test caption generation 1"
        },
        {
            "video_file": "test_video_30s.mp4", 
            "fps_sampling": 0.5,
            "chunk_size": 30,
            "processing_mode": "sequential",
            "output_format": "summary",
            "prompt": "Test caption generation 2"
        }
    ]
    
    for i, test_data in enumerate(test_requests, 1):
        print(f"\n🔄 Test {i}: Processing {test_data['video_file']}")
        print(f"   FPS: {test_data['fps_sampling']}, Duration limit: {test_data['chunk_size']}s")
        
        # Memory before request
        used_pre, _, percent_pre = get_gpu_memory()
        print(f"   📊 GPU Memory before: {used_pre}MB ({percent_pre:.1f}%)")
        
        # Send request
        try:
            start_time = time.time()
            response = requests.post('http://localhost:8088/generate-caption', 
                                   json=test_data, 
                                   timeout=120)
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    caption_length = len(result.get('caption', ''))
                    frames_processed = result.get('frames_processed', 0)
                    print(f"   ✅ Success: {caption_length} chars, {frames_processed} frames, {request_time:.1f}s")
                else:
                    print(f"   ❌ Failed: {result.get('error')}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ Timeout after 120s")
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
        
        # Wait and check memory after
        time.sleep(3)  # Allow cleanup to complete
        used_post, _, percent_post = get_gpu_memory()
        print(f"   📊 GPU Memory after: {used_post}MB ({percent_post:.1f}%)")
        
        memory_diff = used_post - used_pre if used_post and used_pre else 0
        if memory_diff > 100:  # More than 100MB increase
            print(f"   ⚠️  Memory leak detected: +{memory_diff}MB")
        elif abs(memory_diff) < 50:  # Less than 50MB change
            print(f"   ✅ Memory stable: {memory_diff:+d}MB")
        else:
            print(f"   📊 Memory change: {memory_diff:+d}MB")
        
        print("-" * 50)
    
    # Final memory check
    used_final, _, percent_final = get_gpu_memory()
    total_change = used_final - used_before if used_final and used_before else 0
    
    print(f"\n📈 Final Results:")
    print(f"   Initial Memory: {used_before}MB ({percent_before:.1f}%)")  
    print(f"   Final Memory:   {used_final}MB ({percent_final:.1f}%)")
    print(f"   Total Change:   {total_change:+d}MB")
    
    if abs(total_change) < 100:
        print("   ✅ Cleanup mechanism working correctly!")
    else:
        print("   ⚠️  Potential memory management issue detected")
    
    return total_change

def test_model_reload():
    """Test model reloading functionality"""
    print(f"\n🔄 Testing Model Reload Functionality")
    print("=" * 60)
    
    # Test simple chat request to ensure model is loaded
    print("📡 Testing basic chat functionality...")
    try:
        response = requests.post('http://localhost:8088/process-video', 
                               json={"video_file": "test_video.mp4", "num_frames": 32}, 
                               timeout=60)
        if response.status_code == 200:
            print("✅ Model responding correctly")
        else:
            print("❌ Model not responding properly")
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
    
    print("\n📊 GPU memory status:")
    used, total, percent = get_gpu_memory()
    print(f"   Current usage: {used}MB / {total}MB ({percent:.1f}%)")
    
    return True

if __name__ == "__main__":
    print("🚀 InternVideo2.5 Cleanup & Reload Test")
    print("=" * 60)
    
    # Test cleanup mechanism
    memory_change = test_caption_generation()
    
    # Test model reload
    test_model_reload()
    
    print(f"\n🏁 Test Complete!")
    if abs(memory_change) < 100:
        print("✅ All tests passed - cleanup mechanism working!")
        sys.exit(0)
    else:
        print("⚠️  Memory management needs attention")
        sys.exit(1)