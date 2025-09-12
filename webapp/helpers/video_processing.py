"""
Video Processing Helpers for WebApp
Extracted from app.py to improve modularity
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Generator, Any
from datetime import datetime

logger = logging.getLogger(__name__)

def process_video_segment(model_manager, video_path: str, prompt: str, num_frames: int, 
                         start_time: float, end_time: float, fps_sampling: float = 1.0) -> Dict:
    """
    Process a video segment with temporal bounds and frame control
    
    Args:
        model_manager: Enhanced model manager instance
        video_path: Path to video file
        prompt: Analysis prompt
        num_frames: Number of frames to extract
        start_time: Start time in seconds
        end_time: End time in seconds
        fps_sampling: FPS sampling rate
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Processing video segment: {start_time:.1f}s-{end_time:.1f}s ({num_frames} frames)")
    
    try:
        # Process video segment directly using the model manager with temporal bounds
        result = model_manager.process_video(
            video_path=video_path,
            prompt=prompt,
            num_frames=num_frames,
            fps_sampling=fps_sampling,
            start_time=start_time,
            end_time=end_time
        )
        
        if result and result.get('success'):
            # Extract the caption/response
            caption = result.get('response', '')
            
            # Calculate actual frame timestamps based on sampling
            frame_timestamps = []
            for i in range(num_frames):
                timestamp = start_time + (i / fps_sampling)
                if timestamp <= end_time:
                    frame_timestamps.append(timestamp)
            
            return {
                'success': True,
                'response': caption,  # Include response field for consistency
                'caption': caption,
                'start_time': start_time,
                'end_time': end_time,
                'num_frames': num_frames,
                'fps_sampling': fps_sampling,
                'frame_timestamps': frame_timestamps,
                'processing_time': result.get('processing_time', 0),
                'actual_duration': end_time - start_time
            }
        else:
            error_msg = result.get('error', 'Unknown processing error') if result else 'Processing failed'
            logger.error(f"Model processing failed: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'start_time': start_time,
                'end_time': end_time
            }
            
    except Exception as e:
        logger.error(f"Exception processing video segment: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'start_time': start_time,
            'end_time': end_time
        }

def process_temporal_video_segment(model_manager, video_path: str, prompt: str, num_frames: int,
                                 start_time: float, end_time: float, fps_sampling: float = 1.0) -> Dict:
    """
    Process video segment with temporal analysis - focused on time-based events
    
    Args:
        model_manager: Enhanced model manager instance
        video_path: Path to video file
        prompt: Analysis prompt
        num_frames: Number of frames to extract
        start_time: Start time in seconds
        end_time: End time in seconds
        fps_sampling: FPS sampling rate
        
    Returns:
        Dictionary with temporal analysis results
    """
    logger.info(f"Processing temporal video segment: {start_time:.1f}s-{end_time:.1f}s")
    
    try:
        # Use temporal-specific prompt
        temporal_prompt = f"""
        Analyze this video segment from {start_time:.1f}s to {end_time:.1f}s and provide:
        1. Key events and actions that occur
        2. Timeline of activities
        3. Notable changes or transitions
        4. {prompt}
        
        Focus on temporal progression and sequence of events.
        """
        
        result = process_video_segment(
            model_manager=model_manager,
            video_path=video_path,
            prompt=temporal_prompt,
            num_frames=num_frames,
            start_time=start_time,
            end_time=end_time,
            fps_sampling=fps_sampling
        )
        
        if result.get('success'):
            result['analysis_type'] = 'temporal'
            result['temporal_focus'] = True
            
        return result
        
    except Exception as e:
        logger.error(f"Exception in temporal video processing: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'start_time': start_time,
            'end_time': end_time,
            'analysis_type': 'temporal'
        }

def calculate_video_chunks(duration: float, chunk_size: int = 60, processing_mode: str = 'sequential') -> List[Dict]:
    """
    Calculate video chunks for processing
    
    Args:
        duration: Total video duration in seconds
        chunk_size: Size of each chunk in seconds
        processing_mode: Processing mode ('sequential', 'overlap', 'adaptive')
        
    Returns:
        List of chunk dictionaries with start/end times
    """
    chunks = []
    
    if processing_mode == 'sequential':
        # Simple sequential chunks
        num_chunks = int(duration / chunk_size) + (1 if duration % chunk_size else 0)
        
        for i in range(num_chunks):
            start_time = i * chunk_size
            end_time = min((i + 1) * chunk_size, duration)
            
            chunks.append({
                'chunk_id': i,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'overlap': 0
            })
            
    elif processing_mode == 'overlap':
        # Overlapping chunks for better continuity
        overlap_seconds = chunk_size * 0.2  # 20% overlap
        
        current_time = 0
        chunk_id = 0
        
        while current_time < duration:
            end_time = min(current_time + chunk_size, duration)
            
            chunks.append({
                'chunk_id': chunk_id,
                'start_time': current_time,
                'end_time': end_time,
                'duration': end_time - current_time,
                'overlap': overlap_seconds if current_time > 0 else 0
            })
            
            current_time = end_time - overlap_seconds
            chunk_id += 1
            
    elif processing_mode == 'adaptive':
        # Adaptive chunks based on content complexity (simplified)
        # For now, use smaller chunks for longer videos
        adaptive_size = min(chunk_size, max(30, duration / 10))  # At least 30s chunks
        
        current_time = 0
        chunk_id = 0
        
        while current_time < duration:
            end_time = min(current_time + adaptive_size, duration)
            
            chunks.append({
                'chunk_id': chunk_id,
                'start_time': current_time,
                'end_time': end_time,
                'duration': end_time - current_time,
                'overlap': 0,
                'adaptive': True
            })
            
            current_time = end_time
            chunk_id += 1
    
    logger.info(f"Calculated {len(chunks)} chunks for {duration:.1f}s video (mode: {processing_mode})")
    return chunks

def get_video_caption_data(video_file: str) -> Dict[str, Any]:
    """
    Get comprehensive video and caption data
    
    Args:
        video_file: Video filename
        
    Returns:
        Dictionary with video information and caption data
    """
    video_path = os.path.join('/workspace/surveillance/inputs/videos', video_file)
    
    if not os.path.exists(video_path):
        return {
            'exists': False,
            'error': 'Video file not found'
        }
    
    try:
        # Get video file info
        file_stat = os.stat(video_path)
        file_size_mb = file_stat.st_size / (1024 * 1024)
        
        # Basic video info (would need video processing library for detailed info)
        video_data = {
            'exists': True,
            'filename': video_file,
            'path': video_path,
            'size_mb': round(file_size_mb, 2),
            'last_modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
        }
        
        # Try to get caption data from database
        try:
            from simple_caption_storage import SimpleCaptionStorage
            caption_db = SimpleCaptionStorage()
            caption_data = caption_db.get_caption(video_file)
            
            if caption_data:
                video_data['caption'] = caption_data
                video_data['has_caption'] = True
            else:
                video_data['has_caption'] = False
        except Exception as e:
            logger.warning(f"Could not get caption data: {str(e)}")
            video_data['has_caption'] = False
        
        return video_data
        
    except Exception as e:
        logger.error(f"Error getting video data: {str(e)}", exc_info=True)
        return {
            'exists': True,
            'error': str(e),
            'filename': video_file
        }