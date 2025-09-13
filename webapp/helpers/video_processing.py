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
                         start_time: float, end_time: float, fps_sampling: float = 3.0,
                         generation_config: Dict = None) -> Dict:
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
            end_time=end_time,
            generation_config=generation_config
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
                                 start_time: float, end_time: float, fps_sampling: float = 3.0,
                                 generation_config: Dict = None) -> Dict:
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
        # Calculate frame timestamps for enhanced temporal analysis
        frame_timestamps = []
        for i in range(num_frames):
            timestamp = start_time + (i / fps_sampling)
            if timestamp <= end_time:
                frame_timestamps.append(timestamp)

        # Create enhanced temporal prompt with full frame annotation
        frame_annotations = []
        # Show all frames with timestamps
        for i, timestamp in enumerate(frame_timestamps):
            frame_annotations.append(f"Frame {i+1} (t={timestamp:.1f}s): <image>")

        temporal_prompt = "What happens in this video? Describe the main actions and events."

        # Use more permissive generation config to avoid tracking mode
        temporal_generation_config = {
            'do_sample': True,
            'temperature': 0.7,    # Higher temperature for more creative output
            'max_new_tokens': 1024,
            'top_p': 0.9,
            'num_beams': 1,
            'repetition_penalty': 1.1,
            'length_penalty': 1.0
        }

        # Override with provided config or use temporal defaults
        final_generation_config = generation_config if generation_config else temporal_generation_config

        result = process_video_segment(
            model_manager=model_manager,
            video_path=video_path,
            prompt=temporal_prompt,
            num_frames=num_frames,
            start_time=start_time,
            end_time=end_time,
            fps_sampling=fps_sampling,
            generation_config=final_generation_config
        )

        if result.get('success'):
            result['analysis_type'] = 'temporal_enhanced'
            result['temporal_focus'] = True
            result['frame_timestamps'] = frame_timestamps

        return result

    except Exception as e:
        logger.error(f"Exception in temporal video processing: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'start_time': start_time,
            'end_time': end_time,
            'analysis_type': 'temporal_enhanced'
        }

def merge_events_from_chunks(chunks: List[Dict], time_threshold: float = 2.0) -> List[Dict]:
    """
    Merge events from different chunks that span chunk boundaries

    Args:
        chunks: List of chunk dictionaries with event data
        time_threshold: Maximum time gap to consider events as continuous (seconds)

    Returns:
        List of merged events with unified timing
    """
    merged_events = []

    for chunk in chunks:
        chunk_events = chunk.get('events', [])
        chunk_start = chunk.get('start_time', 0)
        chunk_end = chunk.get('end_time', 0)

        for event in chunk_events:
            # Convert event times to absolute video timeline
            abs_start = chunk_start + event.get('start_offset', 0)
            abs_end = chunk_start + event.get('end_offset', 0)

            # Check if this event can be merged with any existing event
            merged = False
            for existing_event in merged_events:
                # Check if events overlap or are very close
                if (abs(abs_start - existing_event['end']) <= time_threshold or
                    abs(abs_end - existing_event['start']) <= time_threshold or
                    (abs_start >= existing_event['start'] and abs_start <= existing_event['end']) or
                    (abs_end >= existing_event['start'] and abs_end <= existing_event['end'])):

                    # Merge events
                    existing_event['start'] = min(existing_event['start'], abs_start)
                    existing_event['end'] = max(existing_event['end'], abs_end)
                    existing_event['description'] = f"{existing_event['description']} and {event['description']}"
                    existing_event['details'] = f"{existing_event['details']}; {event.get('details', '')}"
                    merged = True
                    break

            if not merged:
                merged_events.append({
                    'description': event.get('description', ''),
                    'start': abs_start,
                    'end': abs_end,
                    'details': event.get('details', ''),
                    'source_chunks': [chunk.get('chunk_id', 0)]
                })

    # Sort by start time
    merged_events.sort(key=lambda x: x['start'])
    return merged_events

def parse_temporal_events_from_response(response_text: str, chunk_start_time: float = 0) -> List[Dict]:
    """
    Parse temporal events from model response using Event/Start/End/Details format

    Args:
        response_text: Model response text
        chunk_start_time: Start time of this chunk for absolute timeline conversion

    Returns:
        List of parsed events with relative timing
    """
    events = []

    # Split response into lines and look for Event patterns
    lines = response_text.split('\n')
    current_event = {}

    for line in lines:
        line = line.strip()

        # Parse Event line
        if line.startswith('Event:') and ':' in line:
            # Save previous event if exists
            if current_event:
                events.append(current_event.copy())

            current_event = {
                'description': line.split(':', 1)[1].strip(),
                'start_offset': 0,
                'end_offset': 0,
                'details': ''
            }

        # Parse Start time
        elif line.startswith('Start:') and current_event:
            try:
                time_str = line.split(':', 1)[1].strip().replace('s', '')
                current_event['start_offset'] = float(time_str)
            except (ValueError, IndexError):
                pass

        # Parse End time
        elif line.startswith('End:') and current_event:
            try:
                time_str = line.split(':', 1)[1].strip().replace('s', '')
                current_event['end_offset'] = float(time_str)
            except (ValueError, IndexError):
                pass

        # Parse Details
        elif line.startswith('Details:') and current_event:
            current_event['details'] = line.split(':', 1)[1].strip()

    # Add the last event if exists
    if current_event and current_event.get('description'):
        events.append(current_event)

    # Filter out events without proper timing
    valid_events = [e for e in events if e['start_offset'] >= 0 and e['end_offset'] > e['start_offset']]

    return valid_events

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
                'chunk_id': i + 1,  # Start chunk IDs from 1
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
            from .db_utils import get_video_caption
            caption_data = get_video_caption(video_file)
            
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