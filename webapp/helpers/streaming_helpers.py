"""
Streaming and Caption Processing Helpers
Extracted from app.py to improve modularity
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Generator, Any
from datetime import datetime

logger = logging.getLogger(__name__)

def process_video_for_caption_streaming(manager, video_path: str, video_file: str, 
                                      fps_sampling: float, chunk_size: int, prompt: str,
                                      processing_mode: str, output_format: str, 
                                      request_id: str) -> Generator[Dict, None, None]:
    """
    Process video for caption generation with streaming updates
    
    Args:
        manager: Enhanced model manager instance
        video_path: Full path to video file
        video_file: Video filename
        fps_sampling: FPS sampling rate
        chunk_size: Chunk size in seconds
        prompt: Analysis prompt
        processing_mode: Processing mode
        output_format: Output format
        request_id: Request identifier
        
    Yields:
        Dictionary with streaming updates
    """
    logger.info(f"[{request_id}] Starting streaming caption processing for {video_file}")
    
    try:
        # Get video info
        video_info = manager._get_video_info(video_path)
        duration = video_info.get('duration_seconds', video_info.get('duration', 0))
        # Handle case where duration might be a string
        if isinstance(duration, str):
            try:
                duration = float(duration.replace('s', ''))
            except (ValueError, AttributeError):
                duration = 0
        total_frames = video_info.get('frames', 0)
        
        if duration == 0:
            yield {
                'type': 'error',
                'message': f'Could not determine video duration for {video_file}',
                'request_id': request_id
            }
            return
        
        # Calculate chunks
        from .video_processing import calculate_video_chunks
        logger.info(f"[{request_id}] Duration type: {type(duration)}, value: {duration}")
        chunks = calculate_video_chunks(duration, chunk_size, processing_mode)
        
        total_chunks = len(chunks)
        logger.info(f"[{request_id}] Processing {total_chunks} chunks for {duration:.1f}s video")
        
        # Send initial status
        yield {
            'type': 'status',
            'message': f'Processing {total_chunks} chunks for {duration:.1f}s video',
            'total_chunks': total_chunks,
            'duration': duration,
            'request_id': request_id
        }
        
        all_captions = []
        processing_times = []
        
        # Process chunks
        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            
            # Send chunk start status
            yield {
                'type': 'chunk_start',
                'chunk': i + 1,
                'total_chunks': total_chunks,
                'chunk_id': chunk['chunk_id'],
                'start_time': chunk['start_time'],
                'end_time': chunk['end_time'],
                'message': f'Processing chunk {i+1}/{total_chunks}: {chunk["start_time"]:.1f}s-{chunk["end_time"]:.1f}s',
                'request_id': request_id
            }
            
            # Process the chunk with enhanced temporal analysis (uses deterministic config)
            from .video_processing import process_temporal_video_segment
            result = process_temporal_video_segment(
                model_manager=manager,
                video_path=video_path,
                prompt=prompt,
                num_frames=manager.model_manager.get_optimal_frame_count(),
                start_time=chunk['start_time'],
                end_time=chunk['end_time'],
                fps_sampling=fps_sampling,
                generation_config=None  # Let temporal analysis use its deterministic defaults
            )
            
            chunk_processing_time = time.time() - chunk_start_time
            processing_times.append(chunk_processing_time)
            
            if result.get('success'):
                caption = result.get('caption', '')
                all_captions.append({
                    'chunk_id': chunk['chunk_id'],
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'caption': caption,
                    'processing_time': chunk_processing_time
                })
                
                # Send chunk completion
                yield {
                    'type': 'chunk_complete',
                    'chunk': i + 1,
                    'total_chunks': total_chunks,
                    'chunk_id': chunk['chunk_id'],
                    'caption': caption,
                    'processing_time': chunk_processing_time,
                    'progress': ((i + 1) / total_chunks) * 100,
                    'characters': len(caption),
                    'message': f'Completed chunk {i+1}/{total_chunks}',
                    'request_id': request_id
                }
            else:
                error_msg = result.get('error', 'Unknown error')
                yield {
                    'type': 'chunk_error',
                    'chunk': i + 1,
                    'total_chunks': total_chunks,
                    'chunk_id': chunk['chunk_id'],
                    'error': error_msg,
                    'message': f'Error in chunk {i+1}/{total_chunks}: {error_msg}',
                    'request_id': request_id
                }
        
        # Generate final result
        if all_captions:
            if output_format == 'paragraph':
                # Combine all captions into a coherent paragraph
                final_caption = combine_captions_to_paragraph(all_captions)
            elif output_format == 'list':
                # Format as list with timestamps
                final_caption = format_captions_as_list(all_captions)
            else:
                # Default: simple concatenation
                final_caption = '\n\n'.join([c['caption'] for c in all_captions])
            
            total_processing_time = sum(processing_times)
            avg_processing_time = total_processing_time / len(processing_times) if processing_times else 0
            
            # Send final result
            yield {
                'type': 'complete',
                'success': True,
                'caption': final_caption,
                'total_chunks': total_chunks,
                'processed_chunks': len(all_captions),
                'chunks_processed': len(all_captions),
                'total_processing_time': total_processing_time,
                'avg_processing_time': avg_processing_time,
                'video_file': video_file,
                'duration': duration,
                'fps_sampling': fps_sampling,
                'request_id': request_id
            }
        else:
            yield {
                'type': 'error',
                'message': 'No captions were generated successfully',
                'request_id': request_id
            }
            
    except Exception as e:
        logger.error(f"[{request_id}] Exception in streaming processing: {str(e)}", exc_info=True)
        yield {
            'type': 'error',
            'message': f'Processing error: {str(e)}',
            'request_id': request_id
        }
    
    finally:
        # Final cleanup - always run this regardless of success/failure
        try:
            logger.info(f"[{request_id}] Performing final cleanup of streaming processing...")
            
            # Clear preprocessed data from enhanced manager
            if hasattr(manager, '_clear_preprocessed_data'):
                manager._clear_preprocessed_data()
                logger.info(f"[{request_id}] Enhanced manager preprocessed data cleared")
            
            # Clear processing cache from model manager
            if hasattr(manager, 'model_manager') and hasattr(manager.model_manager, 'clear_processing_cache'):
                manager.model_manager.clear_processing_cache()
                logger.info(f"[{request_id}] Model manager processing cache cleared")
            
            # Clear CUDA cache if available
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"[{request_id}] CUDA cache cleared")
            
            logger.info(f"[{request_id}] Final cleanup completed successfully")
            
        except Exception as cleanup_error:
            logger.warning(f"[{request_id}] Cleanup error: {str(cleanup_error)}")

def process_video_for_caption_optimized(manager, video_path: str, video_file: str,
                                       fps_sampling: float, chunk_size: int, prompt: str,
                                       processing_mode: str, output_format: str,
                                       request_id: str) -> Dict:
    """
    Optimized video processing for caption generation
    
    Args:
        manager: Enhanced model manager instance
        video_path: Full path to video file
        video_file: Video filename
        fps_sampling: FPS sampling rate
        chunk_size: Chunk size in seconds
        prompt: Analysis prompt
        processing_mode: Processing mode
        output_format: Output format
        request_id: Request identifier
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"[{request_id}] Starting optimized caption processing for {video_file}")
    
    try:
        # Get video info
        video_info = manager._get_video_info(video_path)
        duration = video_info.get('duration', 0)
        
        if duration == 0:
            return {
                'success': False,
                'error': 'Could not determine video duration'
            }
        
        # For optimized processing, try to process in one go if possible
        max_duration = 120  # Max duration for single processing (2 minutes)
        
        if duration <= max_duration:
            logger.info(f"[{request_id}] Video is short ({duration:.1f}s), processing in single pass")
            
            # Use enhanced temporal analysis for single pass processing (uses deterministic config)
            from .video_processing import process_temporal_video_segment
            result = process_temporal_video_segment(
                model_manager=manager,
                video_path=video_path,
                prompt=prompt,
                num_frames=manager.model_manager.get_optimal_frame_count(),
                start_time=0,
                end_time=duration,
                fps_sampling=fps_sampling,
                generation_config=None  # Let temporal analysis use its deterministic defaults
            )
            
            if result.get('success'):
                return {
                    'success': True,
                    'caption': result.get('caption', ''),
                    'processing_mode': 'single_pass',
                    'total_chunks': 1,
                    'processing_time': result.get('processing_time', 0),
                    'video_file': video_file,
                    'duration': duration
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Single pass processing failed')
                }
        else:
            # Fall back to chunked processing
            logger.info(f"[{request_id}] Video is long ({duration:.1f}s), using chunked processing")
            
            # Collect all results from streaming generator
            final_result = None
            
            for update in process_video_for_caption_streaming(
                manager, video_path, video_file, fps_sampling, chunk_size,
                prompt, processing_mode, output_format, request_id
            ):
                if update.get('type') == 'complete':
                    final_result = update
                    break
                elif update.get('type') == 'error':
                    return {
                        'success': False,
                        'error': update.get('message', 'Streaming processing failed')
                    }
            
            return final_result or {
                'success': False,
                'error': 'Optimized processing failed to produce result'
            }
            
    except Exception as e:
        logger.error(f"[{request_id}] Exception in optimized processing: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

def process_video_for_caption(manager, video_path: str, video_file: str,
                             fps_sampling: float, chunk_size: int, prompt: str,
                             processing_mode: str, output_format: str,
                             request_id: str) -> Dict:
    """
    Main video processing function for caption generation
    
    Args:
        manager: Enhanced model manager instance
        video_path: Full path to video file
        video_file: Video filename
        fps_sampling: FPS sampling rate
        chunk_size: Chunk size in seconds
        prompt: Analysis prompt
        processing_mode: Processing mode
        output_format: Output format
        request_id: Request identifier
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"[{request_id}] Processing video for caption generation: {video_file}")
    
    # For now, use the optimized version
    return process_video_for_caption_optimized(
        manager, video_path, video_file, fps_sampling, chunk_size,
        prompt, processing_mode, output_format, request_id
    )

# Helper functions for caption formatting
def combine_captions_to_paragraph(captions: List[Dict]) -> str:
    """Combine multiple captions into a coherent paragraph"""
    if not captions:
        return ""
    
    # Sort by start time
    sorted_captions = sorted(captions, key=lambda x: x['start_time'])
    
    # Simple combination with transitions
    combined_parts = []
    for i, caption in enumerate(sorted_captions):
        caption_text = caption['caption']
        
        if i == 0:
            combined_parts.append(f"The video begins with {caption_text.lower()}")
        elif i == len(sorted_captions) - 1:
            combined_parts.append(f"Finally, {caption_text.lower()}")
        else:
            combined_parts.append(f"Then, {caption_text.lower()}")
    
    return " ".join(combined_parts)

def format_captions_as_list(captions: List[Dict]) -> str:
    """Format captions as a list with timestamps"""
    if not captions:
        return ""
    
    sorted_captions = sorted(captions, key=lambda x: x['start_time'])
    
    list_items = []
    for caption in sorted_captions:
        start_time = caption['start_time']
        end_time = caption['end_time']
        caption_text = caption['caption']
        
        list_items.append(f"[{start_time:.1f}s - {end_time:.1f}s] {caption_text}")
    
    return "\n".join(list_items)

# Import the process_video_segment function from video_processing
def process_video_segment(model_manager, video_path: str, prompt: str, num_frames: int,
                         start_time: float, end_time: float, fps_sampling: float = 1.0,
                         generation_config: Dict = None) -> Dict:
    """Import from video_processing module"""
    from .video_processing import process_video_segment as video_process_segment
    return video_process_segment(
        model_manager, video_path, prompt, num_frames,
        start_time, end_time, fps_sampling, generation_config
    )