"""
Utility Functions for WebApp
Extracted from app.py to improve modularity
"""

import os
import json
import logging
from flask import send_from_directory, Response, send_file
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

def serve_original_video(video_file: str = None):
    """
    Serve original video file for download or streaming
    
    Args:
        video_file: Video filename (can be None when called as route handler)
        
    Returns:
        Flask response with video file
    """
    try:
        # Handle both direct calls and route parameter calls
        if video_file is None:
            from flask import request
            video_file = request.args.get('video_file')
        
        if not video_file:
            return {'error': 'No video file specified'}, 400
        
        # Security: validate file path to prevent directory traversal
        if '..' in video_file or '/' in video_file or '\\' in video_file:
            return {'error': 'Invalid file path'}, 400
        
        video_path = os.path.join('/workspace/surveillance/inputs/videos', video_file)
        
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return {'error': 'Video file not found'}, 404
        
        # Get file info
        file_size = os.path.getsize(video_path)
        file_ext = os.path.splitext(video_file)[1].lower()
        
        # Set appropriate content type
        content_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm'
        }
        
        content_type = content_types.get(file_ext, 'application/octet-stream')
        
        logger.info(f"Serving video: {video_file} ({file_size} bytes, {content_type})")
        
        # Return file with appropriate headers
        return send_file(
            video_path,
            mimetype=content_type,
            as_attachment=False,
            download_name=video_file
        )
        
    except Exception as e:
        logger.error(f"Error serving video: {str(e)}", exc_info=True)
        return {'error': f'Error serving video: {str(e)}'}, 500

def serve_video(filename: str):
    """
    Serve video file from static directory
    
    Args:
        filename: Video filename
        
    Returns:
        Flask response with video file
    """
    try:
        if not filename:
            return {'error': 'No filename specified'}, 400
        
        # Security: validate filename
        if '..' in filename or '/' in filename or '\\' in filename:
            return {'error': 'Invalid filename'}, 400
        
        # Check if file exists in static/videos directory
        static_video_path = os.path.join('static', 'videos', filename)
        
        if os.path.exists(static_video_path):
            logger.info(f"Serving static video: {filename}")
            return send_from_directory('static', f'videos/{filename}')
        
        # Fallback to inputs directory
        inputs_video_path = os.path.join('/workspace/surveillance/inputs/videos', filename)
        
        if not os.path.exists(inputs_video_path):
            logger.warning(f"Video file not found: {filename}")
            return {'error': 'Video file not found'}, 404
        
        logger.info(f"Serving video from inputs: {filename}")
        
        # Get file extension for content type
        file_ext = os.path.splitext(filename)[1].lower()
        content_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm'
        }
        
        content_type = content_types.get(file_ext, 'application/octet-stream')
        
        return send_file(
            inputs_video_path,
            mimetype=content_type,
            as_attachment=False,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error serving video file: {str(e)}", exc_info=True)
        return {'error': f'Error serving video: {str(e)}'}, 500

def get_available_videos() -> Dict[str, Any]:
    """
    Get list of available video files with metadata
    
    Returns:
        Dictionary with video files list and metadata
    """
    try:
        videos_dir = '/workspace/surveillance/inputs/videos'
        
        if not os.path.exists(videos_dir):
            logger.warning(f"Videos directory not found: {videos_dir}")
            return {'videos': [], 'error': 'Videos directory not found'}
        
        video_files = []
        supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        
        for filename in os.listdir(videos_dir):
            file_path = os.path.join(videos_dir, filename)
            
            # Skip directories and hidden files
            if os.path.isdir(file_path) or filename.startswith('.'):
                continue
            
            # Check file extension
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in supported_extensions:
                continue
            
            try:
                # Get file metadata
                stat = os.stat(file_path)
                file_size_mb = stat.st_size / (1024 * 1024)
                
                video_info = {
                    'filename': filename,
                    'size_mb': round(file_size_mb, 2),
                    'size_bytes': stat.st_size,
                    'last_modified': stat.st_mtime,
                    'extension': file_ext
                }
                
                # Try to get video duration (basic check)
                try:
                    import cv2
                    cap = cv2.VideoCapture(file_path)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        video_info.update({
                            'duration_seconds': round(duration, 2),
                            'fps': round(fps, 2),
                            'frame_count': frame_count
                        })
                    cap.release()
                except Exception as e:
                    logger.debug(f"Could not get video metadata for {filename}: {e}")
                
                video_files.append(video_info)
                
            except Exception as e:
                logger.warning(f"Error processing video file {filename}: {e}")
                continue
        
        # Sort by filename
        video_files.sort(key=lambda x: x['filename'])
        
        logger.info(f"Found {len(video_files)} video files")
        
        return {
            'videos': video_files,
            'count': len(video_files),
            'total_size_mb': round(sum(v['size_mb'] for v in video_files), 2)
        }
        
    except Exception as e:
        logger.error(f"Error getting available videos: {str(e)}", exc_info=True)
        return {'videos': [], 'error': str(e)}

def validate_video_file(video_file: str) -> Dict[str, Any]:
    """
    Validate video file and return metadata
    
    Args:
        video_file: Video filename
        
    Returns:
        Dictionary with validation result and metadata
    """
    try:
        # Basic validation
        if not video_file:
            return {'valid': False, 'error': 'No filename provided'}
        
        # Security check
        if '..' in video_file or '/' in video_file or '\\' in video_file:
            return {'valid': False, 'error': 'Invalid filename (path traversal)'}
        
        # Check file extension
        file_ext = os.path.splitext(video_file)[1].lower()
        supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        
        if file_ext not in supported_extensions:
            return {'valid': False, 'error': f'Unsupported file extension: {file_ext}'}
        
        # Check if file exists
        video_path = os.path.join('/workspace/surveillance/inputs/videos', video_file)
        
        if not os.path.exists(video_path):
            return {'valid': False, 'error': 'File not found'}
        
        # Check file size (basic sanity check)
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            return {'valid': False, 'error': 'File is empty'}
        
        if file_size > 10 * 1024 * 1024 * 1024:  # 10GB limit
            return {'valid': False, 'error': 'File too large (max 10GB)'}
        
        # Try to open with OpenCV for basic validation
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                return {'valid': False, 'error': 'Could not open video file'}
            
            # Get basic video info
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            return {
                'valid': True,
                'metadata': {
                    'filename': video_file,
                    'size_bytes': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'fps': round(fps, 2) if fps > 0 else 0,
                    'frame_count': frame_count,
                    'duration_seconds': round(frame_count / fps if fps > 0 else 0, 2),
                    'resolution': f"{width}x{height}",
                    'width': width,
                    'height': height
                }
            }
            
        except Exception as e:
            logger.warning(f"OpenCV validation failed for {video_file}: {e}")
            # Basic file validation passed even if OpenCV failed
            return {
                'valid': True,
                'metadata': {
                    'filename': video_file,
                    'size_bytes': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'warning': f'OpenCV validation failed: {str(e)}'
                }
            }
        
    except Exception as e:
        logger.error(f"Error validating video file {video_file}: {str(e)}", exc_info=True)
        return {'valid': False, 'error': str(e)}

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def format_duration(seconds: float) -> str:
    """
    Format duration in human readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {minutes}m {remaining_seconds:.1f}s"

def create_response(success: bool = True, data: Any = None, error: str = None, 
                   message: str = None) -> Dict[str, Any]:
    """
    Create standardized response dictionary
    
    Args:
        success: Whether the operation was successful
        data: Response data
        error: Error message (if any)
        message: Additional message
        
    Returns:
        Standardized response dictionary
    """
    response = {'success': success}
    
    if data is not None:
        response['data'] = data
    
    if error:
        response['error'] = error
    
    if message:
        response['message'] = message
    
    return response

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent security issues
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Remove or replace dangerous characters
    dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    filename = filename.strip('. ')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"
    
    return filename