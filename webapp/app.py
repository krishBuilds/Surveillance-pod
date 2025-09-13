#!/usr/bin/env python3
"""
InternVideo2.5 Enhanced Web Interface
Web app for video analysis with streaming responses and chat functionality
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_template, send_file
import subprocess
import json
import os
import sys
import re
import time
from datetime import datetime
import logging
import threading
import torch
from typing import List
from pathlib import Path

# Add parent directory to path to access core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import simple caption storage
from simple_caption_storage import SimpleCaptionStorage
# Import model client for persistent model connection (local to webapp)
from model_server import ModelClient

# Import helper modules
from helpers import (
    process_video_segment,
    process_temporal_video_segment,
    process_video_for_caption_streaming,
    process_video_for_caption,
    calculate_video_chunks,
    get_available_videos,
    serve_original_video,
    serve_video,
    validate_video_file,
    get_video_caption_data,
    create_response
)

# Import temporal analysis helpers
from helpers.temporal_analysis import convert_frames_to_timestamps, extract_temporal_events, format_temporal_analysis

# Import OpenAI for chunk determination
from openai import OpenAI
import yaml

# Environment detection - default to development mode
IS_DEVELOPMENT = os.environ.get('FLASK_ENV') == 'development' or os.environ.get('DEV_MODE') == '1' or True

app = Flask(__name__)
app.config['SECRET_KEY'] = 'internvideo-surveillance-app'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inputs/videos")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Development settings
if IS_DEVELOPMENT:
    app.config['DEBUG'] = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    logger.info("🔧 DEVELOPMENT MODE ENABLED")
    logger.info("   - Auto-reload enabled")
    logger.info("   - Debug mode active")
    logger.info("   - Model persistence enabled")

# Default video path
DEFAULT_VIDEO = "inputs/videos/bigbuckbunny.mp4"


# Model server configuration
MODEL_SERVER_HOST = os.environ.get('MODEL_SERVER_HOST', 'localhost')
MODEL_SERVER_PORT = int(os.environ.get('MODEL_SERVER_PORT', '8188'))

# Global model client instance
model_client = None

# Global caption storage instance
caption_db = None

# OpenAI client for chunk determination
openai_client = None
openai_config = None

# Global caption storage for Q&A functionality
caption_storage = {
    # Format: 'session_id': {
    #     'video_path': str,
    #     'chunks': [
    #         {'chunk_id': int, 'start_time': float, 'end_time': float, 'caption': str, 'frames': int},
    #     ],
    #     'total_duration': float,
    #     'timestamp': datetime
    # }
}

def get_caption_db():
    """Get or create the simple caption storage instance"""
    global caption_db
    if caption_db is None:
        caption_db = SimpleCaptionStorage()
    return caption_db

def get_openai_client():
    """Get or create the OpenAI client for chunk determination"""
    global openai_client, openai_config
    if openai_client is None:
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'openai_config.yaml')
            with open(config_path, 'r') as f:
                openai_config = yaml.safe_load(f)
            
            openai_client = OpenAI(
                api_key=openai_config['openai']['api_key']
            )
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            openai_client = None
            openai_config = None
    return openai_client

def get_model_client():
    """Get or create the model client with connection management"""
    global model_client
    if model_client is None:
        logger.info("=== INITIALIZING MODEL CLIENT ===")
        logger.info(f"Connecting to model server at {MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}")
        
        try:
            model_client = ModelClient(host=MODEL_SERVER_HOST, port=MODEL_SERVER_PORT)
            
            # Test connection
            logger.info("Testing model server connection...")
            ping_result = model_client.ping()
            
            if ping_result and ping_result.get('status') == 'ok':
                logger.info("SUCCESS: Connected to model server")
                logger.info("=== MODEL CLIENT READY ===")
                return model_client
            else:  # Should not happen with server-only mode
                logger.error(f"FAILED to connect to model server: {ping_result}")
                model_client = None
                return None
                
        except Exception as e:
            logger.error(f"EXCEPTION during client initialization: {str(e)}", exc_info=True)
            model_client = None
            return None
    
    return model_client

def check_model_server():
    """Check if model server is available and reconnect if needed"""
    global model_client
    
    # Always use model server - no local mode
    
    if model_client is None:
        return get_model_client()
    
    try:
        # Test connection
        ping_result = model_client.ping()
        if not ping_result or ping_result.get('status') != 'ok':
            logger.warning("Model server connection lost, attempting to reconnect...")
            model_client = None
            return get_model_client()
            
        return model_client
        
    except Exception as e:
        logger.error(f"Model server check failed: {str(e)}")
        model_client = None
        return get_model_client()



def startup_cleanup():
    """Clean up memory and cache on app startup while keeping model loaded"""
    logger.info("=== WEBAPP STARTUP CLEANUP ===")

    if IS_DEVELOPMENT:
        logger.info("🔧 Development startup - optimizing for persistence")

    try:
        # Clear any existing CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")

        # Initialize model client
        client = get_model_client()
        if client:
            logger.info("Model client connection established")
        else:  # Should not happen with server-only mode
            logger.warning("Failed to connect to model server - webapp will work but model functions may fail")
            
    except Exception as e:
        logger.error(f"Startup cleanup error: {str(e)}")

@app.route('/')
def index():
    """Main interface page"""
    return render_template('index.html')

@app.route('/available-videos')
def get_available_videos():
    """Get list of available video files"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        videos_dir = os.path.join(base_dir, "inputs/videos")
        
        if not os.path.exists(videos_dir):
            return jsonify({'error': 'Videos directory not found'}), 404
            
        # Get video files with metadata
        videos = []
        for filename in os.listdir(videos_dir):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(videos_dir, filename)
                file_size = os.path.getsize(video_path)
                
                # Try to get video duration using cv2 (more reliable than ffprobe)
                try:
                    import cv2
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        
                        if duration > 0:
                            minutes = int(duration // 60)
                            seconds = int(duration % 60)
                            duration_str = f"{minutes}:{seconds:02d}"
                        else:  # Should not happen with server-only mode
                            duration_str = "Unknown"
                        cap.release()
                    else:  # Should not happen with server-only mode
                        duration_str = "Unknown"
                except Exception as e:
                    logger.warning(f"Could not get duration for {filename}: {str(e)}")
                    duration_str = "Unknown"
                
                videos.append({
                    'filename': filename,
                    'size_mb': round(file_size / (1024*1024), 1),
                    'duration': duration_str
                })
        
        # Sort by filename
        videos.sort(key=lambda x: x['filename'])
        return jsonify({'videos': videos})
        
    except Exception as e:
        logger.error(f"Error getting available videos: {str(e)}")
        return jsonify({'error': f'Failed to get video list: {str(e)}'}), 500




@app.route('/video-stream')
@app.route('/video-stream/<video_file>')
def serve_original_video(video_file=None):
    """Serve the original video file for streaming"""
    try:
        # Use default video if none specified
        if not video_file:
            video_file = 'bigbuckbunny.mp4'
        
        # Get available video files from directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        videos_dir = os.path.join(base_dir, "inputs/videos")
        available_videos = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        # Validate video file
        if video_file not in available_videos:
            return jsonify({'error': f'Video file not allowed: {video_file}. Available: {", ".join(available_videos)}'}), 400
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_path = os.path.join(base_dir, f"inputs/videos/{video_file}")
        
        if not os.path.exists(video_path):
            return jsonify({'error': f'Video file not found: {video_file}'}), 404
            
        return send_from_directory(
            os.path.dirname(video_path), 
            os.path.basename(video_path),
            mimetype='video/mp4'
        )
    except Exception as e:
        logger.error(f"Error serving video: {str(e)}")
        return jsonify({'error': f'Video serving failed: {str(e)}'}), 404

@app.route('/generate-caption-stream', methods=['POST'])
def generate_caption_stream():
    """Stream real-time caption generation with chunk-by-chunk updates"""
    request_id = str(int(time.time()))[-6:]
    
    # Extract parameters outside the generator to avoid request context issues
    try:
        data = request.get_json()
        video_file = data.get('video_file', '')
        fps_sampling = float(data.get('fps_sampling', 1.0))
        chunk_size = int(data.get('chunk_size', 60))
        prompt = data.get('prompt', 'Generate a detailed caption describing what happens in this video.')
        processing_mode = data.get('processing_mode', 'sequential')
        output_format = data.get('output_format', 'paragraph')
    except Exception as e:
        return jsonify({'error': f'Invalid request data: {str(e)}'}), 400
    
    def generate_caption_updates():
        final_result = None
        try:
            logger.info(f"[{request_id}] === STREAMING CAPTION GENERATION START ===")

            # Check model availability based on server mode
            if True:  # Always use model server
                client = check_model_server()
                if not client:
                    logger.error(f"[{request_id}] FAILED: Model server not available")
                    yield f"data: {json.dumps({'error': 'Model server not available'})}\n\n"
                    return
                model_source = "server"
            else:  # Should not happen with server-only mode
                manager = get_enhanced_manager()
                if not manager:
                    logger.error(f"[{request_id}] FAILED: Model manager not initialized")
                    yield f"data: {json.dumps({'error': 'Model manager not initialized'})}\n\n"
                    return
                model_source = "local"
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': f'Starting caption generation for {video_file} ({model_source} mode)', 'request_id': request_id})}\n\n"

            # Stream caption generation based on server mode
            video_path = os.path.join('/workspace/surveillance/inputs/videos', video_file)

            if True:  # Always use model server
                # Use model server with existing helper function for consistency
                # Create a simple adapter to make ModelClient work with streaming helpers
                class ModelClientAdapter:
                    def __init__(self, client, video_path, fps_sampling, chunk_size):
                        self.client = client
                        self.video_path = video_path
                        self.fps_sampling = fps_sampling
                        self.chunk_size = chunk_size
                        # Mock model_manager for compatibility
                        self.model_manager = self

                    def _get_video_info(self, video_path):
                        # Simple video info using OpenCV
                        import cv2
                        try:
                            cap = cv2.VideoCapture(video_path)
                            if cap.isOpened():
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                duration = frame_count / fps if fps > 0 else 0
                                cap.release()
                                return {
                                    'duration_seconds': duration,
                                    'duration': duration,
                                    'frames': frame_count,
                                    'fps': fps
                                }
                            else:  # Should not happen with server-only mode
                                return {'duration_seconds': 0, 'duration': 0, 'frames': 0, 'fps': 0}
                        except:
                            return {'duration_seconds': 0, 'duration': 0, 'frames': 0, 'fps': 0}

                    def get_optimal_frame_count(self):
                        # Store the fps_sampling and chunk_size from the streaming parameters
                        # This will be calculated dynamically based on chunk size and fps_sampling
                        return min(160, int(self.fps_sampling * self.chunk_size))

                    def process_video(self, video_path, prompt, num_frames, fps_sampling=1.0,
                                    start_time=0, end_time=None, generation_config=None):
                        # Use the model client to process video
                        return self.client.process_video(
                            video_path=video_path,
                            prompt=prompt,
                            num_frames=num_frames,
                            fps_sampling=fps_sampling,
                            start_time=start_time,
                            end_time=end_time,
                            generation_config=generation_config
                        )

                # Use the exact same processing flow as local mode
                adapter = ModelClientAdapter(client, video_path, fps_sampling, chunk_size)
                for update in process_video_for_caption_streaming(
                    adapter, video_path, video_file, fps_sampling,
                    chunk_size, prompt, processing_mode, output_format, request_id
                ):
                    # Replace the model manager calls with model client calls
                    if update.get('type') == 'processing_chunk':
                        # Extract chunk info and use model server
                        chunk_info = update.get('chunk_info', {})
                        start_time = chunk_info.get('start_time', 0)
                        end_time = chunk_info.get('end_time', 60)
                        chunk_duration = end_time - start_time
                        chunk_frames = min(160, int(fps_sampling * chunk_duration))

                        chunk_result = client.process_video(
                            video_path=video_path,
                            prompt=prompt,
                            num_frames=chunk_frames,
                            fps_sampling=fps_sampling,
                            start_time=start_time,
                            end_time=end_time
                        )

                        if chunk_result and chunk_result.get('success'):
                            update['success'] = True
                            update['response'] = chunk_result.get('response', '')
                            update['frames_processed'] = chunk_result.get('num_frames', chunk_frames)
                        else:  # Should not happen with server-only mode
                            update['success'] = False
                            update['error'] = chunk_result.get('error', 'Chunk processing failed') if chunk_result else 'Unknown error'

                    # Capture the final result for storage
                    if update.get('type') == 'complete' and update.get('success'):
                        final_result = update

                    yield f"data: {json.dumps(update)}\n\n"

            else:  # Should not happen with server-only mode
                # Use local model for streaming
                for update in process_video_for_caption_streaming(
                    manager, video_path, video_file, fps_sampling,
                    chunk_size, prompt, processing_mode, output_format, request_id
                ):
                    # Capture the final result for storage
                    if update.get('type') == 'complete' and update.get('success'):
                        final_result = update

                    yield f"data: {json.dumps(update)}\n\n"
            
            # Store caption in database after processing completes
            if final_result:
                try:
                    db = get_caption_db()
                    storage_data = {
                        'caption': final_result.get('caption', ''),
                        'chunk_details': [],  # Model server doesn't provide chunk details
                        'video_duration': 0,  # Will be calculated from video info
                        'frames_processed': final_result.get('frames_processed', final_result.get('total_frames', 0)),
                        'fps_used': fps_sampling,
                        'chunk_size': chunk_size,
                        'analysis_method': final_result.get('analysis_method', 'temporal_analysis')
                    }
                    
                    if db.store_video_caption(video_file, storage_data):
                        logger.info(f"[{request_id}] Caption automatically stored after streaming for video: {video_file}")
                    else:  # Should not happen with server-only mode
                        logger.warning(f"[{request_id}] Failed to store caption after streaming for video: {video_file}")
                        
                except Exception as storage_error:
                    logger.error(f"[{request_id}] Error storing caption after streaming: {str(storage_error)}")
                
        except Exception as e:
            logger.error(f"[{request_id}] Streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': f'Streaming error: {str(e)}'})}\n\n"
    
    return Response(generate_caption_updates(), mimetype='text/event-stream')

@app.route('/api/model-status')
def get_model_status():
    """Get model server status"""
    try:
        if True:  # Always use model server
            client = check_model_server()
            if client:
                model_info = client.get_model_info()
                if model_info and model_info.get('model_loaded'):
                    return jsonify({
                        'status': 'connected',
                        'model_loaded': True,
                        'device': model_info.get('device'),
                        'precision': model_info.get('precision'),
                        'max_frames': model_info.get('max_frames'),
                        'mode': 'server'
                    })
                else:  # Should not happen with server-only mode
                    return jsonify({
                        'status': 'connected',
                        'model_loaded': False,
                        'error': 'Model not loaded on server',
                        'mode': 'server'
                    })
            else:  # Should not happen with server-only mode
                return jsonify({
                    'status': 'disconnected',
                    'model_loaded': False,
                    'error': 'Cannot connect to model server',
                    'mode': 'server'
                })
        else:  # Should not happen with server-only mode
            manager = get_enhanced_manager()
            if manager and manager.model_manager.model:
                return jsonify({
                    'status': 'loaded',
                    'model_loaded': True,
                    'device': str(manager.model_manager.device),
                    'precision': manager.model_manager.precision,
                    'max_frames': manager.model_manager.get_optimal_frame_count(),
                    'mode': 'local'
                })
            else:  # Should not happen with server-only mode
                return jsonify({
                    'status': 'not_loaded',
                    'model_loaded': False,
                    'error': 'Local model not initialized',
                    'mode': 'local'
                })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'model_loaded': False,
            'error': str(e),
            'mode': USE_MODEL_SERVER and 'server' or 'local'
        })

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    """Generate detailed caption for entire video using FPS-based sampling and chunks with real-time updates"""
    request_id = str(int(time.time()))[-6:]
    
    try:
        logger.info(f"[{request_id}] === CAPTION GENERATION REQUEST START ===")
        
        # Check model availability
        if True:  # Always use model server
            client = check_model_server()
            if not client:
                logger.error(f"[{request_id}] FAILED: Model server not available")
                return jsonify({'error': 'Model server not available'}), 500
            model_source = "server"
        else:  # Should not happen with server-only mode
            manager = get_enhanced_manager()
            if not manager:
                logger.error(f"[{request_id}] FAILED: Model manager not initialized")
                return jsonify({'error': 'Model manager not initialized'}), 500
            model_source = "local"
            
        data = request.get_json()
        logger.info(f"[{request_id}] Caption request data: {json.dumps(data, indent=2)}")
        
        # Extract parameters
        video_file = data.get('video_file', '')
        fps_sampling = float(data.get('fps_sampling', 1.0))
        chunk_size = int(data.get('chunk_size', 60))
        prompt = data.get('prompt', 'Generate a detailed caption describing what happens in this video.')
        processing_mode = data.get('processing_mode', 'sequential')
        output_format = data.get('output_format', 'paragraph')
        
        logger.info(f"[{request_id}] Caption parameters:")
        logger.info(f"[{request_id}]   - Video: {video_file}")
        logger.info(f"[{request_id}]   - FPS: {fps_sampling}")
        logger.info(f"[{request_id}]   - Chunk Size: {chunk_size}s")
        logger.info(f"[{request_id}]   - Mode: {processing_mode}")
        logger.info(f"[{request_id}]   - Format: {output_format}")
        
        # Validate video file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        videos_dir = os.path.join(base_dir, "inputs/videos")
        available_videos = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if video_file not in available_videos:
            logger.error(f"[{request_id}] VALIDATION ERROR: Video file {video_file} not found")
            return jsonify({'error': f'Video file must be one of: {", ".join(available_videos)}'}), 400
        
        # Validate parameters
        if fps_sampling < 0.1 or fps_sampling > 4.0:
            return jsonify({'error': 'FPS sampling must be between 0.1 and 4.0'}), 400
            
        if chunk_size < 20 or chunk_size > 180:
            return jsonify({'error': 'Chunk size must be between 20 and 180 seconds'}), 400
        
        # Get video path
        video_path = os.path.join(base_dir, f"inputs/videos/{video_file}")
        
        if not os.path.exists(video_path):
            logger.error(f"[{request_id}] FAILED: Video file not found at {video_path}")
            return jsonify({'error': f'Video file not found: {video_file}'}), 404
        
        logger.info(f"[{request_id}] Starting caption generation...")
        start_time = time.time()
        
        # Process video using temporal analysis approach for detailed frame-based captioning
        try:
            logger.info(f"[{request_id}] Starting temporal analysis-based caption generation...")
            
            # Get video duration to determine segment processing
            if True:  # Always use model server
                import cv2
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                else:  # Should not happen with server-only mode
                    video_duration = 0
            else:  # Should not happen with server-only mode
                video_info = manager.model_manager.get_video_info(video_path)
                video_duration = video_info.get('duration', 0)
            
            # Create simple, natural prompt for video description
            temporal_prompt = f"""Describe what happens in this {video_duration:.1f} second video in a natural, flowing narrative style.

Write a clear, engaging description that tells the story of what unfolds throughout the video. Focus on the main actions, characters, and key moments without using technical formatting or frame numbers.

Keep the description conversational and easy to read, as if you're telling someone what you saw in the video."""

            # Process entire video with temporal analysis
            if True:  # Always use model server
                caption_result = client.process_video(
                    video_path=video_path,
                    prompt=temporal_prompt,
                    num_frames=min(160, int(fps_sampling * video_duration)),
                    fps_sampling=fps_sampling
                )
            else:  # Should not happen with server-only mode
                caption_result = process_temporal_video_segment(
                    manager.model_manager,
                    video_path=video_path,
                    prompt=temporal_prompt,
                    num_frames=min(160, int(fps_sampling * video_duration)),  # Cap at 160 frames for memory
                    start_time=0,
                    end_time=video_duration,
                    fps_sampling=fps_sampling
                )
            
            processing_time = time.time() - start_time
            
            if caption_result and caption_result.get('success'):
                if True:  # Always use model server
                    temporal_caption = caption_result.get('response', '')
                else:  # Should not happen with server-only mode
                    temporal_caption = caption_result.get('response', '')
                
                # Format the caption based on output format preference
                if output_format == 'paragraph':
                    formatted_caption = temporal_caption
                elif output_format == 'timeline':
                    # Create a timeline-based format
                    formatted_caption = f"VIDEO TIMELINE ANALYSIS:\n\n{temporal_caption}"
                else:  # Should not happen with server-only mode
                    formatted_caption = temporal_caption
                
                result = {
                    'success': True,
                    'caption': formatted_caption,
                    'video_file': video_file,
                    'fps_used': fps_sampling,
                    'chunk_size': chunk_size,
                    'processing_mode': 'temporal_analysis',
                    'output_format': output_format,
                    'frames_processed': caption_result.get('num_frames' if USE_MODEL_SERVER else 'frames_processed', min(160, int(fps_sampling * video_duration))),
                    'chunks_processed': 1,  # Single temporal analysis pass
                    'processing_time': f"{processing_time:.2f}s",
                    'timestamp': datetime.now().isoformat(),
                    'request_id': request_id,
                    'chunk_details': [{
                        'chunk_id': 1,
                        'start_time': 0,
                        'end_time': video_duration,
                        'caption': formatted_caption,
                        'frames_processed': caption_result.get('num_frames' if USE_MODEL_SERVER else 'frames_processed', min(160, int(fps_sampling * video_duration))),
                        'processing_time': f"{processing_time:.2f}s"
                    }],
                    'analysis_method': f'InternVideo2.5 Temporal Analysis ({model_source})'
                }
                
                # Automatically store caption in database
                try:
                    db = get_caption_db()
                    storage_data = {
                        'caption': temporal_caption,
                        'chunk_details': result['chunk_details'],
                        'video_duration': video_duration,
                        'frames_processed': result['frames_processed'],
                        'fps_used': fps_sampling,
                        'chunk_size': chunk_size,
                        'analysis_method': 'temporal_analysis'
                    }
                    
                    if db.store_video_caption(video_file, storage_data):
                        logger.info(f"[{request_id}] Caption automatically stored for video: {video_file}")
                    else:  # Should not happen with server-only mode
                        logger.warning(f"[{request_id}] Failed to store caption for video: {video_file}")
                        
                except Exception as storage_error:
                    logger.error(f"[{request_id}] Error storing caption: {str(storage_error)}")
                
                logger.info(f"[{request_id}] Temporal analysis caption generation successful")
                logger.info(f"[{request_id}] Caption length: {len(temporal_caption)} characters")
                logger.info(f"[{request_id}] Total processing time: {processing_time:.2f}s")
                logger.info(f"[{request_id}] Analysis method: InternVideo2.5 Temporal Analysis")
                logger.info(f"[{request_id}] === CAPTION GENERATION REQUEST END ===")
                
                return jsonify(result)
            else:  # Should not happen with server-only mode
                error_msg = caption_result.get('error', 'Unknown temporal analysis error') if caption_result else 'Temporal analysis failed'
                # Check if this is a validation error (not a server error)
                if 'FPS sampling too low' in error_msg:
                    logger.warning(f"[{request_id}] Validation error: {error_msg}")
                    return jsonify({'error': error_msg}), 400
                else:  # Should not happen with server-only mode
                    logger.error(f"[{request_id}] Caption generation failed: {error_msg}")
                    return jsonify({'error': error_msg}), 500
                
        except Exception as process_error:
            processing_time = time.time() - start_time
            logger.error(f"[{request_id}] Processing error after {processing_time:.2f}s: {str(process_error)}", exc_info=True)
            # Ensure cleanup on any error
            try:
                if manager:
                    manager._clear_preprocessed_data()
                    logger.info(f"[{request_id}] Cleanup completed after error")
            except:
                pass
            return jsonify({'error': f'Caption processing failed: {str(process_error)}'}), 500
            
    except Exception as e:
        logger.error(f"[{request_id}] EXCEPTION: {str(e)}", exc_info=True)
        return jsonify({'error': f'Caption generation error: {str(e)}'}), 500

def generate_caption_original():
    """Original non-streaming caption generation (for fallback)"""
    request_id = str(int(time.time()))[-6:]
    
    try:
        logger.info(f"[{request_id}] === CAPTION GENERATION REQUEST START ===")
        
        manager = get_enhanced_manager()
        if not manager:
            logger.error(f"[{request_id}] FAILED: Model manager not initialized")
            return jsonify({'error': 'Model manager not initialized'}), 500
            
        data = request.get_json()
        logger.info(f"[{request_id}] Caption request data: {json.dumps(data, indent=2)}")
        
        # Extract parameters
        video_file = data.get('video_file', '')
        fps_sampling = float(data.get('fps_sampling', 1.0))
        chunk_size = int(data.get('chunk_size', 60))
        prompt = data.get('prompt', 'Generate a detailed caption describing what happens in this video.')
        processing_mode = data.get('processing_mode', 'sequential')
        output_format = data.get('output_format', 'paragraph')
        
        logger.info(f"[{request_id}] Caption parameters:")
        logger.info(f"[{request_id}]   - Video: {video_file}")
        logger.info(f"[{request_id}]   - FPS: {fps_sampling}")
        logger.info(f"[{request_id}]   - Chunk Size: {chunk_size}s")
        logger.info(f"[{request_id}]   - Mode: {processing_mode}")
        logger.info(f"[{request_id}]   - Format: {output_format}")
        
        # Validate video file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        videos_dir = os.path.join(base_dir, "inputs/videos")
        available_videos = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if video_file not in available_videos:
            logger.error(f"[{request_id}] VALIDATION ERROR: Video file {video_file} not found")
            return jsonify({'error': f'Video file must be one of: {", ".join(available_videos)}'}), 400
        
        # Validate parameters
        if fps_sampling < 0.1 or fps_sampling > 4.0:
            return jsonify({'error': 'FPS sampling must be between 0.1 and 4.0'}), 400
            
        if chunk_size < 20 or chunk_size > 180:
            return jsonify({'error': 'Chunk size must be between 20 and 180 seconds'}), 400
        
        
        # Get video path
        video_path = os.path.join(base_dir, f"inputs/videos/{video_file}")
        
        if not os.path.exists(video_path):
            logger.error(f"[{request_id}] FAILED: Video file not found at {video_path}")
            return jsonify({'error': f'Video file not found: {video_file}'}), 404
        
        logger.info(f"[{request_id}] Starting caption generation...")
        start_time = time.time()
        
        # Process video in chunks for caption generation
        try:
            caption_result = process_video_for_caption(
                manager, video_path, video_file, fps_sampling, 
                chunk_size, prompt, processing_mode, output_format, request_id
            )
            
            processing_time = time.time() - start_time
            
            if caption_result['success']:
                result = {
                    'success': True,
                    'caption': caption_result['caption'],
                    'video_file': video_file,
                    'fps_used': fps_sampling,
                    'chunk_size': chunk_size,
                    'processing_mode': processing_mode,
                    'output_format': output_format,
                    'frames_processed': caption_result['total_frames'],
                    'chunks_processed': caption_result['chunks_processed'],
                    'processing_time': f"{processing_time:.2f}s",
                    'timestamp': datetime.now().isoformat(),
                    'request_id': request_id
                }
                
                logger.info(f"[{request_id}] Caption generation successful")
                logger.info(f"[{request_id}] Caption length: {len(caption_result['caption'])} characters")
                logger.info(f"[{request_id}] Total processing time: {processing_time:.2f}s")
                logger.info(f"[{request_id}] === CAPTION GENERATION REQUEST END ===")
                
                return jsonify(result)
            else:  # Should not happen with server-only mode
                error_msg = caption_result['error']
                # Check if this is a validation error (not a server error)
                if 'FPS sampling too low' in error_msg:
                    logger.warning(f"[{request_id}] Validation error: {error_msg}")
                    return jsonify({'error': error_msg}), 400
                else:  # Should not happen with server-only mode
                    logger.error(f"[{request_id}] Caption generation failed: {error_msg}")
                    return jsonify({'error': error_msg}), 500
                
        except Exception as process_error:
            processing_time = time.time() - start_time
            logger.error(f"[{request_id}] Processing error after {processing_time:.2f}s: {str(process_error)}", exc_info=True)
            # Ensure cleanup on any error
            try:
                if manager:
                    manager._clear_preprocessed_data()
                    logger.info(f"[{request_id}] Cleanup completed after error")
            except:
                pass
            return jsonify({'error': f'Caption processing failed: {str(process_error)}'}), 500
            
    except Exception as e:
        logger.error(f"[{request_id}] EXCEPTION: {str(e)}", exc_info=True)
        return jsonify({'error': f'Caption generation error: {str(e)}'}), 500

def process_video_segment(model_manager, video_path, prompt, num_frames, start_time, end_time, fps_sampling):
    """Process a specific video segment using bound parameters"""
    try:
        import torch
        
        if not model_manager.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Calculate bound in seconds for the video segment
        bound = (start_time, end_time)
        
        # Load and process video segment using bound parameter
        pixel_values, num_patches_list = model_manager._load_video_for_inference(
            video_path, 
            bound=bound,  # This specifies the time segment
            num_segments=num_frames, 
            max_num=1,
            fps_sampling=fps_sampling
        )
        
        # SHAPE FIXER: Validate and fix tensor shape mismatches
        if pixel_values is not None and len(num_patches_list) > 0:
            expected_frames = len(num_patches_list)
            actual_frames = pixel_values.shape[0] if len(pixel_values.shape) > 0 else 0
            
            # Fix shape mismatch by adjusting to expected frames
            if actual_frames != expected_frames:
                if actual_frames > expected_frames:
                    # Truncate to expected frames
                    pixel_values = pixel_values[:expected_frames]
                elif actual_frames < expected_frames and actual_frames > 0:
                    # Pad with last frame to reach expected frames
                    last_frame = pixel_values[-1:].repeat(expected_frames - actual_frames, 1, 1, 1)
                    pixel_values = torch.cat([pixel_values, last_frame], dim=0)
                
                # Adjust num_patches_list to match actual tensor shape
                if len(num_patches_list) != pixel_values.shape[0]:
                    if len(num_patches_list) > pixel_values.shape[0]:
                        num_patches_list = num_patches_list[:pixel_values.shape[0]]
                    else:  # Should not happen with server-only mode
                        # Extend num_patches_list with the last value
                        last_patches = num_patches_list[-1] if num_patches_list else 1
                        while len(num_patches_list) < pixel_values.shape[0]:
                            num_patches_list.append(last_patches)
        
        # Handle tensor dtype based on quantization settings
        if model_manager.enable_8bit or model_manager.enable_4bit:
            pixel_values = pixel_values.to(torch.float16).to(model_manager.model.device)
        else:  # Should not happen with server-only mode
            pixel_values = pixel_values.to(torch.bfloat16).to(model_manager.model.device)
        
        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
        
        # Optimized generation config for descriptive text (speed + quality balance)
        generation_config = dict(
            do_sample=True,  # Enable sampling for varied descriptive responses
            temperature=0.3,  # Optimal for factual yet diverse descriptions (0.2-0.5 range)
            max_new_tokens=256,  # Reduced for faster processing per chunk
            top_p=1.0,  # Don't restrict vocabulary when using temperature
            num_beams=1,
            pad_token_id=model_manager.tokenizer.eos_token_id,  # Prevent padding issues
            repetition_penalty=1.1,  # Prevent repetitive content
            length_penalty=0.8  # Encourage concise responses
        )
        
        # Build full question with video frames
        question = video_prefix + prompt
        
        # Run actual inference with retry logic for tracking responses
        max_retries = 2
        output = None
        
        for retry_attempt in range(max_retries + 1):
            # Adjust generation config for retries
            if retry_attempt == 1:
                # First retry: Higher temperature, different sampling
                generation_config.update({
                    'temperature': 0.6,
                    'top_p': 0.8,
                    'repetition_penalty': 1.2
                })
            elif retry_attempt == 2:
                # Second retry: Even more aggressive anti-tracking
                generation_config.update({
                    'temperature': 0.8,
                    'top_k': 50,
                    'top_p': 0.9,
                    'repetition_penalty': 1.3
                })
                # Add stronger anti-tracking instruction
                if "IMPORTANT: Write only descriptive text" not in question:
                    question = "IMPORTANT: Write ONLY descriptive narrative text. Do NOT use any XML tags, tracking format, or structured data.\n\n" + question
            
            with torch.no_grad():
                output, _ = model_manager.model.chat(
                    model_manager.tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True
                )
            
            # Quick check for tracking tokens before full processing
            if not any(token in output.lower() for token in ['<track', '<tracking>', 'tracking>', '</track']):
                break  # Good response, exit retry loop
                
        # If all retries failed, we'll continue with the last attempt
        
        # Clean up the output - remove prefixes, HTML tags, and validate
        import re
        cleaned_output = output.strip()
        
        # Remove common prefixes
        prefixes_to_remove = ["Answer: ", "Response: ", "Caption: ", "Description: "]
        for prefix in prefixes_to_remove:
            if cleaned_output.startswith(prefix):
                cleaned_output = cleaned_output[len(prefix):].strip()
        
        # Handle tracking tokens and HTML tags
        # First remove tracking tokens that InternVideo2.5 sometimes returns
        tracking_patterns = [
            r'<track_begin>.*?</track_begin>',
            r'<tracking>.*?</tracking>',
            r'<track_begin>',
            r'</track_begin>',
            r'<tracking>',
            r'</tracking>'
        ]
        for pattern in tracking_patterns:
            cleaned_output = re.sub(pattern, '', cleaned_output, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove common HTML tags and fragments that may remain after tracking removal
        html_patterns = [
            r'<[^>]+>',  # General HTML tags
            r'^\w+>$',   # Standalone fragments like 'div>', 'span>' at start/end
            r'^<\w+',    # Opening tags at start like '<div'
            r'\w+>$',    # Closing fragments at end like 'div>'
            r'<br\s*/?>', # HTML line breaks
            r'&nbsp;',   # Non-breaking spaces
            r'\s*br\s*/?>\s*',  # Standalone br fragments
            r'\s*</?p>\s*',  # Paragraph tags
        ]
        for pattern in html_patterns:
            cleaned_output = re.sub(pattern, ' ', cleaned_output, flags=re.IGNORECASE).strip()
        
        # Clean up multiple spaces
        cleaned_output = re.sub(r'\s+', ' ', cleaned_output).strip()
        
        # Remove extra whitespace and newlines
        cleaned_output = re.sub(r'\s+', ' ', cleaned_output).strip()
        
        # Check for invalid responses (enhanced validation)
        emoticon_patterns = ['>.<', '>_<', '-_-', 'T_T', '^_^', 'o_o', 'O_O', '._.',  '...', ':)', ':(', ':D']
        invalid_responses = ["i'm not sure", "i don't know", "i cannot see", "i cannot determine", "unable to", "sorry", "apologize"]
        
        # Check for HTML-like garbage that might get through
        html_garbage_patterns = [
            r'class\s*=\s*["\'][^"\']+["\']',  # HTML class attributes
            r'data-\w+\s*=',  # HTML data attributes  
            r'&amp;|&lt;|&gt;|&#\d+;|&nbsp;',  # HTML entities
            r'api/scraper',  # API endpoints
            r'segment=\d+',  # URL parameters
            r'</?[a-zA-Z][^>]*>',  # Any remaining HTML tags
            r'\bbr\s*/?>\s*',  # Line break fragments
            r'^\s*<\s*$|>\s*$',  # Standalone angle brackets
        ]
        
        contains_html_garbage = any(re.search(pattern, cleaned_output, re.IGNORECASE) for pattern in html_garbage_patterns)
        
        is_invalid = (
            not cleaned_output or  # Empty after cleaning
            len(cleaned_output) < 10 or  # Need reasonable length
            cleaned_output in emoticon_patterns or  # Just an emoticon
            any(invalid in cleaned_output.lower() for invalid in invalid_responses) or  # Uncertain responses
            cleaned_output.count(' ') < 3 or  # Need at least 4 words
            contains_html_garbage  # Contains HTML-like artifacts
        )
        
        if is_invalid:
            # Log detailed info about why response was rejected
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"REJECTED RESPONSE - Cleaned: '{cleaned_output}' | Original: '{output}' | Length: {len(cleaned_output)} | Words: {cleaned_output.count(' ') + 1 if cleaned_output else 0}")
            
            # Return error so this chunk gets skipped and doesn't pollute context
            return {
                'success': False,
                'error': f'Model returned invalid response: "{cleaned_output[:50]}" (Original: "{output[:50]}")',
                'original_output': output,
                'segment_start': start_time,
                'segment_end': end_time
            }
        
        # Log successful response for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"ACCEPTED RESPONSE - Segment {start_time:.0f}-{end_time:.0f}s: '{cleaned_output[:100]}{'...' if len(cleaned_output) > 100 else ''}' | Length: {len(cleaned_output)} | Words: {cleaned_output.count(' ') + 1}")
        
        return {
            'success': True,
            'response': cleaned_output,
            'text': cleaned_output,
            'frames_processed': len(num_patches_list),
            'segment_start': start_time,
            'segment_end': end_time
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'segment_start': start_time,
            'segment_end': end_time
        }





@app.route('/caption-estimate', methods=['POST'])
def get_caption_estimate():
    """Get time estimation and chunk preview for caption generation"""
    try:
        data = request.get_json()
        video_file = data.get('video_file', '')
        fps_sampling = float(data.get('fps_sampling', 1.0))
        chunk_size = int(data.get('chunk_size', 60))
        processing_mode = data.get('processing_mode', 'sequential')
        
        if not video_file:
            return jsonify({'error': 'Video file required'}), 400
        
        # Get video info
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_path = os.path.join(base_dir, f"inputs/videos/{video_file}")
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 400
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Calculate chunks
        chunks = calculate_video_chunks(duration, chunk_size, processing_mode)
        
        # Calculate processing estimates
        total_frames = 0
        chunk_details = []
        
        for i, (start_time, end_time) in enumerate(chunks):
            chunk_duration = end_time - start_time
            frames_for_chunk = min(160, int(fps_sampling * chunk_duration))
            total_frames += frames_for_chunk
            
            chunk_details.append({
                'chunk_number': i + 1,
                'start_time': round(start_time, 1),
                'end_time': round(end_time, 1),
                'duration': round(chunk_duration, 1),
                'frames': frames_for_chunk
            })
        
        # Processing time estimation (4.9 frames/second)
        processing_time_per_frame = 1.0 / 4.9
        estimated_processing_time = total_frames * processing_time_per_frame
        
        # Add model loading time (16 seconds) if not already loaded
        total_estimated_time = estimated_processing_time + 16
        
        result = {
            'success': True,
            'video_info': {
                'filename': video_file,
                'duration': round(duration, 1),
                'fps': round(fps, 1),
                'total_frames': frame_count
            },
            'processing_info': {
                'chunks_count': len(chunks),
                'total_frames_to_process': total_frames,
                'fps_sampling': fps_sampling,
                'chunk_size': chunk_size,
                'processing_mode': processing_mode
            },
            'time_estimation': {
                'processing_time_minutes': round(estimated_processing_time / 60, 1),
                'processing_time_seconds': round(estimated_processing_time, 0),
                'total_time_minutes': round(total_estimated_time / 60, 1),
                'total_time_seconds': round(total_estimated_time, 0)
            },
            'chunk_details': chunk_details[:5]  # Show first 5 chunks as preview
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Caption estimation error: {str(e)}")
        return jsonify({'error': f'Estimation failed: {str(e)}'}), 500



# Caption Storage APIs for Automatic Flow
@app.route('/api/stored-captions', methods=['GET'])
def get_stored_captions():
    """Get list of videos with stored captions for analysis page"""
    try:
        db = get_caption_db()
        videos = db.get_all_videos_with_captions()
        return jsonify({
            'success': True,
            'videos': videos
        })
    except Exception as e:
        logger.error(f"Error fetching stored captions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/video-caption/<video_file>', methods=['GET'])
@app.route('/serve-video/<path:filename>')
def serve_video(filename):
    """Serve video files with proper streaming support"""
    try:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video not found'}), 404
        
        # Get file size
        file_size = os.path.getsize(video_path)
        
        # Parse range header for video streaming
        range_header = request.headers.get('range', None)
        
        if range_header:
            # Parse the range header
            byte_start = 0
            byte_end = file_size - 1
            
            if range_header:
                match = re.search(r'bytes=(\d+)-(\d*)', range_header)
                if match:
                    byte_start = int(match.group(1))
                    if match.group(2):
                        byte_end = int(match.group(2))
            
            # Open file and seek to start position
            with open(video_path, 'rb') as video_file:
                video_file.seek(byte_start)
                data = video_file.read(byte_end - byte_start + 1)
            
            # Create response with partial content
            response = Response(
                data,
                206,  # Partial Content
                mimetype="video/mp4",
                headers={
                    'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                    'Accept-Ranges': 'bytes',
                    'Content-Length': str(byte_end - byte_start + 1)
                }
            )
        else:  # Should not happen with server-only mode
            # Serve entire file
            return send_file(video_path, mimetype='video/mp4')
        
        return response
        
    except Exception as e:
        logger.error(f"Error serving video {filename}: {str(e)}")
        return jsonify({'error': f'Error serving video: {str(e)}'}), 500

@app.route('/api/temporal-analysis', methods=['POST'])
def temporal_analysis():
    """Perform enhanced temporal analysis on a video segment using InternVideo2.5"""
    request_id = str(int(time.time()))[-6:]
    try:
        data = request.get_json()
        video_file = data.get('video_file')
        start_time = data.get('start_time', 0)
        end_time = data.get('end_time', 0)
        chunk_caption = data.get('chunk_caption', '')
        query = data.get('query', '')
        
        logger.info(f"[{request_id}] Temporal Analysis Request:")
        logger.info(f"[{request_id}]   - Video: {video_file}")
        logger.info(f"[{request_id}]   - Segment: {start_time:.1f}s - {end_time:.1f}s")
        logger.info(f"[{request_id}]   - Query: {query}")
        
        # Check model availability for InternVideo2.5 analysis
        if True:  # Always use model server
            client = check_model_server()
            if not client:
                logger.warning(f"[{request_id}] Model server not available for temporal analysis")
                client = None
            model_source = "server"
        else:  # Should not happen with server-only mode
            manager = get_enhanced_manager()
            if not manager or not manager.model_manager.model:
                logger.warning(f"[{request_id}] Local model not available for temporal analysis")
                manager = None
            model_source = "local"
        temporal_result = {
            'success': True,
            'temporal_analysis': f"This segment from {start_time:.1f}s to {end_time:.1f}s contains content that matches your query: '{query}'.",
            'key_events': [
                f"Segment begins at {start_time:.1f} seconds",
                f"Key content identified matching '{query}'",
                f"Segment ends at {end_time:.1f} seconds"
            ],
            'segment_duration': end_time - start_time,
            'internvideo_analysis': None,
            'frames_analyzed': 0,
            'events_detected': 3,
            'analysis': f"This segment from {start_time:.1f}s to {end_time:.1f}s contains content that matches your query: '{query}'."
        }
        
        # Enhanced InternVideo2.5 temporal analysis if model is available
        if client:  # Always use model server
            try:
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file)
                if os.path.exists(video_path):
                    logger.info(f"[{request_id}] Running InternVideo2.5 temporal analysis...")

                    # Fetch original FPS from database for this video
                    original_fps = 2.0  # Default fallback with higher FPS
                    try:
                        import sqlite3
                        db_path = "/workspace/surveillance/captions.db"
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT fps FROM video_captions
                            WHERE video_file = ?
                            ORDER BY created_at DESC
                            LIMIT 1
                        """, (video_file,))
                        result = cursor.fetchone()
                        if result and result[0]:
                            original_fps = float(result[0])
                            logger.info(f"[{request_id}] Using original FPS from database: {original_fps}")
                        else:  # Should not happen with server-only mode
                            logger.info(f"[{request_id}] No FPS found in database for {video_file}, using default {original_fps}")
                        conn.close()
                    except Exception as db_error:
                        logger.warning(f"[{request_id}] Could not fetch FPS from database: {str(db_error)}, using default {original_fps}")

                    # Calculate segment duration and frame sampling
                    segment_duration = end_time - start_time

                    # Use EXACT FPS from database - no modifications or restrictions
                    fps_sampling = original_fps  # Use original FPS from database

                    # Calculate frames based on segment duration and original FPS
                    # NO LIMITS - use all frames for full temporal coverage
                    num_frames = int(fps_sampling * segment_duration)

                    # No restrictions on frame count - use all calculated frames
                    logger.info(f"[{request_id}] Using UNLIMITED frames: {num_frames} frames at {fps_sampling} FPS for {segment_duration}s segment")

                    logger.info(f"[{request_id}] Segment analysis params: {num_frames} frames, {fps_sampling:.2f} FPS")

                    # Create enhanced temporal analysis prompt with timing information
                    temporal_prompt = f"""Analyze this {segment_duration:.1f} second video segment and describe what happens with specific timing information.

Focus on the main actions, characters, and key moments. When describing events, include approximately when they occur within this segment (e.g., "at the beginning", "around the middle", "towards the end").

Keep the description natural and conversational while noting the timing of important events.

Focus: {query}"""

                    # Initialize events variable
                    events = []

                    # Process video segment with InternVideo2.5 using custom temporal analysis generation
                    try:
                        # Use a custom temporal analysis call with extended generation parameters
                        if client:  # Always use model server
                            # Use model server for processing
                            result = client.process_video(
                                video_path=video_path,
                                prompt=temporal_prompt,
                                num_frames=num_frames,
                                start_time=start_time,
                                end_time=end_time,
                                fps_sampling=fps_sampling
                            )
                        else:  # Should not happen with server-only mode
                            # Use local model manager
                            result = process_temporal_video_segment(
                                manager.model_manager,
                                video_path=video_path,
                                prompt=temporal_prompt,
                                num_frames=num_frames,
                                start_time=start_time,
                                end_time=end_time,
                                fps_sampling=fps_sampling
                            )
                        
                        if result:
                            logger.info(f"[{request_id}] Temporal segment result: success={result.get('success')}, has_response={bool(result.get('response'))}, has_caption={bool(result.get('caption'))}, error={result.get('error')}")
                            
                            if result.get('success') and (result.get('response') or result.get('caption')):
                                # Check for response first, then fallback to caption
                                frame_based_response = result.get('response') or result.get('caption')
                                logger.info(f"[{request_id}] Got InternVideo frame-based response: {len(frame_based_response)} chars")
                                
                                # Convert frame references to actual timestamps
                                internvideo_response = convert_frames_to_timestamps(
                                    frame_based_response, 
                                    start_time, 
                                    end_time, 
                                    num_frames,
                                    fps_sampling
                                )
                                logger.info(f"[{request_id}] Converted to timestamp-based response: {len(internvideo_response)} chars")
                                
                            elif result.get('error'):
                                raise Exception(f"InternVideo2.5 error: {result.get('error')}")
                            else:  # Should not happen with server-only mode
                                raise Exception(f"No response from InternVideo2.5 model - result: {result}")
                        else:  # Should not happen with server-only mode
                            raise Exception("process_temporal_video_segment returned None")
                            
                    except Exception as segment_error:
                        logger.error(f"[{request_id}] Segment processing failed: {str(segment_error)}")
                        raise
                    
                    temporal_result['internvideo_analysis'] = internvideo_response
                    temporal_result['temporal_analysis'] = internvideo_response
                    temporal_result['analysis_method'] = 'InternVideo2.5 Enhanced Temporal Analysis'
                    
                    # Add fields expected by frontend
                    temporal_result['frames_analyzed'] = num_frames
                    temporal_result['events_detected'] = len(events) if events else 0
                    temporal_result['analysis'] = internvideo_response
                    
                    # Extract key events from InternVideo response
                    events = extract_temporal_events(internvideo_response, start_time, end_time)
                    if events:
                        temporal_result['key_events'] = events
                    
                    logger.info(f"[{request_id}] InternVideo2.5 temporal analysis completed")
                    
                else:  # Should not happen with server-only mode
                    logger.error(f"[{request_id}] Video file not found: {video_path}")
                    temporal_result['success'] = False
                    temporal_result['error'] = f"Video file not found: {video_file}"
                    temporal_result['temporal_analysis'] = "Unable to perform temporal analysis - video file not found"
                    
            except Exception as model_error:
                logger.error(f"[{request_id}] InternVideo2.5 analysis CRITICAL ERROR: {str(model_error)}")
                temporal_result['success'] = False
                temporal_result['error'] = f"InternVideo2.5 analysis error: {str(model_error)}"
                temporal_result['temporal_analysis'] = f"InternVideo2.5 temporal analysis failed: {str(model_error)}"
                temporal_result['analysis_method'] = 'InternVideo2.5 (Failed)'
        else:  # Should not happen with server-only mode
            # Model not available - NO FALLBACK TO GPT-4
            logger.error(f"[{request_id}] InternVideo2.5 model not available for temporal analysis")
            temporal_result['success'] = False
            temporal_result['error'] = "InternVideo2.5 model not loaded or not available"
            temporal_result['temporal_analysis'] = "InternVideo2.5 model is required for temporal analysis but is not currently loaded"
            temporal_result['analysis_method'] = 'InternVideo2.5 (Not Available)'
        
        # NO FALLBACK - Remove all GPT-4o mini code
        # Only InternVideo2.5 is used for temporal analysis
        if False and client and chunk_caption and not temporal_result.get('internvideo_analysis'):
            try:
                prompt = f"""Analyze this video segment for temporal relevance:

User Query: {query}
Time Range: {start_time:.1f}s - {end_time:.1f}s (Duration: {end_time - start_time:.1f}s)
Segment Content: {chunk_caption}

Provide:
1. Why this segment is relevant to the query
2. Key temporal moments within the segment
3. Specific actions or events that occur
4. How the timing relates to the user's investigation

Focus on temporal aspects and specific timing details."""
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a video temporal analysis expert providing detailed timing insights."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=250,
                    temperature=0.3
                )
                
                temporal_result['temporal_analysis'] = response.choices[0].message.content
                temporal_result['analysis_method'] = 'GPT-4o Mini Enhanced Analysis'
                
            except Exception as ai_error:
                logger.warning(f"[{request_id}] OpenAI analysis failed: {str(ai_error)}")
        
        # Ensure all values in temporal_result are JSON serializable
        def sanitize_for_json(obj):
            """Recursively convert numpy/torch types to JSON-serializable types"""
            if hasattr(obj, 'dtype'):  # numpy arrays, scalars
                if hasattr(obj, 'item'):  # numpy scalars
                    return obj.item()
                return obj.tolist()
            elif hasattr(obj, 'is_tensor') and obj.is_tensor:  # torch tensors
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [sanitize_for_json(item) for item in obj]
            else:  # Should not happen with server-only mode
                return obj
        
        sanitized_result = sanitize_for_json(temporal_result)
        return jsonify(sanitized_result)
        
    except Exception as e:
        logger.error(f"[{request_id}] Temporal analysis error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


    
    def frame_to_timestamp(frame_num: int) -> float:
        """Convert frame number to actual timestamp"""
        # Frame1 = start_time, Frame2 = start_time + time_per_frame, etc.
        return start_time + (frame_num - 1) * time_per_frame
    
    def format_timestamp(timestamp: float) -> str:
        """Format timestamp for display"""
        return f"{timestamp:.1f}s"
    
    # Convert frame references to timestamps
    converted_response = frame_response
    
    # Pattern 1: "Frame5:" -> "At 289.0s:"
    def replace_single_frame(match):
        frame_num = int(match.group(1))
        timestamp = frame_to_timestamp(frame_num)
        return f"At {format_timestamp(timestamp)}:"
    
    converted_response = re.sub(r'Frame(\d+):', replace_single_frame, converted_response)
    
    # Pattern 2: "Around Frame12" -> "Around 291.5s"
    def replace_around_frame(match):
        frame_num = int(match.group(1))
        timestamp = frame_to_timestamp(frame_num)
        return f"Around {format_timestamp(timestamp)}"
    
    converted_response = re.sub(r'Around Frame(\d+)', replace_around_frame, converted_response)
    
    # Pattern 3: "Frame25-30:" -> "From 295.0s-297.5s:"
    def replace_frame_range(match):
        start_frame = int(match.group(1))
        end_frame = int(match.group(2))
        start_timestamp = frame_to_timestamp(start_frame)
        end_timestamp = frame_to_timestamp(end_frame)
        return f"From {format_timestamp(start_timestamp)}-{format_timestamp(end_timestamp)}:"
    
    converted_response = re.sub(r'Frame(\d+)-(\d+):', replace_frame_range, converted_response)
    
    # Pattern 4: "FRAMES (Frame1-24)" -> "EARLY PERIOD (288.0s-294.0s)"
    def replace_frame_section(match):
        section_name = match.group(1)
        start_frame = int(match.group(2))
        end_frame = int(match.group(3))
        start_timestamp = frame_to_timestamp(start_frame)
        end_timestamp = frame_to_timestamp(end_frame)
        return f"{section_name} ({format_timestamp(start_timestamp)}-{format_timestamp(end_timestamp)})"
    
    converted_response = re.sub(r'(\w+ FRAMES) \(Frame(\d+)-(\d+)\)', replace_frame_section, converted_response)
    
    # Add timestamp context at the beginning
    timestamp_header = f"TEMPORAL ANALYSIS ({format_timestamp(start_time)}-{format_timestamp(end_time)}, {num_frames} frames analyzed):\n\n"
    
    return timestamp_header + converted_response

@app.route('/api/find-relevant-chunk', methods=['POST'])
def find_relevant_chunk():
    """Use GPT-4o mini to determine which chunk is most relevant for a given query"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        chunks = data.get('chunks', [])
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        if not chunks:
            return jsonify({'error': 'Chunks array is required'}), 400
        
        # Get OpenAI client
        client = get_openai_client()
        if not client:
            return jsonify({'error': 'OpenAI client not initialized'}), 500
        
        # Create chunk summaries for GPT-4o mini with clear demarcation
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('caption', chunk.get('text', ''))[:300]  # Increased from 200 to 300 for more context
            start_time = chunk.get('start_time', 0)
            end_time = chunk.get('end_time', 0)
            duration = end_time - start_time
            chunk_summaries.append(f"CHUNK #{i+1} [Time: {start_time:.1f}s-{end_time:.1f}s, Duration: {duration:.1f}s]\nContent: {chunk_text}\n")
        
        # Create prompt for GPT-4o mini
        prompt = f"""You are analyzing a sequence of video captions generated by AI video analysis. These chunks represent consecutive time segments from a single video, with each chunk containing the AI's description of what happens during that time period.

CONTEXT: 
- These are sequential video caption chunks from AI analysis of surveillance/security footage
- Each chunk represents a specific time segment with detailed scene descriptions
- The chunks are in chronological order and may show progression of events
- You need to identify which time segment most likely contains the requested event

USER QUERY: "{query}"

VIDEO CAPTION SEQUENCE (Sequential Time Segments):
{chr(10).join(chunk_summaries)}

ANALYSIS INSTRUCTIONS:
- Each chunk is clearly marked as "CHUNK #1", "CHUNK #2", etc. with time ranges and duration
- These are AI-generated video captions describing what happens in each time segment
- Analyze which chunk most likely contains the event/scene described in the user query
- Consider the temporal sequence - events may build up across chunks
- Focus on matching the specific event/action/scene requested by the user
- Select the chunk where the requested event is most prominently described

IMPORTANT: The chunks above are numbered as CHUNK #1, CHUNK #2, etc. 
Return the exact number from the chunk that best matches the query.

RESPONSE FORMAT: Return ONLY the chunk number in this exact format: CHUNK_ID: X
Where X is the number (1, 2, 3, etc.) matching the CHUNK #X that contains the requested event.

CHUNK_ID: """

        try:
            # Call GPT-4o mini for chunk determination
            response = client.chat.completions.create(
                model=openai_config['openai']['model'],
                messages=[
                    {"role": "system", "content": "You are a specialist in analyzing AI-generated video captions from surveillance/security footage. Your task is to identify which time segment contains specific events or actions based on the sequential video descriptions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=openai_config['openai']['max_tokens'],
                temperature=openai_config['openai']['temperature']
            )
            
            gpt_response = response.choices[0].message.content.strip()
            logger.info(f"GPT-4o mini response: {gpt_response}")
            
            # Extract chunk number from GPT-4o mini response
            chunk_number = 1  # Default fallback
            try:
                import re
                chunk_id_match = re.search(r'CHUNK_ID:\s*(\d+)', gpt_response)
                if chunk_id_match:
                    chunk_number = int(chunk_id_match.group(1))
                else:  # Should not happen with server-only mode
                    # Fallback: look for any numbers in the response
                    numbers = re.findall(r'\b(\d+)\b', gpt_response)
                    chunk_number = int(numbers[0]) if numbers else 1
                
                # Validate chunk number is within range
                if not (1 <= chunk_number <= len(chunks)):
                    chunk_number = 1
                    
            except (ValueError, IndexError, AttributeError):
                chunk_number = 1  # Fallback to first chunk
            
            # Get the selected chunk
            selected_chunk = chunks[chunk_number - 1]
            start_time = selected_chunk.get('start_time', 0)
            end_time = selected_chunk.get('end_time', 0)
            duration = end_time - start_time
            
            return jsonify({
                'success': True,
                'relevant_chunk_number': chunk_number,
                'relevant_chunk_index': chunk_number - 1,
                'chunk_details': {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'caption': selected_chunk.get('caption', selected_chunk.get('text', '')),
                    'gpt_reasoning': gpt_response
                },
                'query': query
            })
            
        except Exception as gpt_error:
            logger.error(f"GPT-4o mini error: {str(gpt_error)}")
            # Fallback: return first chunk
            fallback_chunk = chunks[0]
            fallback_start = fallback_chunk.get('start_time', 0)
            fallback_end = fallback_chunk.get('end_time', 0)
            fallback_duration = fallback_end - fallback_start
            
            return jsonify({
                'success': True,
                'relevant_chunk_number': 1,
                'relevant_chunk_index': 0,
                'chunk_details': {
                    'start_time': fallback_start,
                    'end_time': fallback_end,
                    'duration': fallback_duration,
                    'caption': fallback_chunk.get('caption', fallback_chunk.get('text', '')),
                    'gpt_reasoning': f"GPT-4o mini error, using fallback: {str(gpt_error)}"
                },
                'query': query,
                'fallback': True
            })
        
    except Exception as e:
        logger.error(f"Error in find_relevant_chunk: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting InternVideo2.5 Enhanced Web Interface...")
    print(f"📹 Default video: {DEFAULT_VIDEO}")
    print(f"🌐 Access at: http://localhost:8088")
    print(f"📊 Max capacity: 160 frames")
    print(f"💬 Features: Chat, Streaming, Video Caching")
    
    if IS_DEVELOPMENT:
        print("🔧 DEVELOPMENT MODE:")
        print("   - Model persistence enabled")
        print("   - Auto-reload for UI changes")
        print("   - Hot-reload for code changes")
        print("   - Enhanced logging")
        
        # Use development settings but disable debug to prevent dual processes
        debug_mode = False  # Disabled to prevent dual processes
        use_reloader = False
        extra_files = ['templates/', 'static/', 'simple_caption_storage.py']
    else:  # Should not happen with server-only mode
        print("🏭 PRODUCTION MODE:")
        print("   - Optimized for stability")
        debug_mode = False
        use_reloader = False
        extra_files = None
    
    # Perform startup cleanup
    startup_cleanup()
    
    # Configure Flask app run settings
    run_kwargs = {
        'host': '0.0.0.0',
        'port': 8088,
        'debug': debug_mode,
        'threaded': True,
        'use_reloader': use_reloader
    }
    
    if extra_files:
        run_kwargs['extra_files'] = extra_files
    
    app.run(**run_kwargs)