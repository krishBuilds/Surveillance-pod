#!/usr/bin/env python3
"""
InternVideo2.5 Enhanced Web Interface
Web app for video analysis with streaming responses and chat functionality
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_template
import subprocess
import json
import os
import sys
import time
from datetime import datetime
import logging
import threading
from pathlib import Path
import torch

# Add parent directory to path to access main_enhanced.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import enhanced model manager
from main_enhanced import EnhancedModelManager
# Import simple caption storage
from simple_caption_storage import SimpleCaptionStorage

# Import OpenAI for chunk determination
from openai import OpenAI
import yaml

app = Flask(__name__)
app.config['SECRET_KEY'] = 'internvideo-surveillance-app'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default video path
DEFAULT_VIDEO = "inputs/videos/bigbuckbunny.mp4"

# Global enhanced manager instance
enhanced_manager = None

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

def get_enhanced_manager():
    """Get or create the enhanced model manager with memory cleanup"""
    global enhanced_manager
    if enhanced_manager is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logger.info("=== INITIALIZING ENHANCED MODEL MANAGER ===")
        logger.info(f"Base directory: {base_dir}")
        
        model_path = os.path.join(base_dir, "models/InternVideo2_5")
        config_path = os.path.join(base_dir, "config.yaml")
        
        logger.info(f"Model path: {model_path}")
        logger.info(f"Model path exists: {os.path.exists(model_path)}")
        logger.info(f"Config path: {config_path}")
        logger.info(f"Config path exists: {os.path.exists(config_path)}")
        
        try:
            logger.info("Creating EnhancedModelManager instance...")
            enhanced_manager = EnhancedModelManager(
                model_path=model_path,
                device="auto",
                precision="bf16",
                config_path=config_path
            )
            logger.info("EnhancedModelManager instance created successfully")
            
            logger.info("Initializing model (this may take 15-20 seconds)...")
            
            if not enhanced_manager.initialize_model():
                logger.error("FAILED to load model - initialization returned False")
                logger.error("Check model files, GPU availability, and dependencies")
                return None
                
            logger.info("SUCCESS: Enhanced Model Manager initialized")
            logger.info(f"Model device: {enhanced_manager.model_manager.device}")
            logger.info(f"Model precision: {enhanced_manager.model_manager.precision}")
            
            # Clean up any cached data while keeping model loaded
            logger.info("Cleaning up cache and preprocessed data...")
            try:
                enhanced_manager.model_manager.clear_processing_cache()
                logger.info("Cache cleared successfully")
            except Exception as cleanup_error:
                logger.warning(f"Cache cleanup warning: {str(cleanup_error)}")
            
            logger.info("=== MODEL MANAGER READY ===")
            
        except Exception as e:
            logger.error(f"EXCEPTION during manager initialization: {str(e)}", exc_info=True)
            logger.error("This could be due to:")
            logger.error("- Missing model files")
            logger.error("- GPU/CUDA issues") 
            logger.error("- Dependency problems")
            logger.error("- Memory issues")
            return None
    
    return enhanced_manager

def startup_cleanup():
    """Clean up memory and cache on app startup while keeping model loaded"""
    logger.info("=== WEBAPP STARTUP CLEANUP ===")
    try:
        # Clear any existing CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
        
        # Initialize manager and clean cache
        manager = get_enhanced_manager()
        if manager:
            logger.info("Startup cleanup completed successfully")
        else:
            logger.error("Manager initialization failed during startup")
            
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
                        else:
                            duration_str = "Unknown"
                        cap.release()
                    else:
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

@app.route('/process-video', methods=['POST'])
def process_video():
    """Process video without any initial prompt - just extract and cache frames"""
    request_id = str(int(time.time()))[-6:]  # Last 6 digits for tracking
    
    try:
        logger.info(f"[{request_id}] === VIDEO PROCESSING REQUEST START ===")
        
        manager = get_enhanced_manager()
        if not manager:
            logger.error(f"[{request_id}] FAILED: Model manager not initialized")
            return jsonify({'error': 'Model manager not initialized'}), 500
            
        data = request.get_json()
        logger.info(f"[{request_id}] Request data: {json.dumps(data, indent=2)}")
        
        # Extract parameters (no prompt needed)
        video_file = data.get('video_file', 'bigbuckbunny.mp4')
        num_frames = int(data.get('num_frames', 64))
        fps_sampling = float(data.get('fps_sampling', 0)) if data.get('fps_sampling') else None
        time_bound = float(data.get('time_bound', 0)) if data.get('time_bound') else None
        
        logger.info(f"[{request_id}] Processing parameters:")
        logger.info(f"[{request_id}]   - Video File: {video_file}")
        logger.info(f"[{request_id}]   - Frames: {num_frames}")
        logger.info(f"[{request_id}]   - FPS Sampling: {fps_sampling}")
        logger.info(f"[{request_id}]   - Time Bound: {time_bound}")
        
        # Get available video files from directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        videos_dir = os.path.join(base_dir, "inputs/videos")
        available_videos = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        # Validate video file
        if video_file not in available_videos:
            logger.error(f"[{request_id}] VALIDATION ERROR: Video file {video_file} not found in directory")
            return jsonify({'error': f'Video file must be one of: {", ".join(available_videos)}'}), 400
        
        # Validate parameters
        if num_frames < 1 or num_frames > 160:
            logger.error(f"[{request_id}] VALIDATION ERROR: Frame count {num_frames} not in range 1-160")
            return jsonify({'error': 'Frame count must be between 1 and 160'}), 400
        
        if fps_sampling and (fps_sampling < 0.1 or fps_sampling > 5.0):
            logger.error(f"[{request_id}] VALIDATION ERROR: FPS sampling {fps_sampling} not in range 0.1-5.0")
            return jsonify({'error': 'FPS sampling must be between 0.1 and 5.0'}), 400
            
        if time_bound and (time_bound < 1 or time_bound > 600):
            logger.error(f"[{request_id}] VALIDATION ERROR: Time bound {time_bound} not in range 1-600")
            return jsonify({'error': 'Time bound must be between 1 and 600 seconds'}), 400
        
        # Get video path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_path = os.path.join(base_dir, f"inputs/videos/{video_file}")
        
        logger.info(f"[{request_id}] Video path: {video_path}")
        logger.info(f"[{request_id}] Video exists: {os.path.exists(video_path)}")
        
        if not os.path.exists(video_path):
            logger.error(f"[{request_id}] FAILED: Video file not found at {video_path}")
            return jsonify({'error': f'Video file not found: {DEFAULT_VIDEO}'}), 404
        
        logger.info(f"[{request_id}] Starting video preprocessing...")
        
        # Use enhanced manager for video processing only
        start_time = time.time()
        
        # Collect processing response with detailed logging
        result = {}
        chunk_count = 0
        
        for chunk in manager.preprocess_video(
            video_path, num_frames, fps_sampling, time_bound
        ):
            chunk_count += 1
            logger.info(f"[{request_id}] Chunk {chunk_count}: {chunk.get('type', 'unknown')} - {chunk.get('message', 'no message')}")
            
            if chunk['type'] == 'video_processed':
                result = {
                    'success': True,
                    'frames_processed': chunk['frames_processed'],
                    'video_info': chunk['video_info'],
                    'message': chunk['message'],
                    'preprocessing_time': chunk['preprocessing_time']
                }
                logger.info(f"[{request_id}] SUCCESS: Video processed successfully")
                logger.info(f"[{request_id}] Frames processed: {chunk['frames_processed']}")
                logger.info(f"[{request_id}] Preprocessing time: {chunk['preprocessing_time']:.2f}s")
                logger.info(f"[{request_id}] Video info: {json.dumps(chunk.get('video_info', {}), indent=2)}")
                break
            elif chunk['type'] == 'error':
                logger.error(f"[{request_id}] PROCESSING ERROR: {chunk['message']}")
                return jsonify({
                    'error': chunk['message']
                }), 500
            elif chunk['type'] == 'status':
                logger.info(f"[{request_id}] STATUS: {chunk['message']}")
        
        processing_time = time.time() - start_time
        
        logger.info(f"[{request_id}] Total processing time: {processing_time:.2f}s")
        logger.info(f"[{request_id}] Total chunks received: {chunk_count}")
        
        # Check if we got a result
        if not result:
            logger.error(f"[{request_id}] FAILED: No processing result received after {chunk_count} chunks")
            return jsonify({'error': 'Video processing failed - no result received'}), 500
        
        result['processing_time'] = f"{processing_time:.2f}s"
        result['timestamp'] = datetime.now().isoformat()
        result['request_id'] = request_id
        
        logger.info(f"[{request_id}] Final result: {json.dumps(result, indent=2, default=str)}")
        logger.info(f"[{request_id}] === VIDEO PROCESSING REQUEST END ===")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"[{request_id}] EXCEPTION: {str(e)}", exc_info=True)
        return jsonify({'error': f'Video processing error: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat_with_video():
    """Handle chat messages with processed video and temporal analysis"""
    request_id = str(int(time.time()))[-6:]  # Last 6 digits for tracking
    
    try:
        logger.info(f"[{request_id}] === CHAT REQUEST START ===")
        
        manager = get_enhanced_manager()
        if not manager:
            logger.error(f"[{request_id}] FAILED: Model manager not initialized")
            return jsonify({'error': 'Model manager not initialized'}), 500
            
        data = request.get_json()
        logger.info(f"[{request_id}] Chat request data: {json.dumps(data, indent=2)}")
        
        prompt = data.get('message', '')
        enable_temporal = data.get('enable_temporal', True)  # Enable temporal analysis by default
        generate_snippets = data.get('generate_snippets', True) and enable_temporal  # Only generate snippets if temporal is enabled
        
        if not prompt.strip():
            logger.error(f"[{request_id}] VALIDATION ERROR: Empty prompt")
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        logger.info(f"[{request_id}] Chat query: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
        logger.info(f"[{request_id}] Full prompt length: {len(prompt)} characters")
        logger.info(f"[{request_id}] Temporal analysis enabled: {enable_temporal}")
        logger.info(f"[{request_id}] Video snippets enabled: {generate_snippets}")
        
        # Check if video is preprocessed
        session_info = manager.get_session_info()
        if session_info:
            logger.info(f"[{request_id}] Video session active:")
            logger.info(f"[{request_id}]   - Video: {session_info.get('video_path', 'N/A')}")
            logger.info(f"[{request_id}]   - Frames: {session_info.get('num_frames', 'N/A')}")
            logger.info(f"[{request_id}]   - Preprocessing time: {session_info.get('preprocessing_time', 'N/A')}")
        else:
            logger.error(f"[{request_id}] FAILED: No video session active")
            return jsonify({'error': 'No video preprocessed. Please process a video first.'}), 400
        
        logger.info(f"[{request_id}] Starting chat inference...")
        start_time = time.time()
        
        # Collect streaming response with detailed logging and temporal analysis
        response_text = ""
        metrics = {}
        temporal_data = None
        video_snippets = []
        chunk_count = 0
        
        for chunk in manager.chat_query(prompt, enable_temporal_analysis=enable_temporal):
            chunk_count += 1
            chunk_type = chunk.get('type', 'unknown')
            
            if chunk_type == 'response_chunk':
                # Log every 10th chunk to avoid spam
                if chunk_count % 10 == 1:
                    logger.info(f"[{request_id}] Response chunk {chunk_count}: '{chunk.get('content', '')}'")
            elif chunk_type == 'response_complete':
                response_text = chunk['full_content']
                metrics = chunk.get('metadata', {})
                temporal_data = chunk.get('temporal_data', {})
                
                logger.info(f"[{request_id}] SUCCESS: Chat response completed")
                logger.info(f"[{request_id}] Response length: {len(response_text)} characters")
                logger.info(f"[{request_id}] Response preview: '{response_text[:200]}{'...' if len(response_text) > 200 else ''}'")
                logger.info(f"[{request_id}] Metadata: {json.dumps(metrics, indent=2, default=str)}")
                
                # Process temporal data if available
                if temporal_data and temporal_data.get('has_temporal_references'):
                    logger.info(f"[{request_id}] Found temporal references: {len(temporal_data.get('timestamps', []))} timestamps")
                    
                    # Generate video snippets if requested and temporal data exists
                    if generate_snippets:
                        logger.info(f"[{request_id}] Generating video snippets...")
                        video_snippets = manager.generate_snippets_for_response(temporal_data, max_snippets=3)
                        logger.info(f"[{request_id}] Generated {len(video_snippets)} video snippets")
                
                break
            elif chunk_type == 'error':
                logger.error(f"[{request_id}] CHAT ERROR: {chunk['message']}")
                return jsonify({
                    'error': chunk['message']
                }), 500
            elif chunk_type == 'status':
                logger.info(f"[{request_id}] STATUS: {chunk['message']}")
            elif chunk_type == 'user_message':
                logger.info(f"[{request_id}] User message logged: '{chunk['content'][:100]}{'...' if len(chunk['content']) > 100 else ''}'")
        
        chat_time = time.time() - start_time
        logger.info(f"[{request_id}] Total chat time: {chat_time:.2f}s")
        logger.info(f"[{request_id}] Total chunks processed: {chunk_count}")
        
        # Check if we got a response
        if not response_text:
            logger.error(f"[{request_id}] FAILED: No response received after {chunk_count} chunks")
            return jsonify({'error': 'Chat failed - no response received'}), 500
        
        # Get current chat history size
        chat_history = manager.get_chat_history()
        logger.info(f"[{request_id}] Chat history size: {len(chat_history)} messages")
        
        result = {
            'success': True,
            'response': response_text,
            'metrics': {
                'inference_time': f"{metrics.get('inference_time', 0):.2f}s",
                'frames_processed': metrics.get('frames_processed', 0)
            },
            'temporal_data': temporal_data,
            'video_snippets': video_snippets,
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'total_chat_time': f"{chat_time:.2f}s",
            'chunks_processed': chunk_count,
            'chat_history_size': len(chat_history)
        }
        
        logger.info(f"[{request_id}] Final chat result: {json.dumps({k: v for k, v in result.items() if k != 'response'}, indent=2)}")
        logger.info(f"[{request_id}] === CHAT REQUEST END ===")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"[{request_id}] EXCEPTION: {str(e)}", exc_info=True)
        return jsonify({'error': f'Chat error: {str(e)}'}), 500

@app.route('/chat/stream', methods=['POST'])
def stream_chat():
    """Stream chat responses"""
    def generate():
        try:
            manager = get_enhanced_manager()
            if not manager:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Model manager not initialized'})}\\n\\n"
                return
                
            data = request.get_json()
            prompt = data.get('message', '')
            
            if not prompt.strip():
                yield f"data: {json.dumps({'type': 'error', 'message': 'Message cannot be empty'})}\\n\\n"
                return
            
            logger.info(f"Streaming chat query: {prompt[:100]}...")
            
            # Stream response chunks
            for chunk in manager.chat_query(prompt):
                yield f"data: {json.dumps(chunk)}\\n\\n"
                
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\\n\\n"
    
    return Response(generate(), mimetype='text/plain')

@app.route('/chat/history')
def get_chat_history():
    """Get chat history"""
    try:
        manager = get_enhanced_manager()
        if not manager:
            return jsonify({'error': 'Model manager not initialized'}), 500
            
        history = manager.get_chat_history()
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        })
        
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        return jsonify({'error': f'History error: {str(e)}'}), 500

@app.route('/chat/clear', methods=['POST'])
def clear_chat_history():
    """Clear chat history"""
    try:
        manager = get_enhanced_manager()
        if not manager:
            return jsonify({'error': 'Model manager not initialized'}), 500
            
        manager.clear_chat_history()
        return jsonify({'success': True, 'message': 'Chat history cleared'})
        
    except Exception as e:
        logger.error(f"Clear history error: {str(e)}")
        return jsonify({'error': f'Clear history error: {str(e)}'}), 500

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
            
            manager = get_enhanced_manager()
            if not manager:
                yield f"data: {json.dumps({'error': 'Model manager not initialized'})}\n\n"
                return
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': f'Starting caption generation for {video_file}', 'request_id': request_id})}\n\n"
            
            # Stream caption generation with real-time updates
            video_path = os.path.join('/workspace/surveillance/inputs/videos', video_file)
            
            # Use the streaming generator version
            for update in process_video_for_caption_streaming(
                manager, video_path, video_file, fps_sampling, 
                chunk_size, prompt, processing_mode, output_format, request_id
            ):
                # Capture the final result for storage
                if update.get('type') == 'complete' and update.get('success'):
                    final_result = update
                
                yield f"data: {json.dumps(update)}\n\n"
            
            # Store caption in database after streaming completes
            if final_result:
                try:
                    db = get_caption_db()
                    storage_data = {
                        'caption': final_result.get('caption', ''),
                        'chunk_details': [],  # Streaming doesn't provide chunk details
                        'video_duration': 0,  # Will be calculated from video info
                        'frames_processed': final_result.get('total_frames', 0)
                    }
                    
                    if db.store_video_caption(video_file, storage_data):
                        logger.info(f"[{request_id}] Caption automatically stored after streaming for video: {video_file}")
                    else:
                        logger.warning(f"[{request_id}] Failed to store caption after streaming for video: {video_file}")
                        
                except Exception as storage_error:
                    logger.error(f"[{request_id}] Error storing caption after streaming: {str(storage_error)}")
                
        except Exception as e:
            logger.error(f"[{request_id}] Streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': f'Streaming error: {str(e)}'})}\n\n"
    
    return Response(generate_caption_updates(), mimetype='text/event-stream')

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    """Generate detailed caption for entire video using FPS-based sampling and chunks with real-time updates"""
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
        
        # Process video in chunks for caption generation with optimized performance
        try:
            caption_result = process_video_for_caption_optimized(
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
                    'request_id': request_id,
                    'chunk_details': caption_result.get('chunk_details', [])
                }
                
                # Automatically store caption in database
                try:
                    db = get_caption_db()
                    storage_data = {
                        'caption': caption_result['caption'],
                        'chunk_details': caption_result.get('chunk_details', []),
                        'video_duration': caption_result.get('chunk_details', [{}])[-1].get('end_time', 0) if caption_result.get('chunk_details') else 0,
                        'frames_processed': caption_result['total_frames']
                    }
                    
                    if db.store_video_caption(video_file, storage_data):
                        logger.info(f"[{request_id}] Caption automatically stored for video: {video_file}")
                    else:
                        logger.warning(f"[{request_id}] Failed to store caption for video: {video_file}")
                        
                except Exception as storage_error:
                    logger.error(f"[{request_id}] Error storing caption: {str(storage_error)}")
                
                logger.info(f"[{request_id}] Caption generation successful")
                logger.info(f"[{request_id}] Caption length: {len(caption_result['caption'])} characters")
                logger.info(f"[{request_id}] Total processing time: {processing_time:.2f}s")
                logger.info(f"[{request_id}] === CAPTION GENERATION REQUEST END ===")
                
                return jsonify(result)
            else:
                error_msg = caption_result['error']
                # Check if this is a validation error (not a server error)
                if 'FPS sampling too low' in error_msg:
                    logger.warning(f"[{request_id}] Validation error: {error_msg}")
                    return jsonify({'error': error_msg}), 400
                else:
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
            else:
                error_msg = caption_result['error']
                # Check if this is a validation error (not a server error)
                if 'FPS sampling too low' in error_msg:
                    logger.warning(f"[{request_id}] Validation error: {error_msg}")
                    return jsonify({'error': error_msg}), 400
                else:
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
                    else:
                        # Extend num_patches_list with the last value
                        last_patches = num_patches_list[-1] if num_patches_list else 1
                        while len(num_patches_list) < pixel_values.shape[0]:
                            num_patches_list.append(last_patches)
        
        # Handle tensor dtype based on quantization settings
        if model_manager.enable_8bit or model_manager.enable_4bit:
            pixel_values = pixel_values.to(torch.float16).to(model_manager.model.device)
        else:
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
        
        # Then remove other HTML tags (like <span>, </span>, etc.)
        cleaned_output = re.sub(r'<[^>]+>', '', cleaned_output).strip()
        
        # Remove extra whitespace and newlines
        cleaned_output = re.sub(r'\s+', ' ', cleaned_output).strip()
        
        # Check for invalid responses
        emoticon_patterns = ['>.<', '>_<', '-_-', 'T_T', '^_^', 'o_o', 'O_O', '._.',  '...', ':)', ':(', ':D']
        invalid_responses = ["i'm not sure", "i don't know", "i cannot see", "i cannot determine", "unable to", "sorry", "apologize"]
        html_fragments = ["span>", "<span", "div>", "<div", "</", "/>"]
        
        is_invalid = (
            not cleaned_output or  # Empty after cleaning
            len(cleaned_output) < 10 or  # Too short (reduced from 15)
            cleaned_output in emoticon_patterns or  # Just an emoticon
            any(invalid in cleaned_output.lower() for invalid in invalid_responses) or  # Uncertain responses (exact match)
            any(html in cleaned_output.lower() for html in html_fragments) or  # HTML fragments
            cleaned_output.count(' ') < 2  # Less than 3 words (reduced from 4)
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

def process_video_for_caption_streaming(manager, video_path, video_file, fps_sampling, chunk_size, prompt, processing_mode, output_format, request_id):
    """Generate comprehensive caption using chunked processing with real-time streaming"""
    try:
        import cv2
        import numpy as np
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield {'type': 'error', 'message': 'Could not open video file'}
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Yield video info
        yield {'type': 'video_info', 'duration': duration, 'fps': fps, 'frames': frame_count}
        
        # Calculate processing parameters  
        max_safe_frames = 160
        total_needed_frames = int(fps_sampling * duration)
        effective_chunks = int(np.ceil(duration / chunk_size))
        
        yield {'type': 'processing_info', 'total_chunks': effective_chunks, 'total_frames': total_needed_frames, 'max_safe_frames': max_safe_frames}
        
        chunk_captions = []
        total_frames_processed = 0
        previous_captions = []  # Track context for streaming too
        
        # Process all chunks with streaming
        for chunk_idx in range(effective_chunks):
            start_time = chunk_idx * chunk_size
            end_time = min((chunk_idx + 1) * chunk_size, duration)
            chunk_duration = end_time - start_time
            chunk_frames = min(max_safe_frames, int(fps_sampling * chunk_duration))
            
            yield {'type': 'chunk_start', 'chunk': chunk_idx + 1, 'total_chunks': effective_chunks, 'start_time': start_time, 'end_time': end_time, 'frames': chunk_frames}
            
            try:
                chunk_start_time = time.time()
                
                # Use explicit description prompt to avoid tracking mode
                base_prompt = f"{prompt}"
                
                if chunk_idx == 0:
                    # First chunk - enhanced detail request with anti-tracking instructions
                    chunk_prompt = f"""IMPORTANT: Write only descriptive text. Do NOT use tracking format or XML tags.

{base_prompt}

Provide a comprehensive, detailed description of what happens in this video segment. Include: setting/environment, people's appearance and actions, objects present, movements, interactions, and any notable details. Write in narrative style with rich visual descriptions. Use only plain text description."""
                else:
                    # Subsequent chunks - use last 2 captions for context, focus on NEW content only
                    if previous_captions:
                        # Build minimal context from last 1-2 captions
                        context_parts = []
                        for i, cap in enumerate(previous_captions[-2:]):  # Last 2 captions max
                            # Clean numbered lists and extract key info
                            import re
                            # Remove numbered list formatting (1., 2., etc.)
                            cleaned_cap = re.sub(r'^\d+\.\s*', '', cap)
                            # Remove any leading numbers or bullets
                            cleaned_cap = re.sub(r'^[\d\-\*\•]\s*', '', cleaned_cap)
                            # Extract first sentence only (key transition info)
                            first_sentence = cleaned_cap.split('.')[0][:40] + "..." if len(cleaned_cap) > 40 else cleaned_cap.split('.')[0]
                            context_parts.append(f"Previous scene: {first_sentence}")
                        
                        context_text = " | ".join(context_parts)
                        
                        chunk_prompt = f"""CONTEXT: {context_text}

IMPORTANT: Write a fresh, complete description of what happens in THIS video segment. Do NOT continue numbered lists. Do NOT just write numbers. Write full descriptive sentences about the NEW content only.

{base_prompt}

Describe what happens in this video segment with complete sentences. Focus on NEW actions, movements, and events. Start your description directly without numbers or list formatting."""
                    else:
                        # No context, standard description request
                        chunk_prompt = f"""IMPORTANT: Write only descriptive text. Do NOT use tracking format or XML tags.

{base_prompt}

Describe what happens in this specific video segment. Focus on actions, movements, setting, people, and events visible in this time period."""
                
                result = process_video_segment(
                    manager.model_manager,
                    video_path=video_path,
                    prompt=chunk_prompt,
                    num_frames=chunk_frames,
                    start_time=start_time,
                    end_time=end_time,
                    fps_sampling=fps_sampling
                )
                
                chunk_time = time.time() - chunk_start_time
                
                if result and ('text' in result or 'response' in result):
                    caption_text = result.get('text', result.get('response', ''))
                    
                    # Check if we actually got text content
                    if not caption_text or not caption_text.strip():
                        error_msg = f"Empty caption returned. Result keys: {result.keys() if result else 'None'}"
                        if result.get('error'):
                            error_msg += f", Error: {result.get('error')}"
                        yield {'type': 'chunk_error', 'chunk': chunk_idx + 1, 'message': error_msg}
                        continue
                        
                    # Apply same length limit
                    MAX_CHUNK_LENGTH = 500
                    if len(caption_text) > MAX_CHUNK_LENGTH:
                        truncate_point = caption_text[:MAX_CHUNK_LENGTH].rfind('. ')
                        if truncate_point > 300:
                            caption_text = caption_text[:truncate_point + 1]
                        else:
                            caption_text = caption_text[:MAX_CHUNK_LENGTH].rsplit(' ', 1)[0] + '...'
                else:
                    error_msg = f"No text/response in result. Keys: {result.keys() if result else 'None'}"
                    if result and result.get('error'):
                        error_msg += f", Error: {result.get('error')}"
                    yield {'type': 'chunk_error', 'chunk': chunk_idx + 1, 'message': error_msg}
                    continue
                
                # Format chunk caption with timestamp
                if output_format == 'timestamped':
                    formatted_chunk = f"**[{int(start_time//60)}:{int(start_time%60):02d} - {int(end_time//60)}:{int(end_time%60):02d}]** {caption_text}"
                else:
                    formatted_chunk = caption_text
                
                chunk_captions.append(formatted_chunk)
                total_frames_processed += chunk_frames
                
                # Add to context for next chunks (use raw caption, limit storage)
                previous_captions.append(caption_text)
                # Keep only last 2 captions to prevent memory buildup
                if len(previous_captions) > 2:
                    previous_captions = previous_captions[-2:]
                
                # Yield chunk completion with the caption
                yield {
                    'type': 'chunk_complete', 
                    'chunk': chunk_idx + 1, 
                    'total_chunks': effective_chunks,
                    'caption': formatted_chunk,
                    'processing_time': chunk_time,
                    'characters': len(caption_text),
                    'frames': chunk_frames,
                    'progress': ((chunk_idx + 1) / effective_chunks) * 100
                }
                
                # Selective cleanup - only every 3 chunks or on last chunk to balance memory vs speed
                if (chunk_idx + 1) % 3 == 0 or chunk_idx == effective_chunks - 1:
                    manager._clear_preprocessed_data()
                    yield {'type': 'cleanup', 'message': f'Cleaned cache after chunk {chunk_idx + 1}'}
                    
                # Light cleanup after each chunk - just clear CUDA cache
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as chunk_error:
                yield {'type': 'chunk_error', 'chunk': chunk_idx + 1, 'message': str(chunk_error)}
                continue
        
        # Final processing and combination
        if not chunk_captions:
            yield {'type': 'error', 'message': 'No video chunks were successfully processed'}
            return
        
        # Combine all captions based on output format
        if output_format == 'timestamped':
            final_caption = f"**Timestamped Video Caption ({video_file}):**\n\n" + "\n\n".join(chunk_captions)
        elif output_format == 'detailed':
            final_caption = f"**Detailed Video Analysis ({video_file}):**\n\n**Technical Details:**\n- Video Duration: {duration:.1f}s\n- Chunk Size: {chunk_size}s\n- Chunks Processed: {len(chunk_captions)}/{effective_chunks}\n- Frame Sampling: {fps_sampling} FPS ({total_frames_processed} total frames)\n- Processing Mode: {processing_mode}\n\n**Content Analysis by Time Segments:**\n\n" + "\n\n".join(chunk_captions)
        elif output_format == 'summary':
            all_content = " ".join([cap.split("**", 2)[-1] if "**" in cap else cap for cap in chunk_captions])
            summary = all_content[:300] + "..." if len(all_content) > 300 else all_content
            final_caption = f"**Video Summary + Details ({video_file}):**\n\n**Quick Summary:** {summary}\n\n**Detailed Timeline:**\n\n" + "\n\n".join(chunk_captions)
        else:  # paragraph format
            final_caption = f"**Video Caption ({video_file}):**\n\n"
            clean_chunks = []
            for chunk in chunk_captions:
                if "**[" in chunk and "]**" in chunk:
                    clean_content = chunk.split("]**", 1)[-1].strip()
                else:
                    clean_content = chunk.strip()
                clean_chunks.append(clean_content)
            final_caption += " ".join(clean_chunks)
        
        import html
        final_caption = html.unescape(final_caption).replace('\x00', '').strip()
        
        # Final result
        yield {
            'type': 'complete',
            'caption': final_caption,
            'total_frames': total_frames_processed,
            'chunks_processed': len(chunk_captions),
            'success': True
        }
        
    except Exception as e:
        yield {'type': 'error', 'message': str(e)}

def process_video_for_caption_optimized(manager, video_path, video_file, fps_sampling, chunk_size, prompt, processing_mode, output_format, request_id):
    """Generate comprehensive caption using optimized chunked processing with performance improvements"""
    try:
        import cv2
        import numpy as np
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'success': False, 'error': 'Could not open video file'}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        logger.info(f"[{request_id}] Video info: {duration:.1f}s, {fps:.1f} FPS, {frame_count} frames")
        
        # Validate minimum frame count per chunk
        frames_per_chunk = fps_sampling * chunk_size
        if frames_per_chunk < 8:  # Minimum 8 frames per chunk
            recommended_fps = max(1.0, 8 / chunk_size)
            logger.warning(f"[{request_id}] Rejecting low frame count per chunk: {frames_per_chunk:.1f} frames")
            return {
                'success': False,
                'error': f'FPS too low for {chunk_size}s chunks. Need {frames_per_chunk:.1f} frames per chunk, minimum 8. Try FPS {recommended_fps:.1f} or higher.'
            }
        
        # Calculate processing parameters
        max_safe_frames = 160  # Tested safe limit
        total_needed_frames = int(fps_sampling * duration)
        effective_chunks = int(np.ceil(duration / chunk_size))
        
        logger.info(f"[{request_id}] Processing strategy:")
        logger.info(f"[{request_id}]   - Video duration: {duration:.1f}s")
        logger.info(f"[{request_id}]   - Total chunks: {effective_chunks}")
        logger.info(f"[{request_id}]   - FPS sampling: {fps_sampling} FPS")
        logger.info(f"[{request_id}]   - Processing mode: {processing_mode}")
        
        chunk_captions = []
        chunk_details = []
        total_frames_processed = 0
        
        if total_needed_frames <= max_safe_frames:
            # Single pass processing for short videos
            logger.info(f"[{request_id}] Using single-pass processing: {total_needed_frames} frames <= {max_safe_frames} limit")
            
            try:
                logger.info(f"[{request_id}] About to call process_video...")
                result = process_video_segment(
                    manager.model_manager,
                    video_path=video_path,
                    prompt=prompt,
                    num_frames=total_needed_frames,
                    start_time=0,
                    end_time=duration,
                    fps_sampling=fps_sampling
                )
                
                if result and ('text' in result or 'response' in result):
                    caption_text = result.get('text', result.get('response', ''))
                    
                    # Check if we actually got text content (same as chunk processing)
                    if not caption_text or not caption_text.strip():
                        logger.warning(f"[{request_id}] Single-pass returned empty caption text")
                        logger.info(f"[{request_id}] Result keys: {result.keys() if result else 'None'}")
                        logger.info(f"[{request_id}] Result success: {result.get('success', False)}")
                        if result.get('error'):
                            logger.error(f"[{request_id}] Error: {result.get('error')}")
                        return {'success': False, 'error': f"Model returned empty caption. Error: {result.get('error', 'Unknown')}"}
                    
                    # Apply same length limit as chunks
                    MAX_CAPTION_LENGTH = 500
                    if len(caption_text) > MAX_CAPTION_LENGTH:
                        truncate_point = caption_text[:MAX_CAPTION_LENGTH].rfind('. ')
                        if truncate_point > 300:
                            caption_text = caption_text[:truncate_point + 1]
                        else:
                            caption_text = caption_text[:MAX_CAPTION_LENGTH].rsplit(' ', 1)[0] + '...'
                    
                    chunk_captions = [caption_text]
                    total_frames_processed = total_needed_frames
                    effective_chunks = 1
                    
                    chunk_details.append({
                        'chunk': 1,
                        'start_time': 0,
                        'end_time': duration,
                        'frames': total_needed_frames,
                        'caption_length': len(caption_text),
                        'processing_time': result.get('processing_time', 0)
                    })
                    
                    logger.info(f"[{request_id}] Single-pass processing completed: {len(caption_text)} characters")
                else:
                    error_msg = "No caption generated from video processing"
                    if result:
                        logger.info(f"[{request_id}] Result keys: {result.keys()}")
                        if result.get('error'):
                            error_msg = f"Processing error: {result.get('error')}"
                    return {'success': False, 'error': error_msg}
                    
            except Exception as process_error:
                logger.error(f"[{request_id}] Single-pass processing failed: {str(process_error)}")
                return {'success': False, 'error': f'Video processing failed: {str(process_error)}'}
        else:
            # Multi-chunk processing for long videos
            logger.info(f"[{request_id}] Video too long for single pass ({total_needed_frames} > {max_safe_frames})")
            logger.info(f"[{request_id}] 🚀 Starting optimized processing of {effective_chunks} chunks...")
            
            # Track previous captions for context
            previous_captions = []
            
            # Process all chunks with optimized cleanup
            for chunk_idx in range(effective_chunks):
                start_time = chunk_idx * chunk_size
                end_time = min((chunk_idx + 1) * chunk_size, duration)
                chunk_duration = end_time - start_time
                chunk_frames = min(max_safe_frames, int(fps_sampling * chunk_duration))
                
                logger.info(f"[{request_id}] 📹 Processing chunk {chunk_idx + 1}/{effective_chunks}: {start_time:.1f}s - {end_time:.1f}s ({chunk_frames} frames)")
                
                try:
                    chunk_start_time = time.time()
                    
                    # Use explicit description prompt to avoid tracking mode
                    base_prompt = f"{prompt}"
                    
                    if chunk_idx == 0:
                        # First chunk - clear description request
                        chunk_prompt = f"{base_prompt}\n\nProvide a detailed description of what happens in this video segment. Do not use tracking or object detection format."
                    else:
                        # Subsequent chunks - add minimal context
                        if previous_captions and len(previous_captions[-1]) > 50:
                            last_caption = previous_captions[-1]
                            if len(last_caption) > 200:
                                last_caption = last_caption[:200] + "..."
                            chunk_prompt = f"Previous: {last_caption}\n\n{base_prompt}\n\nDescribe what happens next in this video segment. Provide a narrative description, not tracking data."
                        else:
                            # No context, clear description request
                            chunk_prompt = f"{base_prompt}\n\nProvide a detailed description of what happens in this video segment. Do not use tracking or object detection format."
                    
                    result = process_video_segment(
                        manager.model_manager,
                        video_path=video_path,
                        prompt=chunk_prompt,
                        num_frames=chunk_frames,
                        start_time=start_time,
                        end_time=end_time,
                        fps_sampling=fps_sampling
                    )
                    
                    chunk_time = time.time() - chunk_start_time
                    
                    if result and ('text' in result or 'response' in result):
                        caption_text = result.get('text', result.get('response', ''))
                        
                        # Check if we actually got text content
                        if not caption_text or not caption_text.strip():
                            logger.warning(f"[{request_id}] Chunk {chunk_idx + 1} returned empty caption text")
                            # Log more details for debugging
                            logger.info(f"[{request_id}] Result keys: {result.keys() if result else 'None'}")
                            logger.info(f"[{request_id}] Result success: {result.get('success', False)}")
                            if result.get('error'):
                                logger.error(f"[{request_id}] Chunk error: {result.get('error')}")
                            continue
                        
                        # Limit caption length to prevent excessive output
                        MAX_CHUNK_LENGTH = 500  # Maximum characters per chunk (balanced for good detail)
                        if len(caption_text) > MAX_CHUNK_LENGTH:
                            # Find a good breakpoint (sentence end) near the limit
                            truncate_point = caption_text[:MAX_CHUNK_LENGTH].rfind('. ')
                            if truncate_point > 300:  # If we found a sentence end after 300 chars
                                caption_text = caption_text[:truncate_point + 1]
                            else:
                                caption_text = caption_text[:MAX_CHUNK_LENGTH].rsplit(' ', 1)[0] + '...'
                            logger.info(f"[{request_id}] Trimmed chunk {chunk_idx + 1} caption from original length to {len(caption_text)} chars")
                    else:
                        logger.warning(f"[{request_id}] Chunk {chunk_idx + 1} result missing text/response fields")
                        if result:
                            logger.info(f"[{request_id}] Result keys: {result.keys()}")
                            if result.get('error'):
                                logger.error(f"[{request_id}] Error: {result.get('error')}")
                        continue
                    
                    # Format chunk caption with timestamp
                    if output_format == 'timestamped':
                        formatted_chunk = f"**[{int(start_time//60)}:{int(start_time%60):02d} - {int(end_time//60)}:{int(end_time%60):02d}]** {caption_text}"
                    else:
                        formatted_chunk = caption_text
                    
                    chunk_captions.append(formatted_chunk)
                    total_frames_processed += chunk_frames
                    
                    # Add to context for next chunks (use raw caption, not formatted)
                    # Only skip clearly broken responses
                    skip_patterns = ['>.<', '>_<', "i'm not sure", "i don't know", "span>", "<span", "sorry"]
                    is_valid_context = (
                        caption_text and  # Not empty
                        len(caption_text) > 20 and  # Reasonable length
                        caption_text.count(' ') >= 3 and  # At least 4 words
                        not any(pattern in caption_text.lower() for pattern in skip_patterns)
                    )
                    
                    if is_valid_context:
                        previous_captions.append(caption_text)
                    else:
                        logger.warning(f"[{request_id}] Skipping invalid caption from context: {caption_text[:50]}")
                    
                    # Store chunk details for UI updates
                    chunk_details.append({
                        'chunk': chunk_idx + 1,
                        'start_time': start_time,
                        'end_time': end_time,
                        'frames': chunk_frames,
                        'caption': formatted_chunk,
                        'caption_length': len(caption_text),
                        'processing_time': chunk_time
                    })
                    
                    # Show detailed chunk completion with text preview and timing
                    text_preview = caption_text[:200] + "..." if len(caption_text) > 200 else caption_text
                    logger.info(f"[{request_id}] ⚡ Chunk {chunk_idx + 1}/{effective_chunks} completed in {chunk_time:.1f}s")
                    logger.info(f"[{request_id}]   📊 Stats: {len(caption_text)} characters, {chunk_frames} frames processed")
                    logger.info(f"[{request_id}]   📝 Generated text: {text_preview}")
                    logger.info(f"[{request_id}]   ⏱️  Cumulative time: {sum(cd['processing_time'] for cd in chunk_details):.1f}s total")
                    
                    # Mandatory cleanup after every chunk to prevent memory accumulation slowdown
                    manager._clear_preprocessed_data()
                    logger.info(f"[{request_id}] 🧹 Cleaned cache after chunk {chunk_idx + 1} (preventing slowdown)")
                        
                except Exception as chunk_error:
                    logger.error(f"[{request_id}] Chunk {chunk_idx + 1} processing failed: {str(chunk_error)}")
                    # Continue processing other chunks instead of failing completely
                    continue
            
            # Check if any chunks were successfully processed
            if not chunk_captions:
                return {'success': False, 'error': 'No video chunks were successfully processed'}
            
            # Comprehensive processing summary with detailed metrics
            total_processing_time = sum(cd.get('processing_time', 0) for cd in chunk_details)
            avg_time_per_chunk = total_processing_time / len(chunk_details) if chunk_details else 0
            total_characters = sum(cd.get('caption_length', 0) for cd in chunk_details)
            
            logger.info(f"[{request_id}] ✅ COMPLETED: Successfully processed {len(chunk_captions)}/{effective_chunks} chunks")
            logger.info(f"[{request_id}] 📊 FINAL Processing Summary:")
            logger.info(f"[{request_id}]   ⏱️  Total processing time: {total_processing_time:.1f}s")
            logger.info(f"[{request_id}]   🏃 Average per chunk: {avg_time_per_chunk:.1f}s")
            logger.info(f"[{request_id}]   📝 Total characters generated: {total_characters}")
            logger.info(f"[{request_id}]   🎬 Total frames processed: {total_frames_processed}")
            logger.info(f"[{request_id}]   🧹 Cleanup strategy: after every chunk (prevents slowdown)")
            
            # Log individual chunk performance details
            logger.info(f"[{request_id}] 📋 Individual Chunk Performance:")
            for cd in chunk_details:
                time_range = f"{cd.get('start_time', 0):.1f}-{cd.get('end_time', 0):.1f}s"
                proc_time = cd.get('processing_time', 0)
                char_count = cd.get('caption_length', 0)
                frame_count = cd.get('frames', 0)
                chunk_num = cd.get('chunk', 0)
                logger.info(f"[{request_id}]   Chunk {chunk_num}: {proc_time:.1f}s, {char_count} chars, {frame_count} frames ({time_range}) 🧹")
        
        # Combine chunk captions based on output format
        if output_format == 'timestamped':
            final_caption = f"**Timestamped Video Caption ({video_file}):**\n\n" + "\n\n".join(chunk_captions)
        elif output_format == 'detailed':
            final_caption = f"**Detailed Video Analysis ({video_file}):**\n\n**Technical Details:**\n- Video Duration: {duration:.1f}s\n- Chunk Size: {chunk_size}s\n- Chunks Processed: {len(chunk_captions)}/{effective_chunks}\n- Frame Sampling: {fps_sampling} FPS ({total_frames_processed} total frames)\n- Processing Mode: {processing_mode}\n\n**Content Analysis by Time Segments:**\n\n" + "\n\n".join(chunk_captions)
        elif output_format == 'summary':
            all_content = " ".join([cap.split("**", 2)[-1] if "**" in cap else cap for cap in chunk_captions])
            summary = all_content[:300] + "..." if len(all_content) > 300 else all_content
            final_caption = f"**Video Summary + Details ({video_file}):**\n\n**Quick Summary:** {summary}\n\n**Detailed Timeline:**\n\n" + "\n\n".join(chunk_captions)
        else:  # paragraph format
            final_caption = f"**Video Caption ({video_file}):**\n\n"
            clean_chunks = []
            for chunk in chunk_captions:
                if "**[" in chunk and "]**" in chunk:
                    clean_content = chunk.split("]**", 1)[-1].strip()
                else:
                    clean_content = chunk.strip()
                clean_chunks.append(clean_content)
            final_caption += " ".join(clean_chunks)
        
        # Final cleanup
        import html
        final_caption = html.unescape(final_caption).replace('\x00', '').strip()
        
        if not final_caption or final_caption.isspace():
            return {'success': False, 'error': 'Generated caption is empty or invalid'}
        
        logger.info(f"[{request_id}] Final caption ready: {len(final_caption)} characters")
        
        # Final cleanup after all processing
        try:
            manager._clear_preprocessed_data()
            logger.info(f"[{request_id}] Final cleanup completed")
        except Exception as cleanup_error:
            logger.warning(f"[{request_id}] Final cleanup warning: {str(cleanup_error)}")
        
        return {
            'success': True,
            'caption': final_caption,
            'total_frames': total_frames_processed,
            'chunks_processed': len(chunk_captions),
            'chunk_details': chunk_details
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Error in process_video_for_caption_optimized: {str(e)}")
        # Cleanup even on error
        try:
            manager._clear_preprocessed_data()
            logger.info(f"[{request_id}] Emergency cleanup completed")
        except:
            pass
        return {'success': False, 'error': str(e)}

def process_video_for_caption(manager, video_path, video_file, fps_sampling, chunk_size, prompt, processing_mode, output_format, request_id):
    """Generate comprehensive caption using proper chunked InternVideo2.5 methodology"""
    try:
        import cv2
        import numpy as np
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'success': False, 'error': 'Could not open video file'}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        logger.info(f"[{request_id}] Video info: {duration:.1f}s, {fps:.1f} FPS, {frame_count} frames")
        
        # Validate minimum frame count per chunk
        frames_per_chunk = fps_sampling * chunk_size
        if frames_per_chunk < 8:  # Minimum 8 frames per chunk for meaningful video understanding
            recommended_fps = max(1.0, 8 / chunk_size)
            logger.warning(f"[{request_id}] Rejecting low frame count per chunk: {frames_per_chunk:.1f} frames")
            return {
                'success': False,
                'error': f'FPS too low for {chunk_size}s chunks. Need {frames_per_chunk:.1f} frames per chunk, minimum 8. Try FPS {recommended_fps:.1f} or higher.'
            }
        
        # Implement proper chunking strategy
        total_chunks = int(np.ceil(duration / chunk_size))
        
        logger.info(f"[{request_id}] Chunking strategy:")
        logger.info(f"[{request_id}]   - Video duration: {duration:.1f}s")
        logger.info(f"[{request_id}]   - Chunk size: {chunk_size}s") 
        logger.info(f"[{request_id}]   - Total chunks: {total_chunks}")
        logger.info(f"[{request_id}]   - FPS sampling: {fps_sampling} FPS")
        logger.info(f"[{request_id}]   - Processing mode: {processing_mode}")
        
        # Calculate safe processing parameters
        # Limit to safe frame count for initial implementation
        max_safe_frames = 160  # Tested safe limit
        total_needed_frames = int(fps_sampling * duration)
        
        if total_needed_frames <= max_safe_frames:
            # Single pass processing for short videos
            logger.info(f"[{request_id}] Using single-pass processing: {total_needed_frames} frames <= {max_safe_frames} limit")
            
            try:
                logger.info(f"[{request_id}] Manager type: {type(manager)}")
                logger.info(f"[{request_id}] Has model_manager: {hasattr(manager, 'model_manager')}")
                if hasattr(manager, 'model_manager'):
                    logger.info(f"[{request_id}] Model manager type: {type(manager.model_manager)}")
                    logger.info(f"[{request_id}] Model manager has process_video: {hasattr(manager.model_manager, 'process_video')}")
                else:
                    logger.error(f"[{request_id}] Manager does not have model_manager attribute!")
                    return {'success': False, 'error': 'Manager does not have model_manager attribute'}
                
                logger.info(f"[{request_id}] About to clear processing cache...")
                manager._clear_preprocessed_data()
                
                logger.info(f"[{request_id}] About to call process_video...")
                result = process_video_segment(
                    manager.model_manager,
                    video_path=video_path,
                    prompt=prompt,
                    num_frames=total_needed_frames,
                    start_time=0,
                    end_time=duration,
                    fps_sampling=fps_sampling
                )
                
                logger.info(f"[{request_id}] process_video completed successfully")
                
                if result and 'text' in result:
                    caption_text = result['text']
                elif result and 'response' in result:
                    caption_text = result['response'] 
                else:
                    return {'success': False, 'error': 'No caption generated from video processing'}
                
                chunk_captions = [caption_text]
                total_frames_processed = total_needed_frames
                effective_chunks = 1
                
                logger.info(f"[{request_id}] Single-pass processing completed: {len(caption_text)} characters")
                
            except Exception as process_error:
                logger.error(f"[{request_id}] Single-pass processing failed: {str(process_error)}")
                return {'success': False, 'error': f'Video processing failed: {str(process_error)}'}
        else:
            # Multi-chunk processing for long videos
            logger.info(f"[{request_id}] Video too long for single pass ({total_needed_frames} > {max_safe_frames})")
            logger.info(f"[{request_id}] Implementing full chunk-based processing with {chunk_size}s chunks...")
            
            chunk_captions = []
            total_frames_processed = 0
            effective_chunks = int(np.ceil(duration / chunk_size))
            
            # Process all chunks
            logger.info(f"[{request_id}] Starting processing of {effective_chunks} chunks for {duration:.1f}s video...")
            for chunk_idx in range(effective_chunks):
                start_time = chunk_idx * chunk_size
                end_time = min((chunk_idx + 1) * chunk_size, duration)
                chunk_duration = end_time - start_time
                chunk_frames = min(max_safe_frames, int(fps_sampling * chunk_duration))
                
                logger.info(f"[{request_id}] Processing chunk {chunk_idx + 1}/{effective_chunks}: {start_time:.1f}s - {end_time:.1f}s ({chunk_frames} frames)")
                
                try:
                    # Process chunk with minimal cleanup
                    chunk_start_time = time.time()
                    
                    # Create chunk-specific prompt
                    # Combine user prompt with reasonable constraints
                    if "concise" not in prompt.lower() and "brief" not in prompt.lower():
                        # Add focus instruction if not already present
                        chunk_prompt = f"{prompt}\n\nFor this {chunk_duration:.0f}-second segment: Focus on the main actions, events, and movements. Provide a clear but reasonably detailed description (3-5 sentences)."
                    else:
                        # User already requested concise output
                        chunk_prompt = f"{prompt}\n\nSegment: {start_time:.0f}-{end_time:.0f} seconds."
                    
                    # Process video segment using ModelManager directly with proper bounds
                    result = process_video_segment(
                        manager.model_manager,
                        video_path=video_path,
                        prompt=chunk_prompt,
                        num_frames=chunk_frames,
                        start_time=start_time,
                        end_time=end_time,
                        fps_sampling=fps_sampling
                    )
                    
                    chunk_time = time.time() - chunk_start_time
                    
                    if result and 'text' in result:
                        caption_text = result['text']
                    elif result and 'response' in result:
                        caption_text = result['response']
                    else:
                        logger.warning(f"[{request_id}] Chunk {chunk_idx + 1} generated no caption, skipping...")
                        continue
                    
                    # Format chunk caption with timestamp
                    if output_format == 'timestamped':
                        formatted_chunk = f"**[{int(start_time//60)}:{int(start_time%60):02d} - {int(end_time//60)}:{int(end_time%60):02d}]** {caption_text}"
                    else:
                        formatted_chunk = caption_text
                    
                    chunk_captions.append(formatted_chunk)
                    total_frames_processed += chunk_frames
                    
                    logger.info(f"[{request_id}] ⚡ Chunk {chunk_idx + 1}/{effective_chunks} completed in {chunk_time:.1f}s: {len(caption_text)} characters")
                    
                    # Only clear GPU cache every few chunks to improve performance
                    if (chunk_idx + 1) % 3 == 0:  # Every 3rd chunk
                        manager._clear_preprocessed_data()
                        logger.info(f"[{request_id}] 🧹 Cleaned cache after chunk {chunk_idx + 1}")
                    
                except Exception as chunk_error:
                    logger.error(f"[{request_id}] Chunk {chunk_idx + 1} processing failed: {str(chunk_error)}")
                    # Continue processing other chunks instead of failing completely
                    continue
            
            # Check if any chunks were successfully processed
            if not chunk_captions:
                return {'success': False, 'error': 'No video chunks were successfully processed'}
            
            # Comprehensive processing summary with detailed metrics
            total_processing_time = sum(cd.get('processing_time', 0) for cd in chunk_details)
            avg_time_per_chunk = total_processing_time / len(chunk_details) if chunk_details else 0
            total_characters = sum(cd.get('caption_length', 0) for cd in chunk_details)
            
            logger.info(f"[{request_id}] ✅ COMPLETED: Successfully processed {len(chunk_captions)}/{effective_chunks} chunks")
            logger.info(f"[{request_id}] 📊 FINAL Processing Summary:")
            logger.info(f"[{request_id}]   ⏱️  Total processing time: {total_processing_time:.1f}s")
            logger.info(f"[{request_id}]   🏃 Average per chunk: {avg_time_per_chunk:.1f}s")
            logger.info(f"[{request_id}]   📝 Total characters generated: {total_characters}")
            logger.info(f"[{request_id}]   🎬 Total frames processed: {total_frames_processed}")
            logger.info(f"[{request_id}]   🧹 Cleanup strategy: after every chunk (prevents slowdown)")
            
            # Log individual chunk performance details
            logger.info(f"[{request_id}] 📋 Individual Chunk Performance:")
            for cd in chunk_details:
                time_range = f"{cd.get('start_time', 0):.1f}-{cd.get('end_time', 0):.1f}s"
                proc_time = cd.get('processing_time', 0)
                char_count = cd.get('caption_length', 0)
                frame_count = cd.get('frames', 0)
                chunk_num = cd.get('chunk', 0)
                logger.info(f"[{request_id}]   Chunk {chunk_num}: {proc_time:.1f}s, {char_count} chars, {frame_count} frames ({time_range}) 🧹")
            logger.info(f"[{request_id}] Video coverage: {(len(chunk_captions) * chunk_size):.1f}s / {duration:.1f}s")
        
        logger.info(f"[{request_id}] Processing completed with {len(chunk_captions)} segments")
        
        # Combine chunk captions based on output format
        if output_format == 'timestamped':
            final_caption = f"**Timestamped Video Caption ({video_file}):**\n\n"
            final_caption += "\n\n".join(chunk_captions)
            
        elif output_format == 'detailed':
            final_caption = f"**Detailed Video Analysis ({video_file}):**\n\n"
            final_caption += f"**Technical Details:**\n"
            final_caption += f"- Video Duration: {duration:.1f}s\n"
            final_caption += f"- Chunk Size: {chunk_size}s\n"
            final_caption += f"- Chunks Processed: {len(chunk_captions)}/{effective_chunks}\n"
            final_caption += f"- Frame Sampling: {fps_sampling} FPS ({total_frames_processed} total frames)\n"
            final_caption += f"- Processing Mode: {processing_mode}\n\n"
            final_caption += f"**Content Analysis by Time Segments:**\n\n"
            final_caption += "\n\n".join(chunk_captions)
            
        elif output_format == 'summary':
            # Create summary from all chunks
            all_content = " ".join([cap.split("**", 2)[-1] if "**" in cap else cap for cap in chunk_captions])
            summary = all_content[:300] + "..." if len(all_content) > 300 else all_content
            
            final_caption = f"**Video Summary + Details ({video_file}):**\n\n"
            final_caption += f"**Quick Summary:** {summary}\n\n"
            final_caption += f"**Detailed Timeline:**\n\n"
            final_caption += "\n\n".join(chunk_captions)
            
        else:  # paragraph format
            final_caption = f"**Video Caption ({video_file}):**\n\n"
            # Remove timestamps for paragraph format and create flowing narrative
            clean_chunks = []
            for chunk in chunk_captions:
                if "**[" in chunk and "]**" in chunk:
                    # Remove timestamp formatting for paragraph style
                    clean_content = chunk.split("]**", 1)[-1].strip()
                else:
                    clean_content = chunk.strip()
                clean_chunks.append(clean_content)
            final_caption += " ".join(clean_chunks)
        
        # Final cleanup of the complete caption
        import html
        final_caption = html.unescape(final_caption)
        
        # Remove any remaining problematic characters that might break JSON
        final_caption = final_caption.replace('\x00', '').strip()
        
        # Ensure the caption isn't empty
        if not final_caption or final_caption.isspace():
            return {'success': False, 'error': 'Generated caption is empty or invalid'}
        
        logger.info(f"[{request_id}] Final caption ready: {len(final_caption)} characters")
        
        # Cleanup GPU memory and clear video session after processing
        logger.info(f"[{request_id}] Starting cleanup after caption generation...")
        try:
            manager._clear_preprocessed_data()
            logger.info(f"[{request_id}] Video session cleared")
        except Exception as cleanup_error:
            logger.warning(f"[{request_id}] Cleanup warning: {str(cleanup_error)}")
        
        return {
            'success': True,
            'caption': final_caption,
            'total_frames': total_frames_processed,
            'chunks_processed': len(chunk_captions)
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Error in process_video_for_caption: {str(e)}")
        # Cleanup even on error
        try:
            manager._clear_preprocessed_data()
            logger.info(f"[{request_id}] Emergency cleanup completed")
        except:
            pass
        return {'success': False, 'error': str(e)}

def calculate_video_chunks(duration, chunk_size, processing_mode):
    """Calculate video chunks based on processing mode with performance limits"""
    chunks = []
    
    # For very long videos, increase chunk size to limit total chunks
    if duration > 600:  # 10+ minutes
        chunk_size = max(chunk_size, 120)  # Min 2 minutes per chunk
    elif duration > 300:  # 5+ minutes  
        chunk_size = max(chunk_size, 90)   # Min 1.5 minutes per chunk
    
    if processing_mode == 'sequential':
        # Non-overlapping chunks
        current_time = 0
        while current_time < duration:
            end_time = min(current_time + chunk_size, duration)
            chunks.append((current_time, end_time))
            current_time = end_time
            
    elif processing_mode == 'overlapping':
        # 25% overlap between chunks
        overlap = chunk_size * 0.25
        step = chunk_size - overlap  # How much to advance each time
        current_time = 0
        
        while current_time < duration:
            end_time = min(current_time + chunk_size, duration)
            chunks.append((current_time, end_time))
            
            # Move to next chunk start
            current_time += step
            
            # If the remaining duration is less than half a chunk, 
            # merge it with the last chunk or skip
            if current_time < duration and (duration - current_time) < (chunk_size * 0.5):
                # Extend the last chunk to cover remaining duration
                if chunks:
                    chunks[-1] = (chunks[-1][0], duration)
                break
                
    elif processing_mode == 'keyframe':
        # Simulate keyframe detection with fixed intervals (in real implementation, this would use video analysis)
        keyframe_interval = chunk_size * 0.8  # Slightly smaller chunks for key moments
        current_time = 0
        while current_time < duration:
            end_time = min(current_time + keyframe_interval, duration)
            chunks.append((current_time, end_time))
            current_time = end_time
            
    elif processing_mode == 'adaptive':
        # Adaptive sampling - smaller chunks at the beginning and end, larger in middle
        if duration <= chunk_size:
            chunks.append((0, duration))
        else:
            # Smaller chunks for intro/outro
            intro_size = min(chunk_size * 0.7, duration * 0.2)
            outro_size = min(chunk_size * 0.7, duration * 0.2)
            
            chunks.append((0, intro_size))
            
            # Larger chunks for middle section
            middle_start = intro_size
            middle_end = duration - outro_size
            if middle_end > middle_start:
                middle_duration = middle_end - middle_start
                middle_chunks = max(1, int(middle_duration / chunk_size))
                middle_chunk_size = middle_duration / middle_chunks
                
                for i in range(middle_chunks):
                    start = middle_start + (i * middle_chunk_size)
                    end = min(start + middle_chunk_size, middle_end)
                    chunks.append((start, end))
            
            # Outro chunk
            if outro_size > 0:
                chunks.append((duration - outro_size, duration))
    
    return chunks

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

@app.route('/video-snippet', methods=['POST'])
def get_video_snippet_info():
    """Get video snippet information for direct streaming"""
    request_id = str(int(time.time()))[-6:]
    
    try:
        logger.info(f"[{request_id}] === VIDEO SNIPPET INFO REQUEST ===")
        
        manager = get_enhanced_manager()
        if not manager:
            return jsonify({'error': 'Model manager not initialized'}), 500
        
        data = request.get_json()
        timestamp = float(data.get('timestamp', 0))
        duration = float(data.get('duration', 10.0))
        
        logger.info(f"[{request_id}] Getting snippet info for {timestamp}s duration {duration}s")
        
        # Calculate start and end times - 1 second back, rest forward
        start_time = max(0, timestamp - 1.0)  # Only 1 second before target
        end_time = timestamp + (duration - 1.0)  # Remaining duration after target
        
        # Get video info from session if available
        session_info = manager.get_session_info()
        video_duration = 0
        if session_info and session_info.get('video_info'):
            video_info = session_info['video_info']
            if 'duration' in video_info:
                # Parse duration from string like "180.5s" or just get numeric value
                duration_str = str(video_info['duration'])
                if 's' in duration_str:
                    video_duration = float(duration_str.replace('s', ''))
                else:
                    video_duration = float(duration_str)
            elif 'duration_seconds' in video_info:
                video_duration = video_info['duration_seconds']
        
        # Ensure end time doesn't exceed video duration
        if video_duration > 0:
            end_time = min(end_time, video_duration)
            
        logger.info(f"[{request_id}] Video duration: {video_duration}s, calculated segment: {start_time:.1f}s-{end_time:.1f}s")
        
        snippet_info = {
            'success': True,
            'video_url': '/video-stream',
            'start_time': start_time,
            'end_time': end_time,
            'target_timestamp': timestamp,
            'duration': end_time - start_time,
            'description': f'Video segment from {start_time:.1f}s to {end_time:.1f}s'
        }
        
        logger.info(f"[{request_id}] Snippet info: {start_time:.1f}s to {end_time:.1f}s")
        
        return jsonify(snippet_info)
        
    except Exception as e:
        logger.error(f"[{request_id}] Snippet info error: {str(e)}")
        return jsonify({'error': f'Snippet info failed: {str(e)}'}), 500

@app.route('/status')
def status():
    """Get system status with comprehensive logging"""
    try:
        logger.info("=== STATUS REQUEST ===")
        
        # Check if video file exists using absolute path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_path = os.path.join(base_dir, DEFAULT_VIDEO)
        main_py_path = os.path.join(base_dir, 'main.py')
        enhanced_py_path = os.path.join(base_dir, 'main_enhanced.py')
        
        logger.info(f"Base directory: {base_dir}")
        logger.info(f"Default video path: {video_path}")
        logger.info(f"Main.py path: {main_py_path}")
        logger.info(f"Enhanced main path: {enhanced_py_path}")
        
        video_exists = os.path.exists(video_path)
        main_py_exists = os.path.exists(main_py_path)
        enhanced_py_exists = os.path.exists(enhanced_py_path)
        
        logger.info(f"Video exists: {video_exists}")
        logger.info(f"main.py exists: {main_py_exists}")
        logger.info(f"main_enhanced.py exists: {enhanced_py_exists}")
        
        # Check model manager status
        manager_status = "not_initialized"
        manager_error = None
        try:
            manager = get_enhanced_manager()
            if manager:
                manager_status = "initialized"
                session_info = manager.get_session_info()
                if session_info:
                    manager_status = "video_loaded"
                    logger.info(f"Manager has video session: {session_info.get('video_path', 'N/A')}")
                else:
                    logger.info("Manager initialized but no video session")
            else:
                manager_status = "failed"
                manager_error = "Manager initialization returned None"
                logger.error("Manager initialization failed")
        except Exception as e:
            manager_status = "error"
            manager_error = str(e)
            logger.error(f"Manager initialization error: {str(e)}")
        
        # Get video info if exists
        video_info = {}
        if video_exists:
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    video_info = {
                        'fps': fps,
                        'frames': frame_count,
                        'duration': f"{duration:.2f}s"
                    }
                    logger.info(f"Video info: {json.dumps(video_info, indent=2)}")
                cap.release()
            except Exception as e:
                video_info = {'error': f'Could not read video properties: {str(e)}'}
                logger.error(f"Video info error: {str(e)}")
        
        result = {
            'status': 'online',
            'video_available': video_exists,
            'main_py_available': main_py_exists,
            'enhanced_py_available': enhanced_py_exists,
            'video_path': DEFAULT_VIDEO,
            'full_video_path': video_path,
            'main_py_path': main_py_path,
            'enhanced_py_path': enhanced_py_path,
            'video_info': video_info,
            'manager_status': manager_status,
            'manager_error': manager_error,
            'max_frames': 160,
            'max_fps': 5.0,
            'max_time_bound': 600,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Status result: {json.dumps(result, indent=2, default=str)}")
        logger.info("=== STATUS REQUEST END ===")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Status check exception: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

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
def get_video_caption_data(video_file):
    """Get caption data for a specific video"""
    try:
        db = get_caption_db()
        caption_data = db.get_video_caption(video_file)
        
        if caption_data:
            return jsonify({
                'success': True,
                'data': caption_data
            })
        else:
            return jsonify({'error': 'No caption found for this video'}), 404
            
    except Exception as e:
        logger.error(f"Error fetching caption for video {video_file}: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Debug API to reset storage
@app.route('/api/debug-reset-storage', methods=['POST'])
def debug_reset_storage():
    """Debug endpoint to force reset the storage instance"""
    global caption_db
    try:
        caption_db = None  # Reset the global instance
        new_db = get_caption_db()  # This will create a new instance with correct path
        
        # Test the new instance
        test_videos = new_db.get_all_videos_with_captions()
        
        return jsonify({
            'success': True, 
            'message': 'Storage instance reset successfully',
            'videos_found': len(test_videos),
            'db_path': new_db.db_path
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


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
                else:
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
    
    # Perform startup cleanup
    startup_cleanup()
    
    app.run(
        host='0.0.0.0',
        port=8088,
        debug=False,  # Disable debug mode to prevent duplicate model loading
        threaded=True
    )