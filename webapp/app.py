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

# Add parent directory to path to access main_enhanced.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import enhanced model manager
from main_enhanced import EnhancedModelManager

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

def get_enhanced_manager():
    """Get or create the enhanced model manager"""
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
                
                # Try to get video duration using ffprobe
                try:
                    import subprocess
                    result = subprocess.run([
                        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                        '-of', 'csv=p=0', video_path
                    ], capture_output=True, text=True, timeout=5)
                    
                    duration = float(result.stdout.strip()) if result.stdout.strip() else 0
                    duration_str = f"{int(duration//60)}:{int(duration%60):02d}" if duration > 0 else "Unknown"
                except:
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
        num_frames = int(data.get('num_frames', 120))
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

if __name__ == '__main__':
    print("🚀 Starting InternVideo2.5 Enhanced Web Interface...")
    print(f"📹 Default video: {DEFAULT_VIDEO}")
    print(f"🌐 Access at: http://localhost:8088")
    print(f"📊 Max capacity: 160 frames")
    print(f"💬 Features: Chat, Streaming, Video Caching")
    
    app.run(
        host='0.0.0.0',
        port=8088,
        debug=False,  # Set to False for production
        threaded=True
    )