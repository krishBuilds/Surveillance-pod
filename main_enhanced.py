#!/usr/bin/env python3
"""
InternVideo2.5 Enhanced Model Manager
Uses singleton ModelManager with separated video preprocessing and chat queries
"""

import os
import sys
import time
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Chat message data structure"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Optional[Dict] = None

@dataclass
class VideoSession:
    """Video session data structure for preprocessed video with temporal data"""
    video_path: str
    num_frames: int
    fps_sampling: Optional[float]
    time_bound: Optional[float]
    video_info: Optional[Dict]
    preprocessing_time: float
    frame_timestamps: List[float] = None  # Timestamps for each frame
    video_fps: float = 0.0  # Video FPS for temporal calculations
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.frame_timestamps is None:
            self.frame_timestamps = []

class EnhancedModelManager:
    """
    Enhanced model manager that uses the singleton ModelManager
    Separates video preprocessing from chat queries for multiple conversations
    """
    
    def __init__(
        self,
        model_path: str = "./models/InternVideo2_5",
        device: str = "auto",
        precision: str = "bf16",
        config_path: str = "config.yaml"
    ):
        """Initialize enhanced manager with singleton ModelManager"""
        self.model_manager = ModelManager(
            model_path=model_path,
            device=device,
            precision=precision,
            config_path=config_path
        )
        
        # Preprocessed video storage (cleared on each new video)
        self.current_session: Optional[VideoSession] = None
        self.preprocessed_frames = None  # pixel_values tensor
        self.num_patches_list = None     # patch information
        self.chat_history: List[ChatMessage] = []
        
        logger.info("Enhanced Model Manager initialized with singleton ModelManager")
    
    def initialize_model(self) -> bool:
        """Load the singleton model"""
        logger.info("Loading singleton model...")
        try:
            result = self.model_manager.load_model()
            if result:
                logger.info(f"Model loaded successfully!")
                logger.info(f"Model quantization: 8-bit={self.model_manager.enable_8bit}, 4-bit={self.model_manager.enable_4bit}")
                logger.info(f"Device: {self.model_manager.device}")
                logger.info(f"Model path: {self.model_manager.model_path}")
                return True
            else:
                logger.error("Model loading returned False - check model files and dependencies")
                return False
        except Exception as e:
            logger.error(f"Exception during model loading: {str(e)}", exc_info=True)
            return False
    
    def preprocess_video(
        self,
        video_path: str,
        num_frames: int = None,
        fps_sampling: float = None,
        time_bound: float = None
    ) -> Generator[Dict, None, None]:
        """
        Preprocess video frames and store for multiple chat queries.
        Clears previous preprocessing.
        """
        if not self.model_manager.model:
            yield {
                'type': 'status',
                'message': 'Model not loaded yet, initializing automatically...',
                'timestamp': datetime.now().isoformat()
            }
            
            # Auto-initialize the model
            logger.info("Model not loaded, auto-initializing...")
            if not self.initialize_model():
                yield {
                    'type': 'error',
                    'message': 'Failed to initialize model automatically',
                    'timestamp': datetime.now().isoformat()
                }
                return
            
            yield {
                'type': 'status',
                'message': 'Model initialized successfully, proceeding with video processing',
                'timestamp': datetime.now().isoformat()
            }
        
        # Clear previous session
        self._clear_preprocessed_data()
        
        yield {
            'type': 'status',
            'message': f'Starting video preprocessing: {os.path.basename(video_path)}',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            start_time = time.time()
            
            # Auto-select optimal frame count if not specified
            if num_frames is None:
                num_frames = self.model_manager.get_optimal_frame_count()
                logger.info(f"Auto-selected {num_frames} frames based on quantization settings")
            
            # Get video info
            video_info = self._get_video_info(video_path)
            
            yield {
                'type': 'status',
                'message': f'Loading {num_frames} frames from video...',
                'timestamp': datetime.now().isoformat()
            }
            
            # Prepare parameters for video loading
            # If time_bound is specified, it limits the video to first N seconds
            # If fps_sampling is specified, it samples at that FPS rate within the time bound
            # Otherwise, it distributes num_frames evenly across the time bound or whole video
            
            logger.info(f"Video loading params: num_frames={num_frames}, fps_sampling={fps_sampling}, time_bound={time_bound}")
            
            # Convert time_bound to bound tuple format if specified
            bound = None
            if time_bound:
                bound = (0, time_bound)  # From start to time_bound seconds
                logger.info(f"Using time bound: 0 to {time_bound} seconds")
            
            # Load video frames using the same method as original ModelManager
            pixel_values, num_patches_list = self.model_manager._load_video_for_inference(
                video_path,
                num_segments=num_frames,
                bound=bound,  # Use bound tuple instead of time_bound
                fps_sampling=fps_sampling,
                time_bound=time_bound  # Still pass for custom FPS logic
            )
            
            # Handle tensor dtype based on quantization settings (same as original)
            if self.model_manager.enable_8bit or self.model_manager.enable_4bit:
                pixel_values = pixel_values.to(torch.float16).to(self.model_manager.model.device)
                logger.info("Using float16 precision for quantized model compatibility")
            else:
                pixel_values = pixel_values.to(torch.bfloat16).to(self.model_manager.model.device)
                logger.info("Using bfloat16 precision for full precision model")
            
            preprocessing_time = time.time() - start_time
            
            # Store preprocessed data
            self.preprocessed_frames = pixel_values
            self.num_patches_list = num_patches_list
            
            # Create session record with temporal data
            self.current_session = VideoSession(
                video_path=video_path,
                num_frames=len(num_patches_list),
                fps_sampling=fps_sampling,
                time_bound=time_bound,
                video_info=video_info,
                preprocessing_time=preprocessing_time,
                frame_timestamps=self.model_manager.current_frame_timestamps.copy(),
                video_fps=self.model_manager.current_video_fps
            )
            
            logger.info(f"Video preprocessing completed: {len(num_patches_list)} frames in {preprocessing_time:.2f}s")
            
            yield {
                'type': 'video_processed',
                'message': f'Video preprocessed successfully: {len(num_patches_list)} frames cached',
                'frames_processed': len(num_patches_list),
                'video_info': video_info,
                'preprocessing_time': preprocessing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Video preprocessing error: {str(e)}"
            logger.error(error_msg)
            yield {
                'type': 'error',
                'message': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    def chat_query(
        self,
        prompt: str,
        max_tokens: int = 1024,
        enable_temporal_analysis: bool = True
    ) -> Generator[Dict, None, None]:
        """
        Process chat query using preprocessed video frames with temporal analysis.
        Returns streaming response chunks with timestamps when relevant.
        """
        if not self.model_manager.model:
            yield {
                'type': 'status',
                'message': 'Model not loaded yet, initializing automatically...',
                'timestamp': datetime.now().isoformat()
            }
            
            # Auto-initialize the model
            logger.info("Model not loaded, auto-initializing for chat...")
            if not self.initialize_model():
                yield {
                    'type': 'error',
                    'message': 'Failed to initialize model automatically',
                    'timestamp': datetime.now().isoformat()
                }
                return
            
            yield {
                'type': 'status',
                'message': 'Model initialized successfully, proceeding with chat',
                'timestamp': datetime.now().isoformat()
            }
        
        if not self.current_session or self.preprocessed_frames is None:
            yield {
                'type': 'error',
                'message': 'No video preprocessed. Please preprocess a video first.',
                'timestamp': datetime.now().isoformat()
            }
            return
        
        if not prompt.strip():
            yield {
                'type': 'error',
                'message': 'Prompt cannot be empty',
                'timestamp': datetime.now().isoformat()
            }
            return
        
        # Add user message to chat history
        user_msg = ChatMessage(
            role='user',
            content=prompt,
            timestamp=datetime.now().isoformat()
        )
        self.chat_history.append(user_msg)
        
        yield {
            'type': 'user_message',
            'content': prompt,
            'timestamp': user_msg.timestamp
        }
        
        yield {
            'type': 'status',
            'message': 'Generating response using preprocessed video...',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            logger.info(f"Processing chat query: {prompt[:100]}...")
            
            # Build question with appropriate context based on temporal analysis setting
            if enable_temporal_analysis and self.current_session and self.current_session.frame_timestamps:
                # Enhanced temporal context with explicit timestamp instructions
                temporal_context = f"TEMPORAL VIDEO ANALYSIS:\n"
                temporal_context += f"- Video duration: {self.current_session.frame_timestamps[0]:.1f}s to {self.current_session.frame_timestamps[-1]:.1f}s\n"
                temporal_context += f"- Total frames: {len(self.num_patches_list)} frames\n"
                
                if len(self.current_session.frame_timestamps) > 1:
                    frame_interval = (self.current_session.frame_timestamps[-1] - self.current_session.frame_timestamps[0]) / (len(self.current_session.frame_timestamps) - 1)
                    temporal_context += f"- Frame sampling: every {frame_interval:.1f}s\n"
                
                temporal_context += f"\nIMPORTANT: When answering temporal questions like 'when does X appear', provide SPECIFIC TIMESTAMPS in seconds.\n"
                temporal_context += f"Examples of proper temporal responses:\n"
                temporal_context += f"- 'The bird appears at 15.3 seconds'\n"
                temporal_context += f"- 'The action starts around 23.7 seconds'\n"
                temporal_context += f"- 'At 45.2 seconds, the character enters'\n"
                temporal_context += f"\nAlways include precise timing when describing video events.\n\n"
                
                video_prefix = "".join([f"Frame{i+1} (t={self.current_session.frame_timestamps[i]:.1f}s): <image>\n" for i in range(len(self.num_patches_list))])
                question = temporal_context + video_prefix + f"\nQUESTION: {prompt}\n\nProvide your answer with specific timestamps:"
            else:
                # Standard video analysis format without temporal instructions
                video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(self.num_patches_list))])
                question = video_prefix + prompt
            
            # Prepare generation config (same as original)
            generation_config = dict(
                do_sample=True,
                temperature=0.7,
                max_new_tokens=max_tokens,
                top_p=0.9,
                num_beams=1,
                pad_token_id=self.model_manager.tokenizer.eos_token_id
            )
            
            # Build conversation history for context
            # Format: List[Tuple[str, str]] where each tuple is (user_query, assistant_response)
            # Note: We exclude the current user message since it's already added to chat_history
            conversation_history = []
            for i in range(0, len(self.chat_history) - 1, 2):  # -1 to exclude current message
                if i + 1 < len(self.chat_history):
                    user_msg = self.chat_history[i]
                    assistant_msg = self.chat_history[i + 1]
                    if user_msg.role == 'user' and assistant_msg.role == 'assistant':
                        conversation_history.append((user_msg.content, assistant_msg.content))
            
            logger.info(f"Using conversation history with {len(conversation_history)} previous exchanges")
            if conversation_history:
                logger.debug(f"Last exchange - User: '{conversation_history[-1][0][:50]}...' Assistant: '{conversation_history[-1][1][:50]}...'")
            
            # Run inference using preprocessed frames
            inference_start = time.time()
            
            with torch.no_grad():
                full_output, _ = self.model_manager.model.chat(
                    self.model_manager.tokenizer,
                    self.preprocessed_frames,  # Use preprocessed frames
                    question,
                    generation_config,
                    num_patches_list=self.num_patches_list,
                    history=conversation_history,
                    return_history=True
                )
            
            inference_time = time.time() - inference_start
            
            # Simulate streaming by sending word chunks
            words = full_output.split()
            response_text = ""
            
            for i, word in enumerate(words):
                if i == 0:
                    response_text = word
                else:
                    response_text += " " + word
                
                yield {
                    'type': 'response_chunk',
                    'content': word,
                    'full_content': response_text,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Small delay to simulate streaming
                time.sleep(0.05)
            
            # Process response for temporal references only if temporal analysis is enabled
            temporal_data = None
            if enable_temporal_analysis and self.current_session:
                temporal_data = self._extract_temporal_references(response_text)
            else:
                # When temporal analysis is disabled, return empty temporal data
                temporal_data = {'has_temporal_references': False}
            
            # Add assistant message to chat history
            assistant_msg = ChatMessage(
                role='assistant',
                content=response_text,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'inference_time': inference_time,
                    'frames_processed': len(self.num_patches_list),
                    'video_session': self.current_session.video_path,
                    'temporal_data': temporal_data
                }
            )
            self.chat_history.append(assistant_msg)
            
            yield {
                'type': 'response_complete',
                'full_content': response_text,
                'metadata': assistant_msg.metadata,
                'timestamp': assistant_msg.timestamp,
                'temporal_data': temporal_data
            }
            
            logger.info(f"Chat query completed in {inference_time:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Chat query error: {str(e)}"
            logger.error(error_msg)
            
            yield {
                'type': 'error',
                'message': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_chat_history(self) -> List[Dict]:
        """Get chat history as serializable format"""
        return [asdict(msg) for msg in self.chat_history]
    
    def clear_chat_history(self):
        """Clear chat history only (keep preprocessed video)"""
        self.chat_history.clear()
        logger.info("Chat history cleared")
    
    def get_session_info(self) -> Optional[Dict]:
        """Get current session information"""
        if self.current_session:
            return asdict(self.current_session)
        return None
    
    def _clear_preprocessed_data(self):
        """Clear preprocessed video data and chat history"""
        if self.current_session:
            logger.info(f"Clearing previous video session: {self.current_session.video_path}")
            logger.info(f"Previous session had {self.current_session.num_frames} frames")
        else:
            logger.info("No previous video session to clear")
        
        # Clear session data
        self.current_session = None
        
        # Clear preprocessed tensors
        if self.preprocessed_frames is not None:
            logger.info("Clearing preprocessed frames from GPU memory")
            del self.preprocessed_frames
            self.preprocessed_frames = None
        
        if self.num_patches_list is not None:
            logger.info(f"Clearing patch list ({len(self.num_patches_list)} patches)")
            self.num_patches_list = None
        
        # Clear chat history
        if self.chat_history:
            logger.info(f"Clearing chat history ({len(self.chat_history)} messages)")
            self.chat_history.clear()
        
        # Clear model's processing cache (embeddings, tokens) while keeping model alive
        if self.model_manager:
            try:
                self.model_manager.clear_processing_cache()
            except Exception as e:
                logger.warning(f"Model cache cleanup warning: {e}")
        
        logger.info("Previous video session completely cleared")
    
    def _get_video_info(self, video_path: str) -> Dict:
        """Get video information using OpenCV"""
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
                    'duration': f"{duration:.2f}s",
                    'duration_seconds': duration
                }
                cap.release()
                return video_info
            else:
                return {'error': 'Could not open video file'}
        except Exception as e:
            return {'error': f'Could not read video properties: {str(e)}'}
    
    def cleanup(self):
        """Cleanup resources"""
        self._clear_preprocessed_data()
        # Don't cleanup the singleton model manager - let it stay loaded
        logger.info("Enhanced manager cleaned up (singleton model kept loaded)")
    
    def _extract_temporal_references(self, response_text: str) -> Optional[Dict]:
        """
        Extract temporal references from AI response text.
        Returns timestamps and relevant information for video snippet generation.
        """
        try:
            # Enhanced patterns to match timestamps in various formats
            time_patterns = [
                r'(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b',  # "5.2 seconds", "10s"
                r'at\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b',  # "at 15.5 seconds"
                r'around\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b',  # "around 20 seconds"
                r'appears\s+at\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b',  # "appears at 15.5 seconds"
                r'starts\s+around\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b',  # "starts around 23.7 seconds"
                r'happens\s+at\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b',  # "happens at 30 seconds"
                r'(\d+):(\d+)(?:\.(\d+))?',  # "1:30", "1:30.5" (mm:ss format)
                r't=(\d+(?:\.\d+)?)\s*s',  # "t=15.2s" (frame timestamp format)
                r'(\d+(?:\.\d+)?)\s*(?:second|sec)',  # "15.3 second", "20 sec"
                r'beginning|start',  # Match "beginning" or "start" as 0s
                r'end|ending|final',  # Match "end" as video end time
                r'middle|halfway',  # Match "middle" as video midpoint
            ]
            
            extracted_times = []
            
            for pattern in time_patterns:
                matches = re.finditer(pattern, response_text, re.IGNORECASE)
                for match in matches:
                    if ':' in match.group(0):  # Handle mm:ss format
                        minutes = float(match.group(1))
                        seconds = float(match.group(2))
                        milliseconds = float(match.group(3) or 0) / 1000 if match.group(3) else 0
                        timestamp = minutes * 60 + seconds + milliseconds
                        text_context = match.group(0)
                    elif match.group(0).lower() in ['beginning', 'start']:
                        timestamp = 0.0
                        text_context = match.group(0)
                    elif match.group(0).lower() in ['end', 'ending', 'final']:
                        # Use video end time
                        if self.current_session and self.current_session.frame_timestamps:
                            timestamp = self.current_session.frame_timestamps[-1]
                        else:
                            timestamp = 120.0  # Default fallback
                        text_context = match.group(0)
                    elif match.group(0).lower() in ['middle', 'halfway']:
                        # Use video midpoint
                        if self.current_session and self.current_session.frame_timestamps:
                            timestamp = self.current_session.frame_timestamps[-1] / 2
                        else:
                            timestamp = 60.0  # Default fallback
                        text_context = match.group(0)
                    else:
                        try:
                            timestamp = float(match.group(1))
                            text_context = match.group(0)
                        except (ValueError, IndexError):
                            continue  # Skip invalid matches
                    
                    # Only include timestamps within video bounds
                    if (self.current_session and 
                        self.current_session.frame_timestamps and
                        0 <= timestamp <= self.current_session.frame_timestamps[-1] + 5):  # Allow 5s buffer
                        extracted_times.append({
                            'timestamp': timestamp,
                            'text_context': text_context,
                            'sentence': self._extract_sentence_context(response_text, match.start(), match.end())
                        })
            
            if extracted_times:
                # Remove duplicates and sort by timestamp
                unique_times = []
                seen_timestamps = set()
                for time_data in extracted_times:
                    ts = round(time_data['timestamp'], 1)  # Round to 0.1s precision
                    if ts not in seen_timestamps:
                        seen_timestamps.add(ts)
                        unique_times.append(time_data)
                
                unique_times.sort(key=lambda x: x['timestamp'])
                
                logger.info(f"Found {len(unique_times)} temporal references: {[t['timestamp'] for t in unique_times]}")
                
                return {
                    'has_temporal_references': True,
                    'timestamps': unique_times,
                    'video_fps': self.current_session.video_fps if self.current_session else 0,
                    'video_path': self.current_session.video_path if self.current_session else ''
                }
            
            return {'has_temporal_references': False}
            
        except Exception as e:
            logger.error(f"Error extracting temporal references: {str(e)}")
            return {'has_temporal_references': False, 'error': str(e)}
    
    def _extract_sentence_context(self, text: str, start: int, end: int, context_chars: int = 100) -> str:
        """Extract sentence context around a matched timestamp"""
        # Find sentence boundaries around the match
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)
        
        # Try to find sentence boundaries
        before_text = text[context_start:start]
        after_text = text[end:context_end]
        
        # Find the last sentence start before the match
        sentence_starts = [m.end() for m in re.finditer(r'[.!?]\s+', before_text)]
        if sentence_starts:
            context_start = context_start + sentence_starts[-1]
        
        # Find the first sentence end after the match  
        sentence_end = re.search(r'[.!?]', after_text)
        if sentence_end:
            context_end = end + sentence_end.end()
        
        return text[context_start:context_end].strip()
    
    def generate_video_snippet(
        self,
        timestamp: float,
        duration: float = 10.0,
        output_dir: str = "outputs/snippets"
    ) -> Optional[Dict]:
        """
        Generate a video snippet around the specified timestamp.
        
        Args:
            timestamp: Center timestamp for the snippet (seconds)
            duration: Total duration of the snippet (seconds)
            output_dir: Directory to save the snippet
            
        Returns:
            Dictionary with snippet information or None if failed
        """
        if not self.current_session or not self.current_session.video_path:
            logger.error("No video session available for snippet generation")
            return None
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Calculate start and end times
            start_time = max(0, timestamp - duration / 2)
            end_time = timestamp + duration / 2
            
            # Generate output filename
            video_name = Path(self.current_session.video_path).stem
            snippet_filename = f"{video_name}_snippet_{timestamp:.1f}s_{duration:.1f}s.mp4"
            output_path = os.path.join(output_dir, snippet_filename)
            
            # Use ffmpeg to extract the snippet
            cmd = [
                'ffmpeg', '-y',  # Overwrite existing files
                '-i', self.current_session.video_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264',  # Video codec
                '-c:a', 'aac',      # Audio codec
                '-avoid_negative_ts', 'make_zero',
                output_path
            ]
            
            logger.info(f"Generating video snippet: {start_time:.1f}s to {end_time:.1f}s")
            
            # Run ffmpeg command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                snippet_info = {
                    'success': True,
                    'output_path': output_path,
                    'filename': snippet_filename,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'target_timestamp': timestamp,
                    'file_size': os.path.getsize(output_path)
                }
                
                logger.info(f"Video snippet generated successfully: {snippet_filename}")
                return snippet_info
            else:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return {
                    'success': False,
                    'error': f"FFmpeg error: {result.stderr}",
                    'command': ' '.join(cmd)
                }
                
        except subprocess.TimeoutExpired:
            logger.error("Video snippet generation timed out")
            return {'success': False, 'error': 'Snippet generation timed out'}
        except Exception as e:
            logger.error(f"Error generating video snippet: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def generate_snippets_for_response(
        self, 
        temporal_data: Dict, 
        max_snippets: int = 3
    ) -> List[Dict]:
        """
        Generate video snippet info for all timestamps found in a response.
        Returns streaming info instead of generating files.
        
        Args:
            temporal_data: Temporal data from _extract_temporal_references
            max_snippets: Maximum number of snippets to generate
            
        Returns:
            List of snippet streaming information dictionaries
        """
        if not temporal_data or not temporal_data.get('has_temporal_references'):
            return []
        
        snippets = []
        timestamps = temporal_data.get('timestamps', [])[:max_snippets]
        
        for time_data in timestamps:
            timestamp = time_data['timestamp']
            duration = 10.0  # Default 10 second clips
            
            # Calculate start and end times - 1 second back, rest forward
            start_time = max(0, timestamp - 1.0)  # Only 1 second before target
            end_time = timestamp + (duration - 1.0)  # Remaining duration after target
            
            # Get video duration from current session
            video_duration = 0
            if self.current_session and self.current_session.frame_timestamps:
                video_duration = self.current_session.frame_timestamps[-1]
                end_time = min(end_time, video_duration)
            
            snippet_info = {
                'success': True,
                'video_url': '/video-stream',
                'start_time': start_time,
                'end_time': end_time,
                'target_timestamp': timestamp,
                'duration': end_time - start_time,
                'description': time_data['sentence'],
                'original_text': time_data['text_context'],
                'type': 'stream_segment'
            }
            snippets.append(snippet_info)
        
        logger.info(f"Generated {len(snippets)} video stream segments from {len(timestamps)} timestamps")
        return snippets

def main():
    """Example usage of EnhancedModelManager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced InternVideo2.5 with Preprocessing")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--num-frames", type=int, default=120, help="Number of frames")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        return
    
    # Initialize enhanced manager
    manager = EnhancedModelManager()
    
    # Load model
    if not manager.initialize_model():
        print("❌ Failed to load model")
        return
    
    try:
        # Preprocess video
        print("🎬 Preprocessing video...")
        for chunk in manager.preprocess_video(args.video, args.num_frames):
            if chunk['type'] == 'video_processed':
                print(f"✅ {chunk['message']}")
                print(f"📊 Processing time: {chunk['preprocessing_time']:.2f}s")
                break
            elif chunk['type'] == 'error':
                print(f"❌ {chunk['message']}")
                return
        
        if args.chat:
            # Interactive chat
            print("\n💬 Chat mode - Ask questions about the video:")
            print("(Type 'quit' to exit)")
            
            while True:
                query = input("\n🎬 You: ").strip()
                if query.lower() == 'quit':
                    break
                
                if not query:
                    continue
                
                print("🤖 Assistant: ", end="", flush=True)
                
                for chunk in manager.chat_query(query):
                    if chunk['type'] == 'response_chunk':
                        print(chunk['content'], end=" ", flush=True)
                    elif chunk['type'] == 'response_complete':
                        print(f"\n⏱️ {chunk['metadata']['inference_time']:.2f}s")
                        break
                    elif chunk['type'] == 'error':
                        print(f"\n❌ {chunk['message']}")
                        break
        else:
            # Single query example
            print("\n🤔 Example query...")
            for chunk in manager.chat_query("Describe what you see in this video"):
                if chunk['type'] == 'response_complete':
                    print(f"🤖 Response: {chunk['full_content'][:200]}...")
                    print(f"⏱️ Time: {chunk['metadata']['inference_time']:.2f}s")
                    break
    
    finally:
        manager.cleanup()

if __name__ == "__main__":
    main()