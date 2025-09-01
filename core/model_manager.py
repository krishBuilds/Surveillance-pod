#!/usr/bin/env python3
"""
InternVideo2.5 Model Manager
Core module for loading and managing the video understanding model
"""

import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
import cv2
import av
from typing import List, Union, Optional, Dict
import logging
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import yaml
import time
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Singleton model manager for InternVideo2.5 - ensures only one model instance"""
    
    _instance = None
    _shared_model = None
    _shared_tokenizer = None
    _shared_model_path = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern - only one instance allowed"""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self, 
        model_path: str = "./models/InternVideo2_5",
        device: str = "auto",
        precision: str = "bf16",
        config_path: str = "config.yaml"
    ):
        """
        Initialize the model manager (singleton pattern)
        
        Args:
            model_path: Path to the downloaded model
            device: Device to use ('auto', 'cuda', 'cpu')
            precision: Model precision ('bf16', 'fp16', 'fp32')
            config_path: Path to configuration file
        """
        # Only initialize once
        if not hasattr(self, '_initialized'):
            # Load configuration
            self.config = self._load_config(config_path)
            
            self.model_path = model_path
            self.device = self._setup_device(device)
            self.precision = precision
            
            # Quantization settings from config
            self.enable_8bit = self.config.get('model', {}).get('quantization', {}).get('enable_8bit', False)
            self.enable_4bit = self.config.get('model', {}).get('quantization', {}).get('enable_4bit', False)
            
            # Frame limits from config
            self.frame_limits = self.config.get('inference', {}).get('frame_limits', {})
            self.fps_limits = self.config.get('inference', {}).get('fps_limits', {})
            
            self.model = None
            self.tokenizer = None
            
            # Temporal analysis attributes
            self.current_frame_timestamps = []
            self.current_video_fps = 0.0
            self.current_video_path = ""
            
            self._initialized = True
            
            logger.info(f"ModelManager initialized with 8-bit: {self.enable_8bit}, 4-bit: {self.enable_4bit}")
        else:
            logger.info("ModelManager already initialized - using existing instance")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}
    
    def get_optimal_frame_count(self) -> int:
        """Get optimal frame count based on quantization settings"""
        if self.enable_4bit:
            max_frames = self.frame_limits.get('int4_max_frames', 200)
            logger.info(f"Using 4-bit quantization - max frames: {max_frames}")
        elif self.enable_8bit:
            max_frames = self.frame_limits.get('int8_max_frames', 150)
            logger.info(f"Using 8-bit quantization - max frames: {max_frames}")
        else:
            max_frames = self.frame_limits.get('bf16_max_frames', 48)
            logger.info(f"Using BF16 precision - max frames: {max_frames}")
        
        return max_frames
        
    def _setup_device(self, device: str) -> str:
        """Setup and validate device - GPU only, no CPU fallback"""
        if device == "auto":
            device = "cuda"
        
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available but GPU required. No CPU fallback.")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return "cuda"
        else:
            raise RuntimeError("Only CUDA device supported. CPU fallback disabled.")
    
    def load_model(self):
        """Load the InternVideo2.5 model - reuse if already loaded"""
        
        # Check if model is already loaded with same path
        if (ModelManager._shared_model is not None and 
            ModelManager._shared_tokenizer is not None and 
            ModelManager._shared_model_path == self.model_path):
            logger.info("Reusing already loaded model to save GPU memory")
            self.model = ModelManager._shared_model
            self.tokenizer = ModelManager._shared_tokenizer
            return True
        
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Prepare model loading arguments
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
                "low_cpu_mem_usage": True
            }
            
            # Add quantization settings using BitsAndBytesConfig
            if self.enable_4bit:
                logger.info("Loading model with 4-bit quantization...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16  # Fixed: use float16 instead of bfloat16
                )
                model_kwargs["quantization_config"] = bnb_config
            elif self.enable_8bit:
                logger.info("Loading model with 8-bit quantization...")
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16  # Fixed: explicit compute dtype
                )
                model_kwargs["quantization_config"] = bnb_config
            else:
                # Determine dtype based on precision
                dtype_map = {
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                    "fp32": torch.float32
                }
                dtype = dtype_map.get(self.precision, torch.bfloat16)
                model_kwargs["torch_dtype"] = dtype
                logger.info(f"Loading model with {self.precision} precision on GPU...")
            
            self.model = AutoModel.from_pretrained(
                self.model_path,
                **model_kwargs
            ).eval()
            
            # Store in shared variables
            ModelManager._shared_model = self.model
            ModelManager._shared_tokenizer = self.tokenizer
            ModelManager._shared_model_path = self.model_path
            
            # Log quantization status
            if self.enable_4bit:
                logger.info("Model loaded successfully with 4-bit quantization!")
            elif self.enable_8bit:
                logger.info("Model loaded successfully with 8-bit quantization!")
            else:
                logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def process_video(
        self,
        video_path: str,
        prompt: str,
        num_frames: int = None,
        max_tokens: int = 1024,
        fps_sampling: float = None,
        time_bound: float = None
    ) -> Dict:
        """
        Process a video with the model using real inference
        
        Args:
            video_path: Path to video file
            prompt: Text prompt for the model
            num_frames: Number of frames to sample
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with results
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Auto-select optimal frame count based on quantization if not specified
        if num_frames is None:
            auto_select = self.frame_limits.get('auto_select_frames', True)
            if auto_select:
                num_frames = self.get_optimal_frame_count()
                logger.info(f"Auto-selected {num_frames} frames based on quantization settings")
            else:
                num_frames = 32  # Default fallback
        
        result = {
            "video_path": video_path,
            "prompt": prompt,
            "num_frames": num_frames,
            "response": "",
            "error": None
        }
        
        try:
            import time
            start_time = time.time()
            
            # Load and process video using decord (like in demo)
            logger.info(f"Loading video: {video_path}")
            video_load_start = time.time()
            pixel_values, num_patches_list = self._load_video_for_inference(
                video_path, num_segments=num_frames, max_num=1,
                fps_sampling=fps_sampling, time_bound=time_bound
            )
            video_load_time = time.time() - video_load_start
            logger.info(f"Video loading completed in {video_load_time:.2f} seconds")
            
            # Handle tensor dtype based on quantization settings
            if self.enable_8bit or self.enable_4bit:
                # Use float16 for quantized models to avoid dtype mismatch
                pixel_values = pixel_values.to(torch.float16).to(self.model.device)
                logger.info("Using float16 precision for quantized model compatibility")
            else:
                # Use bfloat16 for non-quantized models
                pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
                logger.info("Using bfloat16 precision for full precision model")
            
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
            
            # Prepare generation config
            generation_config = dict(
                do_sample=False,
                temperature=0.0,
                max_new_tokens=max_tokens,
                top_p=0.1,
                num_beams=1
            )
            
            # Build full question with video frames
            question = video_prefix + prompt
            
            logger.info(f"Running inference on {len(num_patches_list)} frames...")
            inference_start = time.time()
            
            # Run actual inference
            with torch.no_grad():
                output, _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True
                )
            
            inference_time = time.time() - inference_start
            total_time = time.time() - start_time
            
            result["response"] = output
            result["timing"] = {
                "video_loading_time": video_load_time,
                "inference_time": inference_time,
                "total_processing_time": total_time,
                "frames_per_second": len(num_patches_list) / total_time
            }
            
            logger.info(f"Video inference completed successfully in {inference_time:.2f} seconds")
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(f"Processing rate: {len(num_patches_list) / total_time:.2f} frames/second")
            logger.info(f"AI Response length: {len(output)} characters")
            logger.info(f"AI Response preview: {output[:100]}...")
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _extract_frames(self, video_path: str, num_frames: int) -> List[np.ndarray]:
        """Extract frames from video"""
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        total_frames = stream.frames
        if total_frames == 0:
            total_frames = int(stream.duration * stream.average_rate)
        
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        
        frames = []
        frame_idx = 0
        
        for frame in container.decode(stream):
            if frame_idx in indices:
                frames.append(np.array(frame.to_image()))
                if len(frames) == num_frames:
                    break
            frame_idx += 1
        
        container.close()
        return frames
    
    def process_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 512
    ) -> Dict:
        """
        Process an image with the model using real inference
        
        Args:
            image_path: Path to image file
            prompt: Text prompt for the model
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with results
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        result = {
            "image_path": image_path,
            "prompt": prompt,
            "response": "",
            "error": None
        }
        
        try:
            # Load and preprocess image
            logger.info(f"Loading image: {image_path}")
            image = Image.open(image_path).convert('RGB')
            pixel_values = self._load_image_for_inference(image)
            
            # Handle tensor dtype based on quantization settings
            if self.enable_8bit or self.enable_4bit:
                # Use float16 for quantized models to avoid dtype mismatch
                pixel_values = pixel_values.to(torch.float16).to(self.model.device)
                logger.info("Using float16 precision for quantized model compatibility")
            else:
                # Use bfloat16 for non-quantized models
                pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
                logger.info("Using bfloat16 precision for full precision model")
            
            # Prepare generation config
            generation_config = dict(
                do_sample=False,
                temperature=0.0,
                max_new_tokens=max_tokens,
                top_p=0.1,
                num_beams=1
            )
            
            # Build question with image token
            question = "<image>\n" + prompt
            
            logger.info("Running image inference...")
            
            # Run actual inference
            with torch.no_grad():
                output, _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=None,
                    return_history=True
                )
            
            result["response"] = output
            logger.info("Image inference completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def cleanup(self, force_cleanup_shared=False):
        """Clean up model from memory"""
        # Only clean instance references, keep shared model loaded
        if self.model:
            self.model = None
        
        if self.tokenizer:
            self.tokenizer = None
        
        # Clear temporal data
        self.current_frame_timestamps = []
        self.current_video_fps = 0.0
        self.current_video_path = ""
        
        # Only clean shared model if explicitly requested
        if force_cleanup_shared:
            if ModelManager._shared_model:
                del ModelManager._shared_model
                ModelManager._shared_model = None
            
            if ModelManager._shared_tokenizer:
                del ModelManager._shared_tokenizer
                ModelManager._shared_tokenizer = None
            
            ModelManager._shared_model_path = None
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("Shared model cleaned up from GPU memory")
        else:
            logger.info("Instance references cleaned up (shared model kept loaded)")
    
    # === Video/Image Processing Helper Methods ===
    
    def _build_transform(self, input_size=448):
        """Build image transformation pipeline"""
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform
    
    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Find the closest aspect ratio from target ratios"""
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def _dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        """Dynamically preprocess image based on aspect ratio"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        # Calculate target ratios
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        # Find closest aspect ratio
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        
        # Calculate target dimensions
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        # Resize and split image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        
        return processed_images
    
    def _load_image_for_inference(self, image, input_size=448, max_num=1):
        """Load and preprocess image for inference (adjusted for single image)"""
        transform = self._build_transform(input_size=input_size)
        images = self._dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=False, max_num=max_num
        )
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def _get_video_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        """Get frame indices for video sampling"""
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices
    
    def _get_num_frames_by_duration(self, duration):
        """Calculate number of frames based on video duration"""
        local_num_frames = 4
        num_segments = int(duration // local_num_frames)
        
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments
        
        num_frames = min(512, num_frames)
        num_frames = max(128, num_frames)
        
        return num_frames
    
    def _load_video_for_inference(
        self, 
        video_path, 
        bound=None, 
        input_size=448, 
        max_num=1, 
        num_segments=32, 
        get_frame_by_duration=False,
        fps_sampling=None,
        time_bound=None
    ):
        """Load and preprocess video for inference using decord with timestamp tracking"""
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        logger.info(f"Video info: {max_frame + 1} frames, {fps:.2f} FPS, {(max_frame + 1) / fps:.2f}s duration")
        
        pixel_values_list, num_patches_list = [], []
        frame_timestamps = []  # Track timestamps for temporal queries
        transform = self._build_transform(input_size=input_size)
        
        # Handle frame sampling logic
        # Priority 1: If time_bound is specified, limit analysis to first N seconds
        # Priority 2: If fps_sampling is specified, sample at that FPS rate
        # Priority 3: Distribute num_segments evenly across the selected duration
        
        if time_bound:
            # Limit to first N seconds of video
            max_time_frame = min(int(time_bound * fps) - 1, max_frame)
            logger.info(f"Time bound {time_bound}s limits to frame {max_time_frame} (of {max_frame})")
            
            if fps_sampling:
                # Sample at specified FPS within time bound
                # Example: 120 frames at 1 FPS for 120 seconds
                actual_duration = min(time_bound, (max_frame + 1) / fps)
                target_frames = min(int(fps_sampling * actual_duration), num_segments)
                frame_indices = np.linspace(0, max_time_frame, target_frames).astype(int)
                logger.info(f"FPS sampling: {fps_sampling} FPS for {actual_duration:.1f}s = {target_frames} frames")
            else:
                # Distribute num_segments evenly within time bound
                frame_indices = np.linspace(0, max_time_frame, num_segments).astype(int)
                logger.info(f"Distributing {num_segments} frames evenly in first {time_bound}s")
        elif bound:
            # Use provided bound tuple (start, end) in seconds
            frame_indices = self._get_video_index(
                bound, fps, max_frame, first_idx=0, num_segments=num_segments
            )
            logger.info(f"Using bound {bound} for frame selection")
        elif get_frame_by_duration:
            # Auto-adjust based on duration
            duration = max_frame / fps
            num_segments = self._get_num_frames_by_duration(duration)
            logger.info(f"Auto-adjusted num_segments to {num_segments} based on duration")
            frame_indices = self._get_video_index(
                None, fps, max_frame, first_idx=0, num_segments=num_segments
            )
        else:
            # Default: distribute frames evenly across entire video
            frame_indices = self._get_video_index(
                None, fps, max_frame, first_idx=0, num_segments=num_segments
            )
            logger.info(f"Distributing {num_segments} frames evenly across entire video")
        
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            img = self._dynamic_preprocess(
                img, image_size=input_size, use_thumbnail=True, max_num=max_num
            )
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
            
            # Calculate and store timestamp for this frame
            timestamp = float(frame_index) / fps
            frame_timestamps.append(timestamp)
        
        pixel_values = torch.cat(pixel_values_list)
        
        # Store frame timestamps for temporal queries
        self.current_frame_timestamps = frame_timestamps
        self.current_video_fps = fps
        self.current_video_path = video_path
        
        logger.info(f"Stored {len(frame_timestamps)} frame timestamps: {frame_timestamps[0]:.2f}s to {frame_timestamps[-1]:.2f}s")
        
        return pixel_values, num_patches_list