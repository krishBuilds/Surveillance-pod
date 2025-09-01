# CLAUDE.md - Project Documentation

## Project Overview
This is an InternVideo2.5 implementation for video understanding and AI analysis on RunPod with RTX 4090 GPU, featuring enhanced video processing capabilities through 8-bit quantization and singleton architecture.

## Model Details
- **Model**: InternVideo2.5 Chat 8B (OpenGVLab/InternVideo2_5_Chat_8B)
- **Size**: ~16GB downloaded, 8.4B parameters
- **GPU**: RTX 4090 with 24GB VRAM
- **Precision**: 8-bit quantized with Float16 compute dtype
- **Peak Capacity**: 160 frames maximum with detailed narrative analysis
- **Memory Usage**: ~16.86GB model + ~4GB frames = 20.86GB total (88% utilization)

## Directory Structure
```
/workspace/surveillance/
├── core/                   # Core model management modules
├── scripts/                # Utility and test scripts  
├── ai_analysis/            # Custom AI analysis logic (pending)
├── data/                   # Input data storage
├── outputs/                # Analysis results
├── models/                 # Model files location
│   └── InternVideo2_5/    # Downloaded InternVideo2.5 model
└── main.py                # Main entry point
```

## Quick Commands

### Test GPU availability:
```bash
nvidia-smi
```

### Run video analysis (Enhanced Capacity):
```bash
# Standard analysis (up to 160 frames)
python main.py --video <video_path> --prompt "<your_prompt>"

# Extended analysis with custom frame count
python main.py --video <video_path> --prompt "<your_prompt>" --num-frames 120

# Custom FPS sampling
python main.py --video <video_path> --prompt "<your_prompt>" --fps-sampling 2.0
```

### Run image analysis:
```bash
python main.py --image <image_path> --prompt "<your_prompt>"
```

### Advanced Video Analysis Examples:
```bash
# Character development analysis
python main.py --video data/street_scene.mp4 --prompt "Analyze character development and personality traits throughout the story"

# Story structure analysis 
python main.py --video data/street_scene.mp4 --prompt "Analyze story structure and plot points" --num-frames 160
```

### Download model (if needed):
```bash
python scripts/download_model.py --model-dir ./models/InternVideo2_5
```

## Development Notes

### Model Loading
- **Singleton Architecture**: Single model instance shared across all operations
- **8-bit Quantization**: BitsAndBytesConfig reduces memory usage by 50%
- **Loading Time**: ~16 seconds (4x slower than BF16, but enables 3x frame capacity)
- **Memory Optimization**: ~16.86GB model footprint vs ~19.97GB unquantized

### Video Processing
- **Enhanced Capacity**: 160 frames maximum (3.33x improvement over baseline 48)
- **Frame Limits**: 
  - BF16: 48 frames max
  - 8-bit: 160 frames max
  - Memory boundary: 200 frames (CUDA OOM)
- **Processing Speed**: 4.92 frames/second at peak capacity
- **Video Decoding**: Uses decord for efficient frame extraction
- **Internal Logging**: Comprehensive timing metrics and AI response documentation

### AI Analysis Directory
The `ai_analysis/` directory is prepared for custom analysis implementations. Future modules can be added here for specific use cases.

## Environment Info
- Platform: Linux (RunPod)
- Python: 3.11.10
- CUDA: 12.4
- GPU: NVIDIA GeForce RTX 4090 (24GB)
- PyTorch: 2.4.1+cu124

## Key Dependencies
- transformers==4.45.1
- torch with CUDA 12.4 support
- flash-attn for efficient attention
- decord for video processing
- einops for tensor operations
- bitsandbytes for 8-bit quantization
- accelerate for optimized model loading
- pyyaml for configuration management

## Important Files
- `core/model_manager.py`: Main model management class
- `main.py`: CLI interface for running inference
- `config.yaml`: Configuration settings
- `requirements.txt`: Python dependencies

## Testing
To verify the setup:
1. Check GPU: `nvidia-smi`
2. Test system functionality: `python test_system.py`
3. Run enhanced video analysis: `python main.py --video data/street_scene.mp4 --prompt "Analyze the complete story" --num-frames 120`

## Performance Metrics
- **Model Loading Time**: 16 seconds with 8-bit quantization
- **Video Loading Time**: ~19.51 seconds (596-second video)
- **AI Inference Time**: ~12.80 seconds (160 frames)
- **Total Processing Time**: ~32.50 seconds
- **Processing Rate**: 4.92 frames/second
- **Memory Utilization**: 88% of RTX 4090 (20.86GB/24GB)

## Enhanced Capabilities
- **Character Development Analysis**: Detailed AI analysis of character traits and development
- **Story Structure Analysis**: Comprehensive plot progression and narrative understanding
- **Extended Video Coverage**: 2-5 minute video segments with full story comprehension
- **Professional Applications**: Suitable for surveillance, content analysis, and media understanding
- **Internal Documentation**: All AI responses and processing metrics automatically logged

## Notes for Future Development
- Peak frame capacity discovered at 160 frames with detailed narrative output
- Memory boundary mapped at 200 frames (CUDA OOM limit)
- Singleton architecture ensures single model instance across all operations
- Configuration auto-selects optimal frame counts based on quantization level
- Ready for batch processing and sequential video analysis workflows