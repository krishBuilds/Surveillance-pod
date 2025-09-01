# InternVideo2.5 Project

Video understanding and analysis using InternVideo2.5 - an 8.4B parameter multimodal language model.

## Project Structure

```
.
├── core/                   # Core modules
│   ├── model_manager.py   # Model loading and management
│   └── init_internvideo.py # Original initialization script
├── scripts/                # Utility scripts
│   ├── download_model.py  # Download model from HuggingFace
│   └── test_internvideo.py # Test script
├── ai_analysis/            # AI analysis modules (to be added)
├── data/                   # Input data directory
├── outputs/                # Output results directory
├── models/                 # Model files
│   └── InternVideo2_5/    # Downloaded model
├── docs/                   # Documentation
├── main.py                 # Main application entry point
├── config.yaml            # Configuration file
└── requirements.txt       # Python dependencies
```

## Setup

### Prerequisites
- Python 3.11+
- CUDA 12.4 (for GPU support)
- RTX 4090 or similar GPU with 24GB VRAM

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the model (if not already done):
```bash
python scripts/download_model.py --model-dir ./models/InternVideo2_5
```

## Usage

### Process a video:
```bash
python main.py --video path/to/video.mp4 --prompt "Describe what's happening in this video"
```

### Process an image:
```bash
python main.py --image path/to/image.jpg --prompt "What objects do you see?"
```

### Options:
- `--model-path`: Path to model directory (default: ./models/InternVideo2_5)
- `--num-frames`: Number of frames to sample from video (default: 16)
- `--device`: Device to use (auto/cuda/cpu)
- `--precision`: Model precision (bf16/fp16/fp32)

## Model Information

- **Model**: InternVideo2.5 Chat 8B
- **Parameters**: 8.4 billion
- **Architecture**: Video MLLM built on InternVL2.5
- **Capabilities**: Long and rich context video understanding
- **Memory Requirements**: ~16GB VRAM for BF16 inference

## GPU Configuration

The project is optimized for NVIDIA RTX 4090:
- 24GB VRAM (sufficient for full model)
- CUDA 12.4 support
- BFloat16 precision for optimal performance

## AI Analysis Modules

The `ai_analysis/` directory is prepared for custom analysis logic to be added.