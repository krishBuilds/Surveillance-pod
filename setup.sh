#!/bin/bash
# InternVideo2.5 Setup Script for Surveillance System
# Sets up complete environment including Claude Code CLI

set -e  # Exit on error

echo "🚀 Starting InternVideo2.5 Surveillance System Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on CUDA-capable system
print_status "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    print_status "NVIDIA GPU detected"
else
    print_error "NVIDIA GPU not detected. This system requires CUDA support."
    exit 1
fi

# Update system packages
print_status "Updating system packages..."
apt update && apt upgrade -y

# Install system dependencies
print_status "Installing system dependencies..."
apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    python3-dev \
    python3-pip \
    python3-venv

# Install Node.js 20 and Claude Code CLI
print_status "Installing Node.js 20..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

print_status "Installing Claude Code CLI..."
npm install -g @anthropic-ai/claude-code

# Verify Node.js and Claude Code installation
print_status "Verifying installations..."
node --version
npm --version
claude-code --version

# Install Python dependencies
print_status "Installing Python dependencies..."
pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first
print_status "Installing PyTorch with CUDA 12.4 support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install flash-attention (requires specific setup)
print_status "Installing flash-attention (this may take a while)..."
pip3 install flash-attn --no-build-isolation

# Install remaining requirements
print_status "Installing remaining Python requirements..."
pip3 install -r requirements.txt

# Create necessary directories
print_status "Creating project directories..."
mkdir -p data outputs models logs

# Verify CUDA and PyTorch installation
print_status "Verifying CUDA and PyTorch installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Check if model needs to be downloaded
if [ ! -d "./models/InternVideo2_5" ]; then
    print_warning "InternVideo2.5 model not found in ./models/InternVideo2_5"
    print_status "You can download it using: python scripts/download_model.py --model-dir ./models/InternVideo2_5"
else
    print_status "InternVideo2.5 model found"
fi

# Create a simple test script
print_status "Creating test script..."
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify setup"""
import torch
import transformers
import cv2
import av
import decord
from PIL import Image
import numpy as np

print("✅ All imports successful!")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

chmod +x test_setup.py

# Run test
print_status "Running setup verification test..."
python3 test_setup.py

# Print completion message
echo ""
print_status "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Download the InternVideo2.5 model (if not already done):"
echo "   python scripts/download_model.py --model-dir ./models/InternVideo2_5"
echo ""
echo "2. Test video analysis:"
echo "   python main.py --video <video_path> --prompt 'Describe what you see'"
echo ""
echo "3. Start Claude Code in this directory:"
echo "   claude-code"
echo ""
print_status "Environment ready for InternVideo2.5 surveillance analysis!"