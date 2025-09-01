#!/usr/bin/env python3
"""
Download InternVideo2.5 Model from HuggingFace
"""

import os
import torch
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoModel, AutoTokenizer
import argparse

def download_internvideo25(model_dir="./models/InternVideo2_5"):
    """
    Download InternVideo2.5 model files from HuggingFace
    
    Args:
        model_dir: Directory to save the model
    """
    model_id = "OpenGVLab/InternVideo2_5_Chat_8B"
    
    print(f"Downloading InternVideo2.5 model to: {model_dir}")
    print("This is an 8.4B parameter model and requires ~16GB of storage")
    print("=" * 50)
    
    try:
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Download the model using snapshot_download
        print(f"Downloading from HuggingFace: {model_id}")
        print("This may take a while depending on your internet connection...")
        
        snapshot_download(
            repo_id=model_id,
            cache_dir=model_dir,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"\n✅ Model downloaded successfully to: {model_dir}")
        
        # Verify the download
        print("\nVerifying download...")
        required_files = [
            "config.json",
            "configuration_internvideo2_5.py",
            "modeling_internvideo2_5.py",
            "tokenization_internvideo2_5.py"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(model_dir, file)):
                missing_files.append(file)
        
        if missing_files:
            print(f"⚠️  Warning: Some files may be missing: {missing_files}")
        else:
            print("✅ All essential files present")
        
        return model_dir
        
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        return None

def test_model_loading(model_dir):
    """
    Test if the model can be loaded (DOWNLOAD VERIFICATION ONLY)
    For regular inference, use ModelManager singleton pattern with 8-bit quantization instead
    
    Args:
        model_dir: Directory containing the model
    """
    print("\n" + "=" * 50)
    print("Testing model loading...")
    
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory: {gpu_memory:.2f} GB")
            
            if gpu_memory < 12:
                print("⚠️  Warning: Model requires ~12GB VRAM with 8-bit quantization")
                print("   Full precision requires ~20GB VRAM")
                print("   Consider using 4-bit quantization for <12GB GPUs")
        
        # Try loading tokenizer first (lightweight)
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded successfully")
        
        # Try loading model
        print("\nLoading model (this may take a while)...")
        
        if device == "cuda" and gpu_memory >= 12:
            # Use 8-bit quantization for efficient loading (recommended)
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            model = AutoModel.from_pretrained(
                model_dir,
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map="auto"
            )
            print("✅ Model loaded successfully on GPU with 8-bit quantization")
        else:
            print("ℹ️  Skipping full model loading due to memory constraints")
            print("   To load the model efficiently:")
            print("   - Use ModelManager with config.yaml for optimal 8-bit quantization")
            print("   - This provides 3x frame capacity (160 vs 48 frames)")
            print("   - Memory usage: ~17GB vs ~20GB full precision")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download InternVideo2.5 model")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models/InternVideo2_5",
        help="Directory to save the model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test loading the model after download"
    )
    
    args = parser.parse_args()
    
    print("InternVideo2.5 Model Downloader")
    print("=" * 50)
    
    # Download the model
    model_dir = download_internvideo25(args.model_dir)
    
    if model_dir and args.test:
        # Test loading if requested
        test_model_loading(model_dir)
    
    print("\n" + "=" * 50)
    print("Download complete!")
    print(f"\nTo use the model efficiently:")
    print(f"  # Recommended: Use ModelManager with 8-bit quantization")
    print(f"  from core.model_manager import ModelManager")
    print(f"  manager = ModelManager(model_path='{args.model_dir}', config_path='config.yaml')")
    print(f"  manager.load_model()  # Auto-loads with 8-bit quantization")
    print(f"\n  # Direct usage (not recommended for production):")
    print(f"  from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig")
    print(f"  bnb_config = BitsAndBytesConfig(load_in_8bit=True)")
    print(f"  model = AutoModel.from_pretrained('{args.model_dir}', quantization_config=bnb_config, trust_remote_code=True)")

if __name__ == "__main__":
    main()