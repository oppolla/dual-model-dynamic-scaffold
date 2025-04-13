## SOVL System Testing Guide - Windows 11 (RTX 3070)

# This project is still under development 

### 1. Prerequisites

- Install Python 3.8+ from python.org

- Install CUDA 11.8 for RTX 30-series NVIDIA CUDA Toolkit 

- Install PyTorch with CUDA support `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cux.x`

- FastAPI (for server backend) `pip install fastapi`

- Transformers, Peft, BitsandBytes `pip install transformers peft bitsandbytes`

### 2. File Setup

All files must be in the same directory.

Initial File Personality Seed = sample_log.jsonl

### 3. Verification

Pre Run
- For environment test run test.bat
  
First Run
- Execute with dry-run mode `python sovl_system.py --dry-run`

Expected output:
```
Initializing SOVL System...
Using device: cuda
Base model 'gpt2-medium' loaded...
Dry run activated (max_samples=2, max_length=128)
System Ready.
```

### 5. Basic Test Commands
Try these in order:
Wake up sequence:
> wake
> (Should generate initial response)
> Simple generation
> Hello, how are you?
> (Should generate response)


Training test:
> train
(Should show dry-run training stats)

## 6. GPU Memory Management
For RTX 3070 (8GB VRAM):

Start with these config settings:
```
"training_config": {
  "batch_size": 1,
  "max_seq_length": 256,
  "quantization": "int8"
}
```
Monitor VRAM usage:

bash
Copy
nvidia-smi -l 1
Troubleshooting
Common RTX 30-series issues:

CUDA Out of Memory:

Reduce max_seq_length to 128

Add "quantization": "int8" to config

Driver Issues:
Update to latest NVIDIA Studio Driver:
https://www.nvidia.com/Studio/drivers

Performance Tuning:
Add to config for better RTX 30-series performance:

json
Copy
"controls_config": {
  "enable_dynamic_cross_attention": true,
  "quantization_mode": "int8"
}
Advanced Test Script
Create test_script.py:

python
Copy
from sovl_system import SOVLSystem

def stress_test():
    sovl = SOVLSystem()
    print("-- Cold Start Test --")
    print(sovl.generate("Hello world", max_new_tokens=20))
    
    print("\n-- Training Stress Test --")
    sovl.run_training_cycle(TRAIN_DATA, VALID_DATA, epochs=1)
    
    print("\n-- Memory Test --")
    for _ in range(5):
        sovl.generate("Test" * 50, max_new_tokens=100)
    
    print("\n-- Error Recovery Test --")
    print(sovl.generate("", max_new_tokens=500))  # Force OOM

if __name__ == "__main__":
    stress_test()
Key Windows-Specific Notes

Use WSL2 for better performance:

bash
Copy
wsl --install -d Ubuntu-22.04
If seeing DLL errors:

Install latest VC++ Redist: https://aka.ms/vs/17/release/vc_redist.x64.exe

Update DirectX Runtime: https://www.microsoft.com/en-us/download/details.aspx?id=35

For mixed precision issues:

bash
Copy
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
Let me know if you need help with:

Specific error messages

Performance optimization for your 3070

Custom training data preparation

Quantization configuration
