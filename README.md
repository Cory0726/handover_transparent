# handover_transparent
My project for handover transparent object.

## System setup
```bash
# python version
# python 3.12.12

# basler camera (ToF) API
pip install pypylon

# torch, torchvision
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
# xformers
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126
# OpenCV, tqdm, scikit-learn, scikit-image, addict
pip install opencv-python, tqdm, scikit-learn, scikit-image, addict

# depth-anything-3
# Step1 : git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
# Step2 : cd to the Depth-Anything-3 folder (Don't delete this folder)
pip install -e .

# triton
# Install `triton` package for Windows system
## Step 1 : Download the Windows `triton` package at the [**HuggingFace**](https://hf-mirror.com/madbuda/triton-windows-builds)
## Step 2 : Install the `triton` package
pip install triton-3.0.0-cp312-cp312-win_amd64.whl

# techmanpy
pip install techmanpy
```

## Script
### pipeline.py
#### Flow
1. `tof_cam/tof_data_grab.py`
2. `u_net/preditct.py`
3. `da3/predict.py`
4. `gr_convnet/preditct.py`


