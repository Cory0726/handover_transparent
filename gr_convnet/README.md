# note_for_GR_ConvNet
My note for Generative Residual Convolutional Neural Network (GR-ConvNet).

- [Reference](#reference)
- [Introduction](#introduction)
- [System setup](#system-setup)
  - [Package installation](#package-installation)
  - [Issue](#issue)
- [Script](#script)

## Reference
- Paper : 
  - [Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network](https://arxiv.org/abs/1909.04810)
- Github : 
  - [robotic-grasping](https://github.com/skumra/robotic-grasping)

## Introduction
**Generative Residual Convolutional Neural Network (GR-ConvNet)**
- The model architecture which detects objects in the camera's field of view and **predicts a suitable antipodal grasp** configuration for the objects in the image.

## System setup
### Package installation
```bash
# torch, torchvision
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# other
pip install opencv-python, matplotlib, scikit-image, imageio, torchsummary, tensorboardX
```
### Issue
#### In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`
- 改`run_offline.py`裡的`torch.load`那一行
```python
# Origin :
net = torch.load(args.network)
# Change to :
net = torch.load(args.network, weights_only=False)
```

## Script
### predict.py