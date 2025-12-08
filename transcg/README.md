# note_for_TransCG
My notes for TransCG.

- [Refernece](#reference)
- [Introduction](#introduction)
  - [TransCG](#transcg)
  - [TransCG Dataset](#transcg-dataset)
- [System setup](#system-setup)
  - [Package installation](#package-installation)
- [Issue](#issue)
- [Script](#script)

# Reference
- Paper : [TransCG: A Large-Scale Real-World Dataset for Transparent Object Depth Completion and a Grasping Baseline](https://ieeexplore.ieee.org/document/9796631)
- Web :
  - [Github - TransCG](https://github.com/Galaxies99/TransCG)
  - [GraspNet - TransCG](https://graspnet.net/transcg)

# Introduction
## TransCG
Include the **dataset** and the proposed **Depth Filler Net (DFNet)** models.

## TransCG Dataset
- TransCG dataset is available on [**official page**](https://graspnet.net/transcg).
- TransCG dataset is the first large-scale real-world dataset for transparent object depth completion and grasping,
- In total, the dataset contains **57,715 RGB-D images** of **51 transparent objects**.
- The **3D mesh model** of the transparent objects are also provided in the dataset.

## System setup
### Package installation
```commandline
pip install -r requirements.txt
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Issue
### In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`
- 改`inferencer.py`裡的`torch.load`那一行
```python
# Origin :
checkpoint = torch.load(checkpoint_file, map_location=self.device)
# Change to :
checkpoint = torch.load(checkpoint_file, map_location=self.deivce, weights_only=False)
```

## Script
### predict.py
- Configuration : `./configs/inference.yaml`
- Pre-trained weight : `./checkpoints/checkpoint.tar`
