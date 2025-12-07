<div align="center">

# QUAN: Quaternion Approximation Networks

### Rotation-Equivariant Object Detection via Separable Quaternion Neural Networks

[![Paper](https://img.shields.io/badge/IROS%202025-Paper-blue)](https://arxiv.org/abs/2509.05512)
[![Website](https://img.shields.io/badge/Website-QUANpaper-purple)](https://cwru-aism.github.io/QUANpaper/)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/bryceag11/QUAN_ultralytics.git)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

<img src="https://github.com/user-attachments/assets/8b4b083b-d657-43cf-8cf5-a2d6cdb8af46" width="600" alt="Poincaré Quaternion Mapping">

*RGB-to-Quaternion Poincaré mapping for rotation-equivariant feature extraction*

</div>

---

## Overview

**QUAN** (Quaternion Approximation Networks) introduces a novel approach to rotation-equivariant deep learning by leveraging quaternion algebra within neural network architectures. Built on top of [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), QUAN provides:

- **Quaternion Convolutions (QConv2D)**: Separable quaternion convolution that preserves rotational relationships
- **Independent Quaternion Batch Normalization (IQBN)**: Normalizes each quaternion component independently
- **Poincaré RGB-to-Quaternion Mapping**: Maps RGB images to quaternion space using the Poincaré ball model
- **Custom CUDA Kernels**: Optimized GPU kernels for efficient quaternion operations
- **Quaternion Angular Loss**: Geodesic distance on SO(3) for orientation-aware training

### Key Results

| Task | Dataset | Model | Performance |
|------|---------|-------|-------------|
| OBB Detection | DOTA-v1.0 | QUAN-YOLO11n | 78.X mAP50 |
| OBB Detection | DOTA-v1.0 | QUAN-YOLO11s | 79.X mAP50 |
| Classification | CIFAR-10 | Q-WRN-16-4 | 95.X% |
| Classification | ImageNet | Q-ResNet-34 | 76.X% Top-1 |

---

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/bryceag11/QUAN_ultralytics.git
cd QUAN_ultralytics

# Run the setup script (creates conda env + builds CUDA kernels)
./setup.sh

# Activate the environment
conda activate quan
```

### Manual Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate quan

# Install package in editable mode
pip install -e .

# Build CUDA kernels (optional, requires CUDA toolkit)
cd ultralytics/nn/cuda
python setup.py build_ext --inplace
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- CUDA >= 11.8 (for GPU acceleration)
- NVIDIA GPU with compute capability >= 7.0

---

## Project Structure

```
QUAN_ultralytics/
├── ultralytics/
│   ├── nn/
│   │   ├── modules/
│   │   │   ├── conv.py          # QConv2D, IQBN, QUpsample
│   │   │   └── head.py          # QER (Quaternion Extraction)
│   │   └── cuda/
│   │       ├── quaternion_ops.cu           # CUDA kernels
│   │       ├── quaternion_ops_optimized.cu # Optimized kernels
│   │       └── setup.py                    # CUDA build script
│   ├── cfg/models/11/
│   │   └── yolo11-obb-quan.yaml # QUAN model configs
│   └── utils/
│       └── loss.py              # Quaternion angular loss
├── classification/              # Classification experiments
│   ├── classification.py        # Training script
│   ├── models/                  # Q-ResNet, Q-WRN architectures
│   └── utils/                   # Data loading, experiment management
├── environment.yml              # Conda environment
├── setup.sh                     # Setup script
└── README.md
```

---

## Usage

### Object Detection (OBB)

#### Training on DOTA

```bash
# Train QUAN-YOLO11n on DOTA-v1.0
yolo train model=ultralytics/cfg/models/11/yolo11n-obb-quan.yaml \
           data=DOTAv1.yaml \
           epochs=300 \
           imgsz=1024 \
           batch=16 \
           device=0

# Train QUAN-YOLO11s (larger model)
yolo train model=ultralytics/cfg/models/11/yolo11s-obb-quan.yaml \
           data=DOTAv1.yaml \
           epochs=300 \
           imgsz=1024 \
           batch=8 \
           device=0
```

#### Inference

```python
from ultralytics import YOLO

# Load trained QUAN model
model = YOLO("runs/obb/train/weights/best.pt")

# Run inference on aerial images
results = model.predict("path/to/aerial/images", imgsz=1024)

# Visualize results
for result in results:
    result.show()
```

#### Validation

```bash
yolo val model=runs/obb/train/weights/best.pt data=DOTAv1.yaml
```

### Classification (CIFAR / ImageNet)

```bash
cd classification

# Train Q-WRN-16-4 on CIFAR-10
python classification.py \
    --model qwrn16_4 \
    --dataset cifar10 \
    --mapping poincare \
    --epochs 200 \
    --bs 128 \
    --lr 0.1

# Train Q-ResNet-34 on ImageNet
python classification.py \
    --model qrn34_imagenet \
    --dataset imagenet \
    --mapping poincare \
    --epochs 90 \
    --bs 256 \
    --data_root /path/to/imagenet

# Resume from checkpoint
python classification.py \
    --model qwrn16_4 \
    --dataset cifar10 \
    --exp_dir experiments/qwrn16_4_cifar10_XXX \
    --resume
```

#### Available Models

| Model | Description | Parameters |
|-------|-------------|------------|
| `qwrn16_2` | Q-Wide ResNet 16-2 | ~0.7M |
| `qwrn16_4` | Q-Wide ResNet 16-4 | ~2.8M |
| `qrn18` | Q-ResNet-18 | ~2.8M |
| `qrn34` | Q-ResNet-34 | ~5.3M |
| `qrn34_imagenet` | Q-ResNet-34 (ImageNet) | ~5.3M |
| `qwrn50_2` | Q-Wide ResNet 50-2 | ~17M |

#### Mapping Strategies

- `poincare` (default): Poincaré ball model mapping - best for rotation equivariance
- `hamilton`: Standard Hamilton quaternion construction
- `raw_normalized`: Direct normalized RGB mapping
- `mean_brightness`: Brightness-based quaternion mapping

---

## Key Components

### Quaternion Convolution (QConv2D)

Implements separable quaternion convolution using the approximation:

```
y = W ⊗ x ≈ M · (W_sep * x)
```

Where `M` is the mixing matrix and `W_sep` are separable real-valued weights.

### Poincaré Mapping

Maps RGB images to quaternion space using the Poincaré ball model:

```python
# RGB [0,1] → Quaternion [q_r, q_i, q_j, q_k]
r = sqrt(R² + G² + B²)
q_r = (1 - r²) / (1 + r²)      # Real component
q_ijk = 2 * [R, G, B] / (1 + r²)  # Imaginary components
```

### Quaternion Angular Loss

Geodesic distance on SO(3) for orientation-aware OBB training:

```python
L_angular = 2 * arccos(|q_pred · q_target|)
```

---

## CUDA Kernels

QUAN includes optimized CUDA kernels for:

- Quaternion convolution forward/backward passes
- Independent quaternion batch normalization
- Fused QConv + BN + SiLU for inference

To verify CUDA kernels are working:

```python
import torch
from ultralytics.nn.modules.conv import QConv2D

# Create quaternion conv layer
qconv = QConv2D(64, 128, kernel_size=3).cuda()

# Test with quaternion tensor [B, C, H, W, 4]
x = torch.randn(1, 16, 32, 32, 4).cuda()
y = qconv(x)
print(f"Output shape: {y.shape}")  # [1, 32, 32, 32, 4]
```

---

## Citation

If you use QUAN in your research, please cite:

```bibtex
@inproceedings{quan2025,
  title={QUAN: Rotation-Equivariant Object Detection via Separable Quaternion Neural Networks},
  author={Your Name},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025}
}
```

---

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO framework
- [Quaternion Neural Networks](https://arxiv.org/abs/1903.08478) for foundational quaternion deep learning

---

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

For commercial use, please contact for enterprise licensing options.
