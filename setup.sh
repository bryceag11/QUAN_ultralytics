#!/bin/bash
# QUAN Setup Script
# Sets up conda environment and builds CUDA kernels

set -e  # Exit on error

echo "=========================================="
echo "  QUAN: Quaternion Approximation Networks"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install Miniconda or Anaconda first.${NC}"
    exit 1
fi

# Check for CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
    echo -e "${GREEN}Found CUDA $CUDA_VERSION${NC}"
else
    echo -e "${YELLOW}Warning: nvcc not found. CUDA kernels will not be built.${NC}"
    echo "PyTorch fallback will be used for quaternion operations."
fi

# Create conda environment
echo ""
echo "Step 1: Creating conda environment 'quan'..."
if conda env list | grep -q "^quan "; then
    echo -e "${YELLOW}Environment 'quan' already exists. Updating...${NC}"
    conda env update -f environment.yml -n quan
else
    conda env create -f environment.yml
fi

echo ""
echo "Step 2: Activating environment and installing package..."
eval "$(conda shell.bash hook)"
conda activate quan

# Install the package in editable mode
pip install -e . --no-deps

# Build CUDA kernels if nvcc is available
if command -v nvcc &> /dev/null; then
    echo ""
    echo "Step 3: Building CUDA quaternion kernels..."
    cd ultralytics/nn/cuda

    if [ -f "setup.py" ]; then
        python setup.py build_ext --inplace
        echo -e "${GREEN}CUDA kernels built successfully!${NC}"
    else
        echo -e "${YELLOW}No CUDA setup.py found. Skipping kernel build.${NC}"
    fi

    cd "$SCRIPT_DIR"
else
    echo ""
    echo -e "${YELLOW}Skipping CUDA kernel build (no nvcc found)${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "To activate the environment:"
echo "  conda activate quan"
echo ""
echo "Quick start:"
echo "  # Train QUAN-YOLO11 on DOTA"
echo "  yolo train model=ultralytics/cfg/models/11/yolo11-obb-quan.yaml data=DOTAv1.yaml"
echo ""
echo "  # Run classification experiments"
echo "  python classification/classification.py --model qwrn16_4 --dataset cifar10"
echo "=========================================="
