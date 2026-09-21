# leaffliction


## Setup

This project uses **Python 3.10**. Create and activate a virtual environment:

On macOS or Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

For development, install the runtime dependencies and Flake8 together:

```bash
python -m pip install -r requirements-dev.txt
```

### GPU acceleration

Training and prediction automatically use CUDA when it is available,
and otherwise use the CPU.

The tested Windows configuration uses **PyTorch 2.6.0**,
**torchvision 0.21.0**, and **CUDA 12.6**.

After installing the dependencies above, replace the PyTorch packages
with their CUDA 12.6 builds:

```bash
python -m pip install --force-reinstall --no-deps torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
```

This command reinstalls only PyTorch and torchvision, preserving the
dependencies installed in the previous step. This procedure was tested
on Windows; other platforms may require different installation steps.

Verify that PyTorch detects your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

A result of `True` indicates that CUDA is available to PyTorch.


## Usage

Run the dataset analysis on a plant directory.

```bash
./Distribution.py data/Apple
./Distribution.py ./data/Apple
```

Apply augmentations on an image

```bash
./Augmentation.py 'data/Apple/Apple_Black_rot/image (1).JPG'
```

Create a balanced dataset with augmentations (without train/validation split)

```bash
python ./Augmentation.py data/Apple/
```

Apply transformations on an image

```bash
./Transformation.py 'data/Apple/Apple_Black_rot/image (1).JPG'
```

Restore dataset: Remove augmentations and manifest file

```bash
python ./Restore.py data/Apple
```

Train model

```bash
python train.py data/Apple  --run-name baseline
```

Predict image

```bash
python predict.py demo_images/predict_apple_black_rot.JPG --checkpoint checkpoints/custom_cnn/baseline/best.pt
python predict.py demo_images/predict_apple_rust.JPG --checkpoint checkpoints/custom_cnn/baseline/best.pt
python predict.py demo_images/predict_apple_scab.JPG --checkpoint checkpoints/custom_cnn/baseline/best.pt
```
