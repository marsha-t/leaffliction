# Leaffliction [Work in Progress]

An image-processing and machine-learning pipeline for plant-disease classification from leaf images, built in Python and PyTorch.

The project follows the [Leaffliction project subject](leaffliction.pdf),
which structures the work into four parts: dataset analysis, augmentation,
image transformation, and classification.

### 1. Dataset analysis

Inspect class distributions and visualise image counts to identify imbalanced datasets.

### 2. Augmentation

Apply geometric transformations and distortions, and generate additional images to balance underrepresented classes.

### 3. Image transformation

Segment leaves and visualise contours, landmarks, shape measurements, and colour distributions.

### 4. Classification

Train and evaluate image classifiers in PyTorch and run inference from saved checkpoints. The training pipeline splits data before augmentation, records splits in CSV manifests, computes normalisation statistics from training data only, and keeps experiment outputs separate.

The current implementation includes a custom CNN. Planned experiments include pretrained CNNs and vision transformers.

See [DESIGN.md](DESIGN.md) for implementation decisions and tradeoffs.

## Project structure

The root-level scripts provide the command-line entry points required by the
project subject, while the implementation is organised into modules by stage:

- `analysis/` — dataset distribution analysis and visualisation
- `augmentation/` — image augmentation and dataset balancing
- `transformation/` — leaf segmentation and feature visualisation
- `classification/` — data loading, model definitions, training utilities, and preprocessing
- `manifests/` — CSV files recording dataset splits
- `demo_images/` — example images for running inference

Generated datasets and model checkpoints are kept in
`augmented_directory/` and `checkpoints/` respectively and are not tracked by Git.

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

The standard setup is sufficient to run the project on CPU. Training and prediction automatically use CUDA when it is available, and otherwise use the CPU.

The tested Windows configuration uses PyTorch 2.6.0, torchvision 0.21.0, and CUDA 12.6. After installing the dependencies above, replace the PyTorch packages with their CUDA 12.6 builds:

```bash
python -m pip install --force-reinstall --no-deps torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
```

This procedure was tested on Windows; other platforms may require different installation steps.

Verify that PyTorch detects your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

A result of `True` indicates that CUDA is available to PyTorch.


## Usage

### Analyse a dataset

Visualise the class distribution for a plant dataset:

```bash
python Distribution.py data/Apple
```

### Preview augmentations

Apply and visualise the available augmentation techniques on a single image:

```bash
python Augmentation.py "data/Apple/Apple_Black_rot/image (1).JPG"
```

### Balance a dataset

Generate augmented images for underrepresented classes:

```bash
python Augmentation.py data/Apple
```

The balanced dataset is written to `augmented_directory/`.


### Apply Transformations 

Apply and visualise the image-transformation pipeline on a single image:

```bash
python Transformation.py 'data/Apple/Apple_Black_rot/image (1).JPG'
```

### Train a classifier

Train the custom CNN and save output in `checkpoints/custom_cnn/baseline/`:

```bash
python train.py data/Apple --run-name baseline
```

### Predict an image label

Run inference using a checkpoint produced by `train.py`:

```bash
python predict.py demo_images/predict_apple_black_rot.JPG --checkpoint checkpoints/custom_cnn/baseline/best.pt
```

### Restore the dateset

Remove generated augmentation outputs and restore the dataset to its original state:

```bash
python ./Restore.py data/Apple
```
