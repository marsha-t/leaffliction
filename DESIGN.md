# Documentation for Design Decisions

## Project Architecture
The project is organised into the following packages:

```
analysis/         Dataset analysis and visualisation
augmentation/     Data augmentation operations
transformation/   Classical image processing and feature extraction
classification/   Model training and prediction
```

Some structural choices follow the project's submission and evaluation
requirements. Top-level scripts such as `Distribution.py` provide the required
command-line entry points, while `augmented_directory/` provides a complete
balanced dataset for demonstrating augmentation. Classification uses its own
training samples and manifests rather than reading this balanced copy.

This separation keeps the executable scripts focused on parsing arguments and orchestrating the workflow, and encapsulates implementation details within reusable packages. 


## Image Processing Pipeline
Parts 2 and 3 focus on image manipulation and analysis, in line with classical computer vision. Accordingly, we chose to work with NumPy arrays using Pillow and OpenCV (instead of deep learning frameworks like PyTorch and TensorFlow). As the project requires the processing and saving of one image at a time, introducing tensors at this stage would add unnecessary conversions without meaningful benefits. 

In general, we aimed to balance educational value with engineering practicality. Where feasible, we implement the core algorithms ourselves to understand the underlying mathematics, while relying on established libraries for low-level operations that are well-tested, computationally intensive, or orthogonal to the project's learning objectives.


## Augmentation Design

The augmentation module supports both individual image augmentation and
dataset balancing. For dataset balancing, augmentations are generated until
each class reaches the size of the current majority class.

### Augmentation outputs

The augmentation module is shared by the standalone augmentation command (Part 2) and the classification pipeline (Part 4). The two workflows use the same augmentation functions but have different output requirements.

| Workflow | Source dataset | `augmented_directory` |
| --- | --- | --- |
| Single-image augmentation | Save all six augmentations alongside the original | Not modified |
| Dataset balancing | Save generated augmentations alongside the originals | Create a complete balanced copy of the dataset |
| Classification | Save planned training augmentations alongside the originals | Not modified |

The image-saving functions accept an optional `augmented_root` argument. When omitted, augmented images are saved only alongside their originals. When provided, an additional copy is saved under the specified destination, preserving the dataset's directory structure.

This keeps the classification pipeline independent of `augmented_directory`, which is reserved for demonstrating dataset balancing during evaluation.

Augmentation execution skips an image-augmentation pair when all required output files already exist. This allows interrupted executions to resume without unnecessarily regenerating completed outputs.

### Augmentation scheduling

Only original images are used as augmentation sources; generated images are
never augmented further.

Images are processed in randomly shuffled rounds. Each original image is
considered once before images are reused, while the augmentation applied to
each image is randomly selected from those not previously used.

This combines randomness with several constraints:

- source images are distributed across the dataset rather than repeatedly
  sampling the same images, giving the generated dataset greater diversity
  of source images;
- during dataset balancing, previously generated image-augmentation pairs
  are excluded from the plan, preventing duplicate samples;
- previously augmented images cannot become sources for further augmentation.

The augmentation capacity of each class is checked before execution so that
the program fails early if balancing cannot be achieved using the available
unique augmentations.

The single-image command applies every supported augmentation and can
overwrite previously generated outputs for that image. Duplicate prevention
applies to the dataset-balancing workflow.

### Rerunnable pipeline

The pipeline is designed to be safely rerunnable. During dataset balancing, existing augmentations are detected automatically and count towards the balancing target, preventing unnecessary augmentation and uncontrolled dataset growth.

During execution, planned augmentations are skipped when all required output files already exist. This allows interrupted operations to resume without regenerating completed outputs.

### Alternatives considered

| Approach | Reason not used |
| --- | --- |
| Apply every augmentation to every image | Preserves the original class imbalance and generates unnecessary images. |
| Pure random image/augmentation sampling | May repeatedly select some images while ignoring others, can generate duplicate image-augmentation pairs, and results in poor utilisation of the source dataset. |
| Augment augmented images | Repeated transformations may progressively degrade image quality, cause the generated distribution to drift further from the original dataset, and lead to uncontrolled dataset growth. |
| Ignore existing augmentations on rerun | Repeats unnecessary computation, may regenerate existing samples, and prevents interrupted augmentation from being resumed cleanly. |

The final approach therefore uses targeted dataset balancing, shuffled
round-robin image selection, augmentation-history tracking, and rerunnable
execution.

## Dataset Splitting and Validation

Training preparation splits original images into training and validation
sets before planning additional augmentations. Only training originals and their augmentations enter the training set.
Validation contains original images only. This prevents an original image and its augmented versions from appearing on opposite sides of the split, which could make validation results overly optimistic.

Dataset balancing is then applied to the training set. The standalone augmentation command balances a dataset without creating a split.

## Dataset Manifests

A CSV manifest records the samples assigned to each split:

| Field | Purpose |
| --- | --- |
| `path` | Image path relative to the dataset directory |
| `class_name` | Class assigned to the image |
| `split` | Either `train` or `validation` |

Relative paths allow a dataset to move without storing machine-specific
absolute paths.

The manifest fixes the training and validation file assignments and the augmentation plan, ensuring that experiments use the same samples.
It records both original images and the specific augmentations selected for training.
The manifest is saved before augmentation execution. Subsequent runs reuse its assignments and reconstruct the same augmentation plan rather than selecting new augmentations or creating a new split. Missing augmentation
outputs are regenerated according to this plan.

Reusing the manifest assumes that the original images and dataset structure remain unchanged. 

### Dataset restoration and evaluation preparation

`Restore.py` supports two cleanup modes to accommodate development and evaluation workflows.

| Component | Full restoration | Evaluation preparation (`--evaluation`) |
| --- | --- | --- |
| Original images | Preserve | Preserve |
| Augmented images in source dataset | Remove | Remove |
| Balanced dataset in `augmented_directory` | Remove | Preserve |
| Classification manifest | Remove | Preserve |
| Generated training and validation symlinks | Remove | Remove |

Full restoration returns the source dataset to its original state and removes generated artifacts, including the classification manifest. A subsequent training run therefore creates a new split and augmentation plan.

Evaluation preparation removes generated images from the source dataset and the symbolic links used to inspect training and validation assignments, while preserving the balanced dataset and classification manifest.

The preserved manifest allows the classification pipeline to reconstruct the same augmentation plan and regenerate missing training images when training is executed again. This enables evaluation of the classification workflow without requiring `augmented_directory` to be regenerated or modified.

## Classification Overview

Four model architectures share the dataset preparation, training, and prediction
pipeline:

| Model | Feature extractor | Training approach |
| --- | --- | --- |
| Custom CNN | Four convolutional blocks | Train the entire model |
| Pretrained CNN | ImageNet ResNet18 | Train a replacement classification head |
| Custom ViT | Six custom transformer encoder blocks | Train the entire model |
| Pretrained ViT | ImageNet ViT-B/16 | Train a replacement classification head |

Apple and Grape are classified separately, with four classes per dataset.
Class names are sorted to produce a deterministic label mapping, shared across
training and validation and saved in checkpoints. All models receive the number
of classes through the model factory; their classification heads produce one
unnormalised score, or logit, per class.

## Preprocessing

Custom CNN and custom ViT use RGB images resized directly to 224 × 224 pixels.
Per-channel normalisation statistics are computed from training samples,
including training augmentations, after resizing and conversion to tensors.
Validation images do not contribute to these statistics.

Pretrained ResNet18 and ViT-B/16 use the transforms associated with their
torchvision weights, including resizing, centre cropping, and normalisation.
The weight identifier and normalisation values are recorded in checkpoint
metadata.

Each model uses the same deterministic preprocessing during training,
validation, and prediction. Augmentations are generated during dataset
preparation rather than sampled dynamically by the preprocessing transforms.
Prediction reconstructs preprocessing from the configuration saved in the
checkpoint.

## Custom CNN

The feature extractor contains four convolutional blocks with 32, 64, 128,
and 256 output channels. Each block contains two 3 × 3 convolutions, each
followed by batch normalisation and ReLU activation. A 2 × 2 max-pooling
layer reduces the spatial dimensions at the end of each block.

Global average pooling reduces the final feature maps to a 256-element
vector per image. A linear layer converts this vector into one
unnormalised score, or logit, per class.

## Custom Vision Transformer

The custom ViT divides each 224 × 224 image into 196 non-overlapping 16 × 16
patches. Each flattened RGB patch is projected into a 512-dimensional embedding.
A learned CLS token and learned positional embeddings produce a sequence of
197 tokens.

Six encoder blocks process the sequence. Each block uses eight attention heads,
LayerNorm before attention and the MLP, residual connections, and an MLP with a
hidden width four times the embedding dimension. Attention and MLP dropout use
a probability of 0.1.

After the final encoder block, the CLS token is passed to a linear classification
head. The implementation does not apply an additional LayerNorm between the
final encoder block and the classifier.

## Pretrained Models

Both pretrained models use ImageNet weights and frozen feature extractors.
The original classification head is replaced after freezing, leaving the new
head trainable. Its input width is obtained from the original head, and its
output width comes from the dataset's class count.

During ResNet training, backbone BatchNorm layers remain in evaluation mode
so their running statistics do not change. ViT uses LayerNorm and does not
require this BatchNorm handling.

Prediction constructs the model with `pretrained=False` before loading
checkpoint weights. This avoids downloading ImageNet weights that would
immediately be replaced. The flag controls initial weight loading; it does
not unfreeze the backbone.

## Training and Model Selection

All four models use cross-entropy loss and AdamW with weight decay 0.0001.
A `ReduceLROnPlateau` scheduler monitors validation loss after each epoch,
with factor 0.5, patience 2, and minimum learning rate 0.000001. The same
training configuration applies to all four architectures. The command line
controls the learning rate, batch size, and number of epochs.

Training samples are shuffled each epoch. Validation samples are evaluated without shuffling, with the model in evaluation mode and gradient
calculation disabled.

Loss and accuracy are recorded for both splits after each epoch. Epoch loss is averaged across samples, accounting for a potentially smaller
final batch.

Each run produces two checkpoints and a training history file:

| File | Purpose |
| --- | --- |
| `best.pt` | Model state from the epoch with the lowest validation loss |
| `last.pt` | Model state from the most recently completed epoch |
| `history.json` | Training and validation loss and accuracy for each epoch |

Validation loss determines which checkpoint is considered best. Accuracy is recorded as an additional metric but does not control
checkpoint selection.

Checkpoints include model weights, optimiser state, epoch number, validation
metrics, model type, class mapping, and preprocessing configuration. Prediction
uses this metadata and the saved weights to reconstruct the classifier.

Although optimiser state is saved, scheduler state is not saved and the command
line does not currently support resuming training. Training history is written
to JSON after the training loop completes, so an interrupted run may have
checkpoints without a history file.

## Prediction and Explainability

Prediction restores the model type, class mapping, preprocessing, and model
weights from checkpoint metadata, then runs classification in evaluation mode.

Grad-CAM is supported for all four architectures. CNNs use their final
convolutional feature blocks. ViTs use normalised token features before
attention in the final encoder block; the CLS token is removed and patch
features are reshaped into a spatial grid.

Custom ViT additionally supports attention rollout, combining head-averaged
attention across encoder blocks while accounting for residual connections.
Rollout describes attention propagation and is not specific to the predicted
class. Pretrained ViT attention rollout is not implemented.

Prediction displays three panels, or four when custom ViT rollout is available.
All panels retain fixed 256 × 256 axis limits to make the 224 × 224
preprocessed image size visible.

## Training Run Organisation

Each training run writes to its own directory: `checkpoints/<model>/<run-name>/`

A descriptive name can be supplied with `--run-name`. When omitted, a timestamp based on local time is generated, including microseconds.

Existing run directories are rejected to prevent previous checkpoints and history from being overwritten. The directory is created before dataset preparation, so a failed run can leave an empty or partially populated
directory behind.

## Reproducibility

Dataset splitting and augmentation scheduling use a fixed seed of 42.
The elastic and grid distortion operations also receive fixed seeds.
Saved manifests preserve the sample assignments reused by later runs.

Training is not fully deterministic. Model initialisation and training
sample shuffling are not explicitly seeded, and numerical results can
also vary with hardware and execution settings.

A shared manifest therefore preserves the dataset split but does not
guarantee identical model weights or metrics across training runs.
