# Documentation for Design Decisions

## Project Architecture
The project is organised into the following packages:

```
analysis/         Dataset analysis and visualisation
augmentation/     Data augmentation operations
transformation/   Classical image processing and feature extraction
classification/   Model training and prediction
```

The top-level scripts (e.g., `Distribution.py`) are the command-line entry points required by the subject. 

This separation keeps the executable scripts focused on parsing arguments and orchestrating the workflow, and encapsulates implementation details within reusable packages. 


## Image Processing Pipeline
Parts 2 and 3 focus on image manipulation and analysis, in line with classical computer vision. Accordingly, we chose to work with NumPy arrays using Pillow and OpenCV (instead of deep learning frameworks like PyTorch and TensorFlow). As the project requires the processing and saving of one image at a time, introducing tensors at this stage would add unnecessary conversions without meaningful benefits. 

In general, we aimed to balance educational value with engineering practicality. Where feasible, we implement the core algorithms ourselves to understand the underlying mathematics, while relying on established libraries for low-level operations that are well-tested, computationally intensive, or orthogonal to the project's learning objectives.


## Augmentation Design

The augmentation module supports both individual image augmentation and
dataset balancing. For dataset balancing, augmentations are generated until
each class reaches the size of the current majority class.

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

The pipeline is designed to be safely rerunnable. If augmentation is
interrupted or executed multiple times, existing augmentations are detected
automatically and count towards the balancing target.

This avoids unnecessary computation and prevents exponential dataset growth
from repeatedly augmenting previously generated images.

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
Reusing the manifest assumes that the original images and dataset structure remain unchanged. Restore.py removes the manifest and detected augmentation outputs, allowing the next training run to create a new manifest.

## Custom CNN

### Preprocessing

Images are converted to RGB, resized to 224 × 224 pixels, converted to
tensors, and normalised using a mean and standard deviation for each colour
channel. Normalisation statistics are computed from the training samples, including
training augmentations, after resizing and conversion to tensors.

Training, validation, and prediction use the same resizing and
normalisation pipeline. Prediction restores the normalisation values
saved in the checkpoint.

Class names are sorted before assigning integer labels. The mapping is
shared by the training and validation datasets and stored in the
checkpoint.

### Architecture

The feature extractor contains four convolutional blocks with 32, 64, 128,
and 256 output channels. Each block contains two 3 × 3 convolutions, each
followed by batch normalisation and ReLU activation. A 2 × 2 max-pooling
layer reduces the spatial dimensions at the end of each block.

Global average pooling reduces the final feature maps to a 256-element
vector per image. A linear layer converts this vector into one
unnormalised score, or logit, per class.

### Training and Model Selection

The custom CNN is trained using cross-entropy loss and the AdamW optimiser with a weight decay of 0.0001. The command line controls the learning rate, batch size, and number of epochs.

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

Checkpoints include model weights, optimiser state, epoch number, validation metrics, model type, class mapping, and normalisation
statistics. Prediction uses the saved model type, weights, class mapping, and normalisation statistics to reconstruct the classifier.

Although optimiser state is saved, resuming training from a checkpoint is not currently exposed through the command line. Training history is
written to JSON after the training loop completes, so an interrupted run may have checkpoints without a history file.

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