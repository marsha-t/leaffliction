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
- the same image-augmentation pair cannot be generated twice;
- previously augmented images cannot become sources for further augmentation.

The augmentation capacity of each class is checked before execution so that
the program fails early if balancing cannot be achieved using the available
unique augmentations.

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
