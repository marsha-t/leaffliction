# Documentation for Design Decisions

## Project Architecture
The project is organised into the following packages:

```
analysis/         Dataset analysis and visualisation
augmentation/     Data augmentation operations
transformation/   Classical image processing and feature extraction
classification/   Model training and prediction
common/           Shared utilities (image I/O, paths, configuration)
```

The top-level scripts (e.g., `Distribution.py`) are the command-line entry points required by the subject. 

This separation keeps the executable scripts focused on parsing arguments and orchestrating the workflow, and encapsulates implementation details within reusable packages. 


## Image Processing Pipeline
Parts 2 and 3 focus on image manipulation and analysis, in line with classical computer vision. Accordingly, we chose to work with NumPy arrays using Pillow and OpenCV (instead of deep learning frameworks like PyTorch and TensorFlow). As the project requires the processing and saving of one image at a time, introducing tensors at this stage would add unnecessary conversions without meaningful benefits. 
