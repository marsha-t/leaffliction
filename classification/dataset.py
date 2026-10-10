from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class LeafDataset(Dataset):
    """
    Load leaf images and numeric labels from paths grouped by class

    Args:
        samples_by_class (dict): Class names mapped to image paths
        class_to_index (dict): Class names mapped to numeric labels
        transform (callable | None): Transformation applied to each image
    """

    def __init__(
        self,
        samples_by_class,
        class_to_index,
        transform=None
    ):
        """Build sorted list of image paths and labels"""
        self.class_to_index = dict(class_to_index)
        self.transform = transform
        self.samples = []

        for class_name in sorted(samples_by_class):
            if class_name not in self.class_to_index:
                raise ValueError(f'No numeric label defined for {class_name}')

            label = self.class_to_index[class_name]

            for image_path in sorted(samples_by_class[class_name]):
                self.samples.append((Path(image_path), label))

    def __len__(self):
        """Return the number of image samples"""
        return len(self.samples)

    def __getitem__(self, index):
        """
        Load an RGB image, apply optional transform, and return its label

        Args:
            index (int): Position of sample in the dataset

        Returns:
            tuple: Image or transformed image, followed by its integer class
                label
        """
        image_path, label = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label
