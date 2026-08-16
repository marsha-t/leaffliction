from pathlib import Path
from augmentation.dataset import scan_dataset
from classification.splits import ensure_split


dataset_root = Path('data/Apple').resolve()
dataset = scan_dataset(dataset_root)
train, validation = ensure_split(
    dataset,
    dataset_root,
    manifest_path=Path('splits/Apple.csv'),
    validation_ratio=0.2,
    seed=42
)
