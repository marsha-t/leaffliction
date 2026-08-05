from pathlib import Path
from collections import defaultdict

IMAGE_EXTENSIONS = ('.jpg', '.png')


def analyse_directory(path: str) -> dict[str, int]:
	"""
	Count image files in each sub-directory 

	Args:
		path (str): directory path
	
	Returns:
		dict: dictionary of {sub-directory: image count}
	"""
	dataset_dir = Path(path)

	distribution = defaultdict(int)
	for sub_dir in dataset_dir.iterdir():
		for file in sub_dir.iterdir():
			if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
				distribution[sub_dir.name] += 1
	return distribution
