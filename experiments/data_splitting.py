import os
import random
import shutil

source_dir = "dataset/images"

train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# Reproducible split
random.seed(42)

# Get class folders
classes = os.listdir(source_dir)

for class_name in classes:

    class_path = os.path.join(source_dir, class_name)

    # Ignore anything that isn't a directory
    if not os.path.isdir(class_path):
        continue

    # Get images
    images = os.listdir(class_path)

    # Randomize them
    random.shuffle(images)

    total = len(images)

    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Create class folders
    os.makedirs(
        os.path.join(train_dir, class_name),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(val_dir, class_name),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(test_dir, class_name),
        exist_ok=True
    )

    # Copy training images
    for image in train_images:
        shutil.copy(
            os.path.join(class_path, image),
            os.path.join(train_dir, class_name, image)
        )

    # Copy validation images
    for image in val_images:
        shutil.copy(
            os.path.join(class_path, image),
            os.path.join(val_dir, class_name, image)
        )

    # Copy test images
    for image in test_images:
        shutil.copy(
            os.path.join(class_path, image),
            os.path.join(test_dir, class_name, image)
        )

    print(
        class_name,
        "Train:", len(train_images),
        "Val:", len(val_images),
        "Test:", len(test_images)
    )
