from augmentation.operations import flip, rotate, shear, skew, crop, elastic_distortion, grid_distortion

AUGMENTATIONS = [
    ('Flip', lambda image: flip(image, 'v')),
    ('Rotate', lambda image: rotate(image, 90)),
    ('Skew', skew),
    ('Shear', lambda image: shear(image, 0.3, horizontal=True)),
    ('Crop', lambda image: crop(image, (0, 0, 100, 100))),
    ('ElasticDistortion', lambda image: elastic_distortion(image)),
    ('GridDistortion', lambda image: grid_distortion(image, 4, 30))
]
