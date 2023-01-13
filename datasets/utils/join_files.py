import shutil
from pathlib import Path
from os import listdir

""" Images are assumed to be located in ../object_detection/all_classes/images/[split]/class/*.jpg
    This script will copy all images to ../object_detection/all_classes/images/[split]/*.jpg """

splits = ['train', 'validation', 'test']
dataset_path = Path(__file__).parents[1] / 'object_detection' / 'all_classes' / 'images'
for split in splits:
    split_count = 0
    classes = listdir(split)
    for class_ in classes:
        # Move all images in class to split folder
        class_dir = Path(dataset_path, split, class_)
        class_files = list(class_dir.glob('*.jpg'))
        for file in class_files:
            shutil.copy(file, class_dir.parents[1] / file.name)
            split_count += 1
    
    print(f'Copied {split_count} images to {split} folder')