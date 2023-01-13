from pathlib import Path
import numpy as np

labels_path = Path(__file__).parents[1] / Path('object_detection', 'all_classes', 'labels')
splits = ['train', 'validation', 'test']
classes = [2, 4] # cars and taxis
threshold = 0.05
for split in splits:
    txts = (labels_path / split).glob('*.txt')
    all_classes = []
    overlapping_instances = []
    for txt in txts:
        with txt.open('r') as fp:
            anns = fp.readlines()
        anns_classes = []
        anns_coords = {class_ : [] for class_ in classes}
        for ann in anns:
            class_ = int(ann.split(' ')[0])
            if class_ in classes:
                anns_classes.append(class_)
                anns_coords[class_].append([float(coord) for coord in ann.split(' ')[1:3]])

        if set(classes) == set(anns_classes):
            all_classes.append(txt.name)
            for class_ in anns_coords:
                class_coords = np.array(anns_coords[class_])
                other_classes = list(classes)
                other_classes.remove(class_)
                for other_class in other_classes:
                    other_class_coords = np.array(anns_coords[other_class])
                    for coords in class_coords:
                        dif = np.abs(other_class_coords - coords)
                        if np.any(np.all(dif < threshold, axis=1)) and not txt.name in overlapping_instances:
                            overlapping_instances.append(txt.name)
    print(f'{split} split: {len(all_classes)} images with cars and taxis; {len(overlapping_instances)} overlapping instances')