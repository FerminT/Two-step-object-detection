from pathlib import Path
import numpy as np

labels_path = Path(__file__).parents[2] / Path('object_detection', 'all_classes', 'labels')
splits = ['train', 'validation', 'test']
# Classes to check for overlapping; the second one will be the one to keep
classes = [6, 7] # glasses and sunglasses
threshold = 0.05
for split in splits:
    txts = (labels_path / split).glob('*.txt')
    overlapping_instances = []
    instances_removed = 0
    for txt in txts:
        with txt.open('r') as fp:
            anns = fp.readlines()
        anns_classes = []
        anns_coords = {class_: {'line_value': [], 'coords': []} for class_ in classes}
        for ann in anns:
            class_ = int(ann.split(' ')[0])
            if class_ in classes:
                anns_classes.append(class_)
                anns_coords[class_]['coords'].append([float(coord) for coord in ann.split(' ')[1:3]])
                anns_coords[class_]['line_value'].append(ann)

        if set(classes) == set(anns_classes):
            other_classes = list(classes)
            for class_ in anns_coords:
                class_coords  = np.array(anns_coords[class_]['coords'])
                other_classes.remove(class_)
                for other_class in other_classes:
                    other_class_coords = np.array(anns_coords[other_class]['coords'])
                    for i, coords in enumerate(class_coords):
                        dif = np.abs(other_class_coords - coords)
                        if np.any(np.all(dif < threshold, axis=1)):
                            line_value = anns_coords[class_]['line_value'][i]
                            anns.remove(line_value)
                            instances_removed += 1
                            if not txt.name in overlapping_instances:
                                overlapping_instances.append(txt.name)
        with open(txt, 'w') as fp:
            for ann in anns:
                fp.write(ann)
    print(f'{split} split: deleted {instances_removed} instances of overlapping class; {len(overlapping_instances)} files modified')