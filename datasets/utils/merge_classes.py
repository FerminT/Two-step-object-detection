from pathlib import Path

# Mapping from original classes to merged classes
# 0 - Bus
# 1 - Truck
# 2 - Car, Taxi
# 3 - Van
# 4 - Tree
# 5 - Backpack, Handbag
# 6 - Shorts
# 7 - Shirt
# 8 - Glasses, Sunglasses
# 9 - Boot
# 10 - Sandal
# 11 - Traffic sign

def getindexof(partvalue, list):
    for i, elem in enumerate(list):
        if partvalue in elem: return i
    return -1

splits = ['train', 'validation', 'test']

all_classes_path    = Path(__file__).parents[1] / 'object_detection' / 'all_classes'
merged_classes_path = Path(__file__).parents[1] / 'object_detection' / 'merged_classes'
if not merged_classes_path.exists(): merged_classes_path.mkdir()

# The classes index in the array correspond to their id in YOLO format
classes_file = all_classes_path / 'classes.txt'
with classes_file.open('r') as fp:
    all_classes = fp.readlines()
    all_classes = [class_.strip() for class_ in all_classes]

merged_classes = ['Bus', 'Truck', 'Car/Taxi', 'Van', 'Tree', 'Backpack/Handbag', 'Shorts', 'Shirt', 'Sunglasses/Glasses', 'Boot', 'Sandal', 'Traffic sign']

merged_classes_file = merged_classes_path / 'classes.txt'
with merged_classes_file.open('w') as fp:
    for class_ in merged_classes:
        fp.write(class_ + '\n')

for split in splits:
    split_path = all_classes_path / 'labels' / split
    new_split_path = merged_classes_path / 'labels' / split
    if not new_split_path.exists(): new_split_path.mkdir(parents=True)
    annotation_files = split_path.glob('*.txt')
    for ann_file in annotation_files:
        with ann_file.open('r') as fp:
            annotations = fp.readlines()
        
        new_annons = [str(getindexof(all_classes[int(line.split(' ')[0])], merged_classes)) + line[len(line.split(' ')[0]):] for line in annotations]
        
        new_anns_file = new_split_path / ann_file.name
        with new_anns_file.open('w') as fp:
            for annon in new_annons:
                fp.write(annon)