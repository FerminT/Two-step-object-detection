import pandas as pd
import tqdm
from pathlib import Path

data_path = Path(__file__).parents[1] / 'object_detection' / 'all_classes'
splits = ['train', 'validation', 'test']
# The classes index in the array correspond to their id in YOLO format
classes = ['Bus', 'Truck', 'Car', 'Van', 'Taxi', 'Tree', 'Backpack', 'Handbag', 'Shorts', 'Shirt', 'Glasses', 'Sunglasses', 'Boot', 'Sandal', 'Traffic sign']
# Save classes to txt separated by line break
classes_file = data_path / 'classes.txt'
if not classes_file.parents[1].exists(): classes_file.parents[1].mkdir(parents=True)
with classes_file.open('w') as fp:
    for class_ in classes:
        fp.write(class_ + '\n')

metadata_path = Path(__file__).parents[2] / 'openimages_utils'
class_descriptions = pd.read_csv(metadata_path / 'class-descriptions-boxable.csv')
labels_id = class_descriptions[class_descriptions['DisplayName'].isin(classes)]['LabelName'].tolist()
mapping   = {class_descriptions[class_descriptions['DisplayName'] == cat]['LabelName'].tolist()[0] : cat for cat in classes}

images_path = data_path / 'images'
labels_path = data_path / 'labels'
for split in splits:
    split_path = images_path / split
    split_labels_path = labels_path / split
    if not split_labels_path.exists(): split_labels_path.mkdir(parents=True)

    split_imgids = [imgpath.name.split('.')[0] for imgpath in split_path.glob('*.jpg')]
    split_anns = pd.read_csv(metadata_path / 'annotations' / (split_path.name + '-annotations-bbox.csv'))
    split_anns = split_anns[split_anns['ImageID'].isin(split_imgids)]
    split_anns = split_anns[split_anns['LabelName'].isin(labels_id)]
    for imgid in tqdm(split_imgids):
        img_anns = split_anns[split_anns['ImageID'] == imgid][['LabelName', 'XMin', 'YMin', 'XMax', 'YMax']]
        img_anns['LabelName'] = img_anns['LabelName'].apply(lambda x: classes.index(mapping[x]))
        xmin = img_anns['XMin']
        ymin = img_anns['YMin']
        # Convert to YOLO format
        img_anns['XMin'] = (img_anns['XMin'] + img_anns['XMax']) / 2
        img_anns['YMin'] = (img_anns['YMin'] + img_anns['YMax']) / 2
        img_anns['XMax'] = img_anns['XMax'] - xmin
        img_anns['YMax'] = img_anns['YMax'] - ymin
        img_anns.to_csv(split_labels_path / (imgid + '.txt'), sep=' ', header=False, index=False)