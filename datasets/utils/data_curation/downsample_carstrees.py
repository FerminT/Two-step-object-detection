import argparse
from pathlib import Path
from ..stats import get_classes_imgs

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="Dataset class - train, validation or test", required=True)
args = parser.parse_args()

images_path = Path(__file__).parents[2] / 'object_detection' / 'all_classes' / 'images' / args.mode
labels_path = Path(__file__).parents[2] / 'object_detection' / 'all_classes' / 'labels' / args.mode
classes = ['Bus', 'Truck', 'Car', 'Van', 'Taxi', 'Tree', 'Backpack', 'Handbag', 'Shorts', 'Shirt', 'Glasses', 'Sunglasses', 'Boot', 'Sandal', 'Traffic sign']

classes_anns = get_classes_imgs(labels_path, classes)

# Get the images id from Car and Tree that are not in any other class
car_imgs_noother  = set(classes_anns['Car'])
tree_imgs_noother = set(classes_anns['Tree'])
for class_ in classes_anns:
    if class_ != 'Car' and class_ != 'Tree':
        car_imgs_noother  = car_imgs_noother - set(classes_anns[class_])
        tree_imgs_noother = tree_imgs_noother - set(classes_anns[class_])

print(f'Number of Car images that are not in any other class: {len(car_imgs_noother)}')
print(f'Number of Tree images that are not in any other class: {len(tree_imgs_noother)}')

# We want to avoid downsampling too much, so we will only delete some of the images
if images_path.name == 'validation':
    tree_imgs_noother = tree_imgs_noother[:1500]
elif images_path.name == 'test':
    tree_imgs_noother = tree_imgs_noother[:4350]

# Delete the images of Car and Tree that are not in any other class
imgs_to_delete = set(car_imgs_noother + tree_imgs_noother)
del_count = 0
for imgid in imgs_to_delete:
    filename = Path(images_path, imgid + '.jpg')
    if filename.exists():
        filename.unlink()
        del_count += 1

print(f'Deleted {del_count} images')