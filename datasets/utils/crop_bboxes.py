from pathlib import Path
from skimage import io, exposure
from tqdm import tqdm
import random
import argparse
import albumentations as ab
import numpy as np

def crop_bounding_boxes(classes_to_crop, padding, crop_range, augment_factor, build_background):
    # Aug_factor format: list of ints [train, validation, test]
    splits = {'train': {'augfactor': augment_factor[0]}, 'validation': {'augfactor': augment_factor[1]}, 'test': {'augfactor': augment_factor[2]}}
    # classes_to_crop = ['Car', 'Taxi', 'Backpack', 'Handbag', 'Glasses', 'Sunglasses']
    # How much black area to allow in the cropped image for background class
    max_black_area = 0.25
    max_attempts = 10

    save_path = Path(__file__).parents[1] / Path('img_classification')
    if not save_path.exists(): save_path.mkdir()

    data_path = Path(__file__).parents[1] / 'object_detection' / 'all_classes'
    classes_file = data_path / 'classes.txt'
    with classes_file.open('r') as fp:
        classes = [class_.strip() for class_ in fp.readlines()]
    classes_indices = [classes.index(class_.replace('_', ' ')) for class_ in classes_to_crop]

    for split in splits:
        print('Processing {} set'.format(split))
        images_path = data_path / 'images' / split
        labels_path = data_path / 'labels' / split
        
        annotations = list(labels_path.glob('*.txt'))
        for it in range(splits[split]['augfactor']):
            for annotation_file in tqdm(annotations):
                with annotation_file.open('r') as fp:
                    bboxes = fp.readlines()
                img_name = annotation_file.name[:-4] + '.jpg'
                img = io.imread(str(images_path / img_name))
                masked_img = np.copy(img)
                crops = []
                h, w = img.shape[:2]
                for bbox in bboxes:
                    bbox_classid = int(bbox.split(' ')[0])
                    if bbox_classid in classes_indices:
                        class_name = classes[bbox_classid]
                        if build_background:
                            class_path = Path(save_path, split, 'Objects')
                            img_name   = annotation_file.name[:-4] + '_{}.jpg'.format(class_name)
                        else:
                            class_path = Path(save_path, split, class_name)
                        if not class_path.exists(): class_path.mkdir(parents=True)

                        x_center, y_center, obj_w, obj_h = [float(coord) * w if i % 2 == 0 else float(coord) * h for i, coord in enumerate(bbox.split(' ')[1:])]
                        # Rectangle to square
                        max_dim = max(obj_h, obj_w)
                        squared_bbox = [int(y_center - max_dim / 2), int(x_center - max_dim / 2), \
                            int(y_center + max_dim / 2), int(x_center + max_dim / 2)]
                        # Apply padding
                        from_ = (max(0, squared_bbox[0] - padding // 2), max(0, squared_bbox[1] - padding // 2))
                        to    = (min(h, squared_bbox[2] + padding // 2), min(w, squared_bbox[3] + padding // 2))
                        obj_img = img[from_[0]:to[0], from_[1]:to[1]]
                        augment = it > 0
                        # Black area for building background class
                        masked_img[from_[0]:to[0], from_[1]:to[1]] = 0

                        if augment:
                            # Do not increase cars instances
                            if class_name == 'Car': continue
                            crop_ratio = random.uniform(crop_range[0], crop_range[1])
                            transform  = ab.Compose([
                                ab.CenterCrop(height=int(obj_img.shape[0] * crop_ratio), width=int(obj_img.shape[1] * crop_ratio), p=1.0),
                                ab.HorizontalFlip(p=0.8),
                                ab.VerticalFlip(p=0.5)])
                            obj_img  = transform(image=obj_img)['image']
                            img_name = annotation_file.name[:-4] + '_aug{}.jpg'.format(it + i)

                        valid_crop = obj_img is not None and obj_img.any()
                        if valid_crop:
                            io.imsave(str(Path(class_path, img_name)), obj_img)
                            crops.append(obj_img.shape[:2])

                if build_background:
                    # Create approximately the same number of instances for the background class
                    for i, crop_size in enumerate(crops):
                        transform  = ab.RandomCrop(crop_size[0], crop_size[1])
                        black_area = 1.0
                        attempts = 0
                        while black_area > max_black_area and attempts < max_attempts:
                            background = transform(image=masked_img)['image']
                            black_area = (background == 0).sum() / np.prod(background.shape[:2])
                            attempts += 1
                        
                        if black_area < max_black_area and not exposure.is_low_contrast(background):
                            background_path = Path(save_path, split, 'Background')
                            if not background_path.exists(): background_path.mkdir(parents=True)
                            img_name = annotation_file.name[:-4] + '_{}.jpg'.format(i)
                            io.imsave(str(Path(background_path, img_name)), background)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', nargs='+', required=True)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--crop_range', nargs='+', type=int, default=[0.4, 0.8])
    parser.add_argument('--aug_factor', type=int, nargs='+', default=[1, 1, 1])
    parser.add_argument('--background', action='store_true', default=False)

    args = parser.parse_args()
    if args.aug_factor is not None and len(args.aug_factor) != 3:
        raise ValueError('Augmentation factor must be a list of 3 ints (one for each split)')
    if args.crop_range is not None and len(args.crop_range) != 2:
        raise ValueError('Crop range must be a list of 2 floats (min and max crop ratio)')

    crop_bounding_boxes(args.classes, args.padding, args.crop_range, args.aug_factor, args.background)