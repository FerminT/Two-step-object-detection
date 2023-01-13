# Author : Sunita Nayak, Big Vision LLC

#### Usage example: python3 downloadOI.py --classes 'Ice_cream,Cookie' --mode train

import argparse
import csv
import subprocess
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool as thread_pool

cpu_count = multiprocessing.cpu_count()

parser = argparse.ArgumentParser(description="Download Class specific images from OpenImagesV6")
parser.add_argument("--mode", help="Dataset category - train, validation or test", required=True)
parser.add_argument("--classes", help="Names of object classes to be downloaded", required=True)
parser.add_argument("--nthreads", help="Number of threads to use", required=False, type=int, default=cpu_count * 2)
parser.add_argument("--occluded", help="Include occluded images", required=False, type=int, default=1)
parser.add_argument("--truncated", help="Include truncated images", required=False, type=int, default=1)
parser.add_argument("--groupOf", help="Include groupOf images", required=False, type=int, default=1)
parser.add_argument("--depiction", help="Include depiction images", required=False, type=int, default=1)
parser.add_argument("--inside", help="Include inside images", required=False, type=int, default=1)

args = parser.parse_args()

run_mode = args.mode

threads = args.nthreads

classes = []
for class_name in args.classes.split(","):
    classes.append(class_name)

# Read `class-descriptions-boxable.csv`
with open("./class-descriptions-boxable.csv", mode="r", encoding='utf-8') as infile:
    reader = csv.reader(infile)
    dict_list = {rows[1]: rows[0] for rows in reader}  # rows[1] is ClassName, rows[0] is ClassCode

max_number_of_images = {'train': 10000, 'validation': 500, 'test': 1000}

# subprocess.run(["rm", "-rf", run_mode])
subprocess.run(["mkdir", run_mode])

pool = thread_pool(threads)
commands = []
cnt = 0
imgs_class = {class_name : [] for class_name in classes}
classes_exceding_limit = []

dataset_path = os.path.join(os.path.pardir, 'dataset', 'divided_by_class')

for ind in range(0, len(classes)):
    class_name = classes[ind]
    print("Class " + str(ind) + " : " + class_name)

    command = "grep " + dict_list[class_name.replace("_", " ")] + " ./annotations/" + run_mode + "-annotations-bbox.csv"
    class_annotations = subprocess.run(command.split(), stdout=subprocess.PIPE).stdout.decode("utf-8")
    class_annotations = class_annotations.splitlines()

    class_path = os.path.join(dataset_path, run_mode, class_name.replace(" ", "_"))
    if not os.path.exists(class_path):
        os.makedirs(class_path, exist_ok=True)

    for line in class_annotations:
        line_parts = line.split(",")
        img_id = line_parts[0]
        save_path = os.path.join(class_path, img_id + ".jpg")

        # If image exists, skip
        if os.path.exists(save_path):
            continue

        # Download options: IsOccluded, IsTruncated, IsGroupOf, IsDepiction, IsInside
        if args.occluded == 0 and int(line_parts[8]) > 0:
            print("Skipped %s", img_id)
            continue
        if args.truncated == 0 and int(line_parts[9]) > 0:
            print("Skipped %s", img_id)
            continue
        if args.groupOf == 0 and int(line_parts[10]) > 0:
            print("Skipped %s", img_id)
            continue
        if args.depiction == 0 and int(line_parts[11]) > 0:
            print("Skipped %s", img_id)
            continue
        if args.inside == 0 and int(line_parts[12]) > 0:
            print("Skipped %s", img_id)
            continue

        # Command to download
        command = f"aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/'{run_mode}'/'{img_id}'.jpg {save_path}"
        commands.append(command)
        imgs_class[class_name].append(img_id)
        cnt += 1
    
    if len(imgs_class[class_name]) > max_number_of_images[run_mode]:
        classes_exceding_limit.append(class_name)
        print(f"Class {class_name} exceeds limit of {str(max_number_of_images[run_mode])} images")

for exceded_class in classes_exceding_limit:
    class_imgs = set(imgs_class[exceded_class])
    imgs_in_other_classes = set()
    for class_name in classes:
        if class_name != exceded_class:
            imgs_in_other_classes.update(set(imgs_class[class_name]) & class_imgs)
    
    for img_id in list(class_imgs):
        if not img_id in imgs_in_other_classes and len(class_imgs) > max_number_of_images[run_mode]:
            class_path = os.path.join(run_mode, class_name.replace(" ", "_"))
            save_path  = os.path.join(class_path, img_id + ".jpg")
            command = f"aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/'{run_mode}'/'{img_id}'.jpg {save_path}"

            if command in commands:
                commands.remove(command)
                class_imgs.remove(img_id)
                cnt -= 1
    # If there are still more images than the limit, remove the rest
    if len(class_imgs) > max_number_of_images[run_mode]:
        excess_imgs = list(class_imgs)[max_number_of_images[run_mode] + 1:]
        imgs_class[exceded_class] = list(class_imgs)[:max_number_of_images[run_mode]]

        for img_id in excess_imgs:
            class_path = os.path.join(run_mode, exceded_class.replace(" ", "_"))
            save_path  = os.path.join(class_path, img_id + ".jpg")
            command = f"aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/'{run_mode}'/'{img_id}'.jpg {save_path}"

            if command in commands:
                commands.remove(command)
                cnt -= 1
    print("Exceded class " + exceded_class + " brought down to " + str(len(imgs_class[exceded_class])) + " images")

# Print number of images per class
for class_name in classes:
    print(f"Class {class_name} has {str(len(imgs_class[class_name]))} images")

print("Annotation Count : " + str(cnt))
commands = list(set(commands))
print("Number of images to be downloaded : " + str(len(commands)))

list(tqdm(pool.imap(os.system, commands), total=len(commands)))

pool.close()
pool.join()