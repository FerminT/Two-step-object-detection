import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def get_classes_imgs(labels_path, classes):
    annotations = labels_path.glob('*.txt')
    classes_anns = {class_ : [] for class_ in classes}
    for annotation in annotations:
        with annotation.open('r') as fp:
            anns = fp.readlines()
        for ann in anns:
            class_ = int(ann.split(' ')[0])
            classes_anns[classes[class_]].append(annotation.name[:-4] + '.jpg')

    return classes_anns

def plot_instances(labels_path, classes):
    classes_anns = get_classes_imgs(labels_path, classes)
    for class_ in classes_anns:
        print(f'Number of {class_} instances: {len(classes_anns[class_])}')

    # Create histogram of the number of instances per class
    plt.bar(classes_anns.keys(), [len(classes_anns[class_]) for class_ in classes_anns])
    plt.xticks(rotation=90)
    plt.title(f'Number of instances per class in {labels_path.name} set')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="Dataset class - train, validation or test", required=True)
    parser.add_argument("--dataset", help="Dataset path", required=True)
    args = parser.parse_args()

    data_path = Path(args.dataset)
    labels_path = data_path / 'labels' / args.mode
    classes_file = data_path / 'classes.txt'
    with classes_file.open('r') as fp:
        classes = [class_.strip() for class_ in fp.readlines()]

    plot_instances(labels_path, classes)