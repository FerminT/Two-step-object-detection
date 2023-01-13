import tensorflow as tf
from pathlib import Path
from sklearn.utils import class_weight
import numpy as np
import argparse
import time

SAVE_PATH = Path('trained_models')

def augment_data(images, labels):
    img_size = 224
    batch_size = 32
    seed   = (1234, 1234)

    pad    = (int(img_size / 8), int(img_size / 8))
    resize = [batch_size, img_size, img_size, 3]

    images = tf.image.resize_with_crop_or_pad(images, img_size + pad[0], img_size + pad[1])
    images = tf.image.stateless_random_crop(images, size=resize, seed=seed)
    images = tf.image.stateless_random_brightness(images, 0.5, seed)
    images = tf.image.stateless_random_contrast(images, 0.2, 1.5, seed)
    images = tf.image.stateless_random_saturation(images, 0.3, 1.5, seed)
    images = tf.image.stateless_random_flip_left_right(images, seed)
    images = tf.image.stateless_random_flip_up_down(images, seed)
    images = tf.image.stateless_random_hue(images, 0.015, seed)

    return images, labels

def train(model_name, dataset_path, num_workers, batch_size, epochs, img_size, weigh_classes, augment_factor):
    backbone = tf.keras.applications.EfficientNetB1(
        include_top=False,
        weights='imagenet', 
        pooling='avg',
        input_shape=(img_size, img_size, 3)
        )

    # backbone.trainable = False
    datapath_train = dataset_path / 'train'
    datapath_valid = dataset_path / 'validation'
    num_classes = len(list(datapath_train.iterdir()))

    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=str(datapath_train),
        batch_size=batch_size,
        image_size=(img_size, img_size),
        seed=1234
        )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=str(datapath_valid),
        batch_size=batch_size,
        image_size=(img_size, img_size),
        seed=1234
        )

    class_weights = None
    if weigh_classes:
        labels = [list(batch[1].numpy()) for batch in train_dataset]
        labels = [batch_labels for batch in labels for batch_labels in batch]
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    AUTOTUNE = tf.data.AUTOTUNE
    if augment_factor > 0:
        train_dataset = train_dataset.repeat(augment_factor).shuffle(train_dataset.cardinality().numpy())
        train_dataset = train_dataset.map(augment_data, num_parallel_calls=AUTOTUNE)
    
    train_dataset = train_dataset.prefetch(AUTOTUNE)

    # Set logger
    model_path = SAVE_PATH / model_name
    logdir = model_path / 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(logdir))
    csv_callback   = tf.keras.callbacks.CSVLogger(str(logdir / 'training.csv'))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    model.summary()
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[early_stopping, tensorboard_callback, csv_callback],
        workers=num_workers,
        use_multiprocessing=True,
        class_weight=class_weights
    )

    model.save(str(model_path))

def predict(model_name, img_path, img_size):
    model_path = SAVE_PATH / model_name
    model = tf.keras.models.load_model(str(model_path))
    img_test = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_size, img_size))
    img_test = np.array([tf.keras.preprocessing.image.img_to_array(img_test)])
    t0 = time.time()
    prediction = model(img_test)
    t = time.time()
    print(prediction)
    print('Feed forward time: ' + str(1E3 * (t - t0)) + 'ms')

def quantize(model_name):
    model_path = SAVE_PATH / model_name
    if not model_path.exists(): 
        raise Exception('Model does not exist.')

    model = tf.keras.models.load_model(model_path)
    print('Quantizing model ' + model_name + '...')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    model_q   = converter.convert()

    model_file = model_path / 'qmodel.tflite'
    model_file.write_bytes(model_q)

def predict_quantized(model_name, img_path, img_size):
    model_path = SAVE_PATH / model_name / 'qmodel.tflite'
    if not model_path.exists():
        quantize(model_name)
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    img_test = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_size, img_size))
    img_test = np.array([tf.keras.preprocessing.image.img_to_array(img_test)])

    t0 = time.time()
    input_index  = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, img_test)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)
    t = time.time()
    print(prediction)
    print('Feed forward time: ' + str(1E3 * (t - t0)) + 'ms')  

def test(model_name, dataset_path, batch_size, img_size):
    datapath_test = dataset_path / 'test'
    model = tf.keras.models.load_model(str(SAVE_PATH / model_name))
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=datapath_test,
        batch_size=batch_size,
        image_size=(img_size, img_size),
        seed=1234
        )

    model.evaluate(test_dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--augment', type=int, default=0)
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--weigh-classes', action='store_true')
    parser.add_argument('--predict', type=str)
    parser.add_argument('--predict_quantized', type=str)

    args = parser.parse_args()
    if args.dataset:
        dataset_path = Path(__file__).parents[1] / 'datasets' / 'img_classification' / args.dataset
    if args.train:
        train(args.name, dataset_path, args.workers, args.batch_size, args.epochs, args.img_size, args.weigh_classes, args.augment)
    if args.test:
        test(args.name, dataset_path, args.batch_size, args.img_size)
    if args.predict:
        predict(args.name, args.predict, args.img_size)
    if args.quantize:
        quantize(args.name)
    if args.predict_quantized:
        predict_quantized(args.name, args.predict_quantized, args.img_size)