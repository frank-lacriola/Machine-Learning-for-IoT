import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras


class WindowGenerator:
    def __init__(self, input_width , label_options, mean, std):
        self.input_width = input_width # the number of samples contained in a single window
        self.label_options = LABEL_OPTIONS
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2]) # sec arg is the shape and during training the first dim is batch
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2]) #

    def split_window(self, features):
        # here the assumption is that features already contains 7 values
        inputs = features[:, :-1, :]  # we leave the last one as label, remember we have the batch at the first dim

        if self.label_options < 2:
            labels = features[:, -1, self.label_options]
            labels = tf.expand_dims(labels, -1)
            num_labels = 1
        else:
            labels = features[:, -1, :]
            num_labels = 2

        inputs.set_shape([None, self.input_width, 2])  # we set the batch dim as None because it will be needed later
        labels.set_shape([None, num_labels])  # same for labels

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)  # we add a small quantity to avoid divisions by zero

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)  # we don't need to normalize labels

        return inputs, labels

    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array( # this is what we need
                data=data,
                targets=None,
                sequence_length=input_width+1,  # + label
                sequence_stride=1,
                batch_size=32)  # we'll try to change batch size later
        ds = ds.map(self.preprocess)  # the preprocess will be applied to every batch
        ds = ds.cache()

        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True) # 100 is the shuffling buffer size

        return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model name among mlp cnn and lstm')
    parser.add_argument('--labels', type=int, default=0,
                        help='0 for temp forecasting, 1 for hum forecasting, 2 or more for both')
    args = parser.parse_args()

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')

    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path)

    column_indices = [2, 5]
    columns = df.columns[column_indices]
    data = df[columns].values.astype(np.float32)

    n = len(data)
    train_data = data[0:int(n * 0.7)]
    val_data = data[int(n * 0.7):int(n * 0.9)]
    test_data = data[int(n * 0.9):]

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    input_width = 6
    LABEL_OPTIONS = args.labels

    generator = WindowGenerator(input_width, LABEL_OPTIONS, mean, std)
    train_ds = generator.make_dataset(train_data, True)
    val_ds = generator.make_dataset(val_data, False)
    test_ds = generator.make_dataset(test_data, False)

    if args.model == 'mlp':
        model = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(units=128),
            keras.layers.ReLU(),
            keras.layers.Dense(units=128),
            keras.layers.ReLU(),
            keras.layers.Dense(units=1)
        ])
    elif args.model == 'cnn':
        model = keras.Sequential([
            keras.layers.Conv1D(filters=64, kernel_size=3),
            keras.layers.ReLU(),
            keras.layers.Flatten(),
            keras.layers.Dense(units=64),
            keras.layers.ReLU(),
            keras.layers.Dense(units=1)
        ])
    elif args.model == 'lstm':
        model = keras.Sequential([
            keras.layers.LSTM(units=64),
            keras.layers.Flatten(),
            keras.layers.Dense(units=1)
        ])

    model.compile(
        optimizer='adam',
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )

    # let's train for 20 epochs

    history = model.fit(train_ds, validation_data=val_ds, epochs=20)
    test_loss, test_mae = model.evaluate(test_ds)
    print("{} \n The MAE is: {}".format(model.summary(), test_mae))


if __name__ == '__main__':
    main()