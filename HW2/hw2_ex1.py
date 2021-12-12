from time import time
import sys
import argparse
import os
import tensorflow_model_optimization as tfmot
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


class WindowGenerator:
    def __init__(self, input_width, label_step, label_options, mean, std, verbose=False):
        self.input_width = input_width
        self.label_options = label_options
        self.label_step = label_step
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])
        self.verbose = verbose

    def split_window(self, features):
        # input_indeces = np.arange(self.input_width)
        inputs = features[:, :self.input_width, :] 
        labels = features[:, self.input_width:, :]
        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, self.label_step, 2])
        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)
        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)
        return inputs, labels

    def make_dataset(self, data, train):
        # It returns a tf.data.Dataset instance
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.input_width+self.label_step,
                sequence_stride=1,
                batch_size=32)
        # Maps map_func across the elements of this dataset.
        dataset = dataset.map(map_func=self.preprocess)
        if self.verbose:
            print(f"Dataset element: {dataset.element_spec}")
        # Caches the elements in this dataset.
        # The first time the dataset is iterated over, its elements will be cached either 
        # in the specified file or in memory. Subsequent iterations will use the cached data.
        dataset = dataset.cache() 
        if train is True:
            dataset = dataset.shuffle(100, reshuffle_each_iteration=True)

        return dataset


class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name='multi_output_mae', **kwargs):
        super().__init__(name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros', shape=[2])
        self.count = self.add_weight('count', initializer='zeros')

    def reset_state(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=[0, 1])
        self.total.assign_add(error)
        self.count.assign_add(1)
        return

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)
        return result


def get_data(args):

    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True,
        cache_dir='../../../../Downloads',
        cache_subdir='data'
    )

    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path)
    
    column_indices = [2, 5]
    columns = df.columns[column_indices]
    
    data = df[columns].values.astype(np.float32)
    n = len(data)
    
    train_data = data[0:int(n*0.7)]
    val_data = data[int(n*0.7):int(n*0.9)]
    test_data = data[int(n*0.9):]

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    generator = WindowGenerator(args.window_length, args.step, args.labels, mean, std)
    train_ds = generator.make_dataset(train_data, True)
    val_ds = generator.make_dataset(val_data, False)
    test_ds = generator.make_dataset(test_data, False)

    return train_ds, val_ds, test_ds

def initialize_model(model_name, input_width, output_steps, labels, alpha):

    units = output_steps * labels
    input_shape = (input_width,labels)

    if model_name == "MLP":
        model = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(units=int(alpha*128), activation='relu'),
            keras.layers.Dense(units=int(alpha*128), activation='relu'),
            keras.layers.Dense(units=units)
        ])
        
    elif model_name == "CNN":
        model = keras.Sequential([
            keras.layers.Conv1D(filters=int(alpha*64), 
                                kernel_size=3, 
                                activation='relu',
                                input_shape = input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(units=int(alpha*64), activation='relu'),
            keras.layers.Dense(units=units),
            keras.layers.Reshape([output_steps,labels])
        ])
        
      
    model.build(input_shape)  
    model.compile(optimizer='adam', 
                  loss='mse', 
                  metrics=[MultiOutputMAE()])

    return model

def save_tflite_model(model, model_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open(f'{model_name}', 'wb') as f:
        f.write(tflite_model)



def main(args):
    
    # generate the window and define the datasets
    train_ds, val_ds, test_ds = get_data(args)
    
    if args.magnitude:
      pruning_params = {'pruning_schedule':
        tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.30, 
        final_sparsity=0.95, 
        begin_step=len(train_ds)*5, # starts after 5 epochs  
        end_step=len(train_ds)*15) # ends after 15 epochs
      }
    else:
      pruning_params=None

    NAME = args.model + "_" + str(args.version) + "_" + str(args.epochs) 
    path_to_dir = 'models/' + NAME

    model = initialize_model(args.model, args.window_length, args.step, args.labels, args.alpha)

    # Pruning callback:
    if pruning_params is not None:
      print("qui")
      prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
      model = prune_low_magnitude(model, **pruning_params)  # new version which is able to perform pruning
      callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    else: 
      callbacks = []

    model.compile(
        optimizer='adam',
        loss=keras.losses.MeanSquaredError(),
        metrics=[MultiOutputMAE()]
    )

    # let's train for 20 epochs
    input_shape = [32, 6, 2]
    model.build(input_shape)
    history = model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks) 

    loss, mae = model.evaluate(test_ds)
    
    if args.magnitude:
      model = tfmot.sparsity.keras.strip_pruning(model)

    model.save(path_to_dir) 

    save_tflite_model(model, f'{NAME}.tflite')


    print(f"Loss = {loss} , MAE = {mae}")

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-m','--model', type=str, required=True, help='model name', choices=['MLP','LSTM','CNN'])
    parser.add_argument('-s','--step', type=int, help='Frame step', default=3)
    parser.add_argument('-l','--labels', type=int, help='model output', default=2, choices=[0,1,2])
    parser.add_argument('-e','--epochs', type=int, help='training epochs', default=1)
    parser.add_argument('-w','--window_length', type=int, help='length of the sliding window', default=6)
    parser.add_argument('-a','--alpha', type=float, help='set width multiplier', default=1)
    parser.add_argument('--magnitude', type=bool, help='activate magnitude based pruning', default=False)
    parser.add_argument('-v', '--version', type=str, default='a')
    args = parser.parse_args()

    main(args)
