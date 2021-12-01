import tensorflow as tf
from tensorflow import keras
import argparse
import os
from board import D4
import adafruit_dht
import pandas as pd
import datetime
import time
from lab3_ex1 import WindowGenerator
import statistics



parser = argparse.ArgumentParser()
parser.add_argument('--saved_model_dir', required=True, help='path to saved model')
args = parser.parse_args()


converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model_dir)
tflite_model = converter.convert()
# The value returned by convert is a “bytes”
# object which contains the serialized model, and
# can be directly written to a binary file

with open('model.tflite', 'wb') as fp:
    fp.write(tflite_model)

print(os.path.getsize('model.tflite'))

dht_device = adafruit_dht.DHT11(D4)
temperature = dht_device.temperature
humidity = dht_device.humidity

freq = 1
period = 6
temps_list = []
hum_list = []

for i in range(int(period/freq)):
    temps_list.append(temperature)
    hum_list.append(humidity)
    time.sleep(3)

generator = WindowGenerator(6, 2, mean, statistics.stdev(te))
train_ds = generator.make_dataset(train_data, True)

