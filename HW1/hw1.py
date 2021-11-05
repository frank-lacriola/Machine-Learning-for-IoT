from argparse import ArgumentParser

from board import D4
import adafruit_dht
import pandas as pd
import datetime
import time
import tensorflow as tf
import numpy as np
import pandas as pd

"""The TFRecord format is a simple format for storing a sequence of binary records.
Protocol buffers are a cross-platform, cross-language library for efficient serialization of structured data.
Protocol messages are defined by .proto files, these are often the easiest way to understand a message type.
The tf.train.Example message (or protobuf) is a flexible message type that represents a {"string": value} mapping. 
It is designed for use with TensorFlow and is used throughout the higher-level APIs such as TFX.
The tf.train.Feature message type can accept one of the following three types:

tf.train.BytesList (the following types can be coerced):
string
byte

tf.train.FloatList (the following types can be coerced):
float (float32)
double (float64)

tf.train.Int64List (the following types can be coerced):
bool
enum
int32
uint32
int64
uint64

All proto messages can be serialized to a binary-string using the .SerializeToString method
"""


def tfRecord(args):
    # SCHEMA: date, time, temp, humidity
    df = pd.read_csv(filepath_or_buffer=args.input)

    with tf.io.TFRecordWriter(args.output) as writer:
        for i in range(df.shape[0]):
            raw_date = ",".join([df.iloc[i, 0], df.iloc[i, 1]])
            date_time = datetime.strftime(raw_date, "%m/%d/%Y,%H:%M:%S")
            posix_date = time.mktime(date_time.timetuple())

            datetime_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=posix_date))
            temp_feat = tf.train.Feature(float_list=tf.train.Int64List(value=df.iloc[i, 2]))
            hum_feat = tf.train.Feature(float_list=tf.train.Int64List(value=df.iloc[i, 3]))

            mapping = {
                'feature0': datetime_feat,
                'feature1': temp_feat,
                'feature2': hum_feat,
            }

            example = tf.train.Example(features=tf.train.Features(feature=mapping))
            writer.write(example.SerializeToString())


def main():
    parser = ArgumentParser()

    parser.add_argument('--output', type=str, default=None, help='Set the path to the file you want to write on')
    parser.add_argument('--input', type=str, default=None, help='Set the path to the file you want to work with')
    parser.add_argument('--normalize', default=False, action='store_true',
                        help='Decide whether to apply the min max normalization or not')

    args = parser.parse_args()


if __name__ == '__main__':
    main()
