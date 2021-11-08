import pandas as pd
import numpy as np

import time
from datetime import datetime
import os

from argparse import ArgumentParser

import tensorflow as tf


def normalize(column):
    max_temp = column.max()
    min_temp = column.min()
    column = (column - min_temp) / (max_temp - min_temp)
    
    return column


def print_statistics(output_path, input_path, start, end):

    print(f'Total time : {(end - start):.4f}\n')

    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    percentage_size = 100.0 * ( (output_size - input_size) / input_size )

    print(f'Initial size : {input_size}')
    print(f'Final size : {output_size} ( {percentage_size:.2f} )' )


def make_dataset(args):

    df = pd.read_csv(args.input_path)

    with tf.io.TFRecordWriter(args.output_path) as writer:
        for i in range(df.iloc[:,0].size):
            
            raw_date = ",".join([df.iloc[i,0],df.iloc[i,1]])
            date = datetime.strptime(raw_date, '%d/%m/%Y,%H:%M:%S')
            posix_date = time.mktime(date.timetuple())
            
            if args.normalize:
                temp_normalized = normalize(df.iloc[:, 2])
                hum_normalized = normalize(df.iloc[:, 3])
            
            datetime_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(posix_date)])) 
            temperature = tf.train.Feature(int64_list=tf.train.Int64List(value=[temp_normalized[i].astype(np.int64)]))
            humidity = tf.train.Feature(int64_list=tf.train.Int64List(value=[hum_normalized[i].astype(np.int64)]))

            mapping = {
                'datetime': datetime_feature,
                'temperature': temperature,
                'humidity': humidity,
            }
                    
            example = tf.train.Example(features=tf.train.Features(feature=mapping))
            writer.write(example.SerializeToString())


def main():

    parser = ArgumentParser()

    parser.add_argument('--input_path', type=str, default="file.csv", help='Set the input path')
    parser.add_argument('--output_path',type=str, default="dataset_output.tfrecord", help='Set the output path')
    parser.add_argument('--normalize', type=bool, store=True, help='Normalize the data')

    args = parser.parse_args()

    start_timer = time.time()

    make_dataset(args)

    end_timer = time.time()

    print_statistics(args.output_path, args.input_path, start_timer, end_timer)


if __name__ == '__main__':

    main()