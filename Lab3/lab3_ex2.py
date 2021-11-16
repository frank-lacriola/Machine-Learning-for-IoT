import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras


class SignalGenerator():
    def __init__(self, filepaths, args):
        self.keywords=args.keyword_list,
        self.rate=args.sampling_rate,
        self.frame_length=args.stft_frame_length,
        self.step_length=args.stft_step_length,
        self.mel_bins=args.mel_bins,
        self.freq_range=args.mel_spectrum_frequency_range,
        self.num_mfccs=args.num_mfccs,
        self.mfcc=args.mfcc
        self.paths=filepaths

    def preprocess(self, padded_file):
        if not self.mfcc:
            # stft
        else:
            # mfcc

    def make_dataset(self):
        for subdir, dirs, files in os.walk(self.paths):
            for filename in files:
                filepath = subdir + os.sep + filename
                label = subdir or dirs ???????
                self.preprocess()


def main():
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    zip_path = tf.keras.utils.get_file(
        origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')


