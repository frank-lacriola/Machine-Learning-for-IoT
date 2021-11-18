import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


class SignalGenerator():
    def __init__(self):
        self.keywords = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
        self.rate = 16000
        self.frame_length = 16
        self.step_length = 8
        self.mel_bins = 40
        self.freq_range = [20, 40]
        self.num_mfccs = 10
        self.mfcc = True
        self.word_label_dict = dict(zip(self.keywords, [0, 1, 2, 3, 4, 5, 6, 7]))

    def preprocess(self, padded_file):

        stft = tf.signal.stft(padded_file,
                              frame_length=int(self.rate * self.frame_length / 1000),  # the window lenght in samples
                              frame_step=int(self.rate * self.step_length / 1000),  # the number of samples to step
                              fft_length=int(
                                  self.rate * self.frame_length / 1000))  # size of FFT to apply: the number of bins used
        spectrogram = tf.abs(stft)
        if not self.mfcc:
            return spectrogram
        else:
            linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(self.mel_bins, spectrogram.shape[-1],
                                                                                self.rate, self.freq_range[0],
                                                                                self.freq_range[1])
            mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
            mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :self.num_mfccs]
            return mfccs

    def paths_and_labels_to_dataset(self, audio_paths):
        """Constructs a dataset of audios and labels."""
        labels = tf.convert_to_tensor([self.word_label_dict[p.split('/')[2]] for p in audio_paths])
        labels_ds = tf.data.Dataset.from_tensor_slices(labels).batch(batch_size=32)
        path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)

        audio_ds = path_ds.map(lambda x: path_to_audio(x))
        for e in audio_ds.as_numpy_iterator():
            print(e.shape)
            break
        preprocessed_ds = audio_ds.map(self.preprocess)
        return tf.data.Dataset.zip((preprocessed_ds, labels_ds))

    def path_to_audio(self, path):
        """Reads and decodes an audio file."""
        audio = tf.io.read_file(path[0])
        tf_audio, _ = tf.audio.decode_wav(audio, 1)
        tf_audio = tf.squeeze(tf_audio, axis=1)
        # let's pad where needed
        """if tf_audio.shape[0] != self.rate:
          paddings = tf.constant([[1, 15999 - int(tf_audio.shape[0])]])
          tf_audio = tf.pad(tf_audio, paddings, 'CONSTANT')"""
        return tf_audio


def main():
    filepaths = []
    generator = SignalGenerator()
    for subdir, dirs, files in os.walk('data/mini_speech_commands'):
        for filename in files:
            filepaths.append(subdir + os.sep + filename)
    X_train, X_test = train_test_split(filepaths, test_size=0.1, random_state=1)
    X_train, X_val = train_test_split(X_train, test_size=0.1, random_state=1)  # 0.25 x 0.8 = 0.2
    train_ds = generator.paths_and_labels_to_dataset(X_train)


