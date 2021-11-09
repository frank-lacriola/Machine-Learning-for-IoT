import argparse
import os
import tensorflow as tf
import time
from scipy.io import wavfile
import numpy as np
from subprocess import Popen

Popen('sudo sh -c "echo performance >''/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"', shell=True).wait()


def mfcc(filepath, args):

    Popen('sudo sh -c "echo performance >' '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
          shell=True).wait()

    print("Starting slow preprocessing...")
    a = time.time()

    input_rate, audio = wavfile.read(filepath)
    tf_audio = tf.convert_to_tensor(audio, dtype=tf.float32)

    stft = tf.signal.stft(tf_audio, 16, 8)

    spectrogram = tf.abs(stft)
    print('Spectrogram shape:', spectrogram.shape)

    num_spectrogram_bins = spectrogram.shape[-1]
    num_mel_bins = 40
    sampling_rate = 16000

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins,
                                                                        sampling_rate, 20, 4000)
    print("linear to mel matrix shape:", linear_to_mel_weight_matrix.shape)

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfccs_slow = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :10]
    conversion = tf.io.serialize_tensor(mfccs_slow)

    b = time.time()
    print('MFCCs shape:', mfccs_slow.shape)

    print('Execution Time for slow preprocessing is: {:.3f}s'.format(b - a))



    print("Starting fast preprocessing...")

    stft = tf.signal.stft(tf_audio, 16, 8)

    spectrogram = tf.abs(stft)
    print('Spectrogram shape:', spectrogram.shape)

    num_spectrogram_bins = spectrogram.shape[-1]
    num_mel_bins = 30
    sampling_rate = 16000

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins,
                                                                        sampling_rate, 20, 4000)
    print("linear to mel matrix shape:", linear_to_mel_weight_matrix.shape)

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfccs_fast = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :10]
    conversion = tf.io.serialize_tensor(mfccs_fast)

    b = time.time()
    print('MFCCs shape:', mfccs_fast.shape)

    print('Execution Time: {:.3f}s'.format(b - a))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--work_dir', type=str, default=None, help='Set the working directory')
    parser.add_argument('--orig_path', type=str, default=None, help='Set the path to the file you want to process')
    parser.add_argument('--stft', default=False, action='store_true',
                        help='Decide whether to apply the stft or not')
    parser.add_argument('--stft_filename', type=str, default=None, help='Set the filename for the spectrum''s file')
    parser.add_argument('--window_length', type=float, default=16, help='Set the window length in milli-seconds')
    parser.add_argument('--step_length', type=float, default=8, help='Set the step length in milli-seconds')
    parser.add_argument('--work_dir', type=str, default=None, help='Set the working directory')
    parser.add_argument('--spectrogram_path', type=str, default=None,
                        help='Set the path to the file you want to process')
    parser.add_argument('--mel_bins', type=int, default=40, help='Set the mel bins')
    parser.add_argument('--lower_freq', type=int, default=20, help='Set the lower freq')
    parser.add_argument('--upper_freq', type=int, default=4000, help='Set the upper freq')
    parser.add_argument('--mfccs', type=int, default=10,
                        help='Set the number of coefficients that make up the mel-freq cepstrum')
    parser.add_argument('--mfccs_filename', type=str, default=None)
    parser.add_argument('--sample_rate', type=int, default=16000)

    args_slow = parser.parse_args()

    # Let's iterate over the yes_no files
    for subdir, dirs, files in os.walk(r'../yes_no'):
        times_slow = []
        for filename in files:
            filepath = subdir + os.sep + filename

            starting_time = time.time()
            mfcc(filepath, args_slow)
            times_slow.append(time.time() - starting_time)

        avg_slow_time = sum(times_slow)/len(times_slow)
        print("The average time for MFCC_slow is {:.4f}".format(avg_slow_time))


if __name__ == '__main__':
    main()
