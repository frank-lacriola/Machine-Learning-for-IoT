import os
from argparse import ArgumentParser
from subprocess import Popen
import tensorflow as tf
import time
import numpy as np

Popen('sudo sh -c "echo performance >'
 '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
 shell=True).wait()


def mfcc_slow(filepath, args):
    print("Starting stft...")

    audio = tf.io.read_file(filepath)  # Audio in a string format
    tf_audio, rate = tf.audio.decode_wav(contents=audio,
                                         desired_channels=1)  # Convert the string tensor to a float32 tensor
    tf_audio = tf.squeeze(tf_audio, 1)  # Remove dimension of size 1 to the shape of the tensor

    # Let's apply the short-time Fourier transf,it's used to determine the sinusoidal frequency and phase content
    # of local sections of a signal as it changes over time --> stft separately on each shorter segment
    # it returns a 2D tensor made of complex number (magnitude and phase) that we can run std convs on. -->
    # tf.abs to return only the magnitude
    stft_tensor = tf.signal.stft(tf_audio,
                                 frame_length=tf.constant(int(rate.numpy() * args.window_length / 1000)),
                                 # the window lenght in samples
                                 frame_step=tf.constant(int(rate.numpy() * args.step_length / 1000)),
                                 # the number of samples to step: we want to mantain an identity of the
                                 # original audio signal
                                 fft_length=tf.constant(int(
                                     rate.numpy() * args.window_length / 1000)))  # size of FFT to apply: the number of bins used
    # for dividing the window into equal bin, that defines the freq resolution
    # of the window.
    spectrogram = tf.abs(stft_tensor)
    byte_tensor = tf.io.serialize_tensor(spectrogram)  # Let'a transform the 2D tensor in a byte string array
    tf.io.write_file(filename='{}/{}'.format(args.work_dir,
                                             args.stft_filename if args.stft_filename is not None else 'stft_processed'),
                     contents=byte_tensor)
    stft_path = '{}/{}'.format(args.work_dir,
                               args.stft_filename if args.stft_filename is not None else 'stft_processed')
    print('Size before stft: {} bytes,  Size after stft: {} bytes \n \n'
          .format(os.path.getsize(filepath), os.path.getsize(stft_path)))




    print('Starting MFCCs extraction....')
    byte_string = tf.io.read_file(filename=stft_path)
    float_spectrogram = tf.io.parse_tensor(byte_string, out_type=tf.float32)
    # Let's compute the log-scaled spectrogram
    num_spectrogram_bins = float_spectrogram.shape[-1]
    """The forward method returns a weight matrix that can be used to re-weight a Tensor containing num_spectrogram_bins linearly 
            sampled frequency information from [0, sample_rate / 2] into num_mel_bins frequency information from 
            [lower_edge_hertz, upper_edge_hertz] on the mel scale."""
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=args.mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=args.sample_rate,
        lower_edge_hertz=args.lower_freq,
        upper_edge_hertz=args.upper_freq
    )
    """Tensordot (also known as tensor contraction) sums the product of elements from a and b over the indices
     specified by axes."""
    mel_spectrogram = tf.tensordot(
        a=float_spectrogram,
        b=linear_to_mel_weight_matrix,
        axes=1
    )
    mel_spectrogram.set_shape(float_spectrogram.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:])
    )
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # We compute the MFCCs and take the first 10 coefficients.
    # It returns a float tensor of the MFCCs of log_mel_spectrograms
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrogram)[..., :args.mfccs]
    path_to_mccs = "{}/{}".format(args.work_dir, args.mfccs_filename)

    # This operation transforms data in a tf.Tensor into a tf.Tensor of type tf.string containing the data
    # in a binary string format
    mfccs_byte = tf.io.serialize_tensor(mfccs)
    tf.io.write_file(filename=path_to_mccs, contents=mfccs_byte)
    print("The size of the MFCCs' file is: ", os.path.getsize(path_to_mccs))


def main():
    parser = ArgumentParser()

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

    args = parser.parse_args()

    # Let's iterate over the yes_no files
    for subdir, dirs, files in os.walk(r'../yes_no'):
        times_slow = []
        for filename in files:
            filepath = subdir + os.sep + filename

            starting_time = time.time()
            mfcc_slow(filepath, args)
            times_slow.append(time.time() - starting_time)

        avg_slow_time = sum(times_slow)/len(times_slow)
        print("The average time for MFCC_slow is {:.4f}".format(avg_slow_time))


if __name__ == '__main__':
    main()
