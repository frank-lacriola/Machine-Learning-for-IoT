from scipy.io import wavfile
from scipy import signal
from argparse import ArgumentParser
import numpy as np
import time
import os
import tensorflow as tf


def preprocess(args):

    if not args.skip_poly:
        print("Starting poly-phase filtering...")

        rate, audio = wavfile.read(args.orig_path)
        start_time = time.time()
        audio = signal.resample_poly(audio, 1, int(rate/args.poly_rate))    # Let's apply poly-phase filtering
        print("---Time needed to compute poly-phase filtering: %s seconds ---" % (time.time() - start_time))

        audio = audio.astype(np.int16)     # Cast the signal to the original datatype (int16)
        wavfile.write("{}.wav".format(args.preprocess_name), args.poly_rate, audio)  # Let's write on disk the new file

        if args.work_dir is None:
            preprocessed_path = "{}.wav".format(args.new_name)
        else:
            preprocessed_path = '{}/{}.wav'.format(args.work_dir, args.preprocess_name)
        print('Size before pre-processing: {} bytes,  Size after pre-processing: {} bytes'
              .format(os.path.getsize(args.orig_path), os.path.getsize(preprocessed_path)))

    if args.stft:
        print("Starting stft...")

        if args.skip_poly:
            preprocessed_path = '{}/{}.wav'.format(args.work_dir, args.preprocess_name) if args.work_dir is not None \
                else '{}.wav'.format(args.preprocess_name)

        audio = tf.io.read_file(preprocessed_path)  # Audio in a string format
        tf_audio, rate = tf.audio.decode_wav(contents=audio, desired_channels=1)  # Convert the string tensor to a float32 tensor
        tf_audio = tf.squeeze(tf_audio, 1)  # Remove dimension of size 1 to the shape of the tensor

        # Let's apply the short-time Fourier transf,it's used to determine the sinusoidal frequency and phase content
        # of local sections of a signal as it changes over time --> stft separately on each shorter segment
        # it returns a 2D tensor made of complex number (magnitude and phase) that we can run std convs on. -->
        # tf.abs to return only the magnitude
        start_time = time.time()
        stft_tensor = tf.signal.stft(tf_audio,
                   frame_length=tf.constant(int(rate.numpy() * args.window_length / 1000)),  # the window lenght in samples
                   frame_step=tf.constant(int(rate.numpy() * args.step_length / 1000)),  # the number of samples to step: we want to mantain an identity of the
                                                # original audio signal
                   fft_length=tf.constant(int(rate.numpy() * args.window_length / 1000)))  # size of FFT to apply: the number of bins used
                                                  # for dividing the window into equal bin, that defines the freq resolution
                                                  # of the window.
        spectrogram = tf.abs(stft_tensor)
        print("---Time needed to compute the spectogram: %s seconds ---" % (time.time() - start_time))
        byte_tensor = tf.io.serialize_tensor(spectrogram)  # Let'a transform the 2D tensor in a byte string array
        #print(byte_tensor)
        tf.io.write_file(filename=args.stft_filename if args.stft_filename is not None else 'stft_processed',
                         contents=byte_tensor)
        stft_path = '{}/{}'.format(args.work_dir,
                                   args.stft_filename if args.stft_filename is not None else 'stft_processed')
        print('Size before stft: {} bytes,  Size after stft: {} bytes'
              .format(os.path.getsize(preprocessed_path), os.path.getsize(stft_path)))


def main():
    parser = ArgumentParser()

    parser.add_argument('--format', type=str, default='Int16',
                        help='Set the format of the audio track [Int8,Int16,Int32]')
    parser.add_argument('--channels', type=int, default=2, help='Set the number of channels')
    parser.add_argument('--poly_rate', type=int, default=48000, help='Set the rate')
    parser.add_argument('--work_dir', type=str, default=None, help='Set the working directory')
    parser.add_argument('--orig_path', type=str, default=None, help='Set the path to the file you want to process')
    parser.add_argument('--preprocess_name', type=str, default=None, help='Set the name of the processed file')
    parser.add_argument('--skip_poly', default=False, action='store_true')
    parser.add_argument('--stft', default=False, action='store_true',
                        help='Decide whether to apply the stft or not')
    parser.add_argument('--stft_filename', type=str, default=None, help='Set the filename for the spectrum''s file')
    parser.add_argument('--window_length', type=float, default=40, help='Set the window length in milli-seconds')
    parser.add_argument('--step_length', type=float, default=20, help='Set the step length in milli-seconds')

    args = parser.parse_args()
    preprocess(args)


if __name__ == '__main__':
    main()