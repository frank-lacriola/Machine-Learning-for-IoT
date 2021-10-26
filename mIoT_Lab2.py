from scipy.io import wavfile
from scipy import signal
from argparse import ArgumentParser
import numpy as np
import time
import os


def preprocess(args):
    if args.directory is None:
        path = "{}.wav".format(args.name)
    else:
        path = "{}/{}.wav".format(args.directory, args.name)
    rate, audio = wavfile.read(path)
    start_time = time.time()
    audio = signal.resample_poly(audio, 1, args.rate)    # Let's apply poly-phase filtering
    print("--- %s seconds ---" % (time.time() - start_time))

    audio = audio.astype(np.int16)     # Cast the signal to the original datatype (int16)
    wavfile.write("{}.wav".format(args.new_name), args.rate, audio)  # Let's write on disk the new file
    if args.directory is None:
        new_path = "{}.wav".format(args.new_name)
    else:
        new_path = '{}/{}.wav'.format(args.directory, args.new_name)
    print('Size before pre-processing: {} bytes,  Size after pre-processing: {} bytes'
          .format(os.path.getsize(path), os.path.getsize(new_path)))



def main():
    parser = ArgumentParser()

    parser.add_argument('--format', type=str, default='Int16',
                        help='Set the format of the audio track [Int8,Int16,Int32]')
    parser.add_argument('--channels', type=int, default=2, help='Set the number of channels')
    parser.add_argument('--rate', type=int, default=48000, help='Set the rate')
    parser.add_argument('--name', type=str, default=None, help='Set the name of the file you want to process')
    parser.add_argument('--directory', type=str, default=None, help='Set the directory (container) of your audio')
    parser.add_argument('--new_name', type=str, default=None, help='Set the name of the processed file')

    args = parser.parse_args()
    preprocess(args)


if __name__ == '__main__':
    main()