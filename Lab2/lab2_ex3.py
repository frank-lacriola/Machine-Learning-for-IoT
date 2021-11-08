from argparse import ArgumentParser
import os
import tensorflow as tf
import numpy as np

"""Here we want to Extract the MFCC from audio signals.
The Mel-frequency cepstrum is a representation of the STFT of a sound, that tries to mimic how the 
membrane in our ear senses the vibrations of sounds. The cepstrum is the result of computing the inverse Fourier 
transform (IFT) of the logarithm of the estimated signal spectrum. The MFCCs are coefficients that composes the 
Mel-frequency cepstrum. They represent phonemes (distinct units of sound) as the shape of the vocal 
tract (which is responsible for sound generation) is manifest in them"""


def extract_mfcc(args):
    byte_string = tf.io.read_file(filename=args.spectrogram_path)
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
        log_mel_spectrogram)[..., :10]
    path_to_mccs = "{}/{}".format(args.work_dir, args.mfccs_filename)
    mfccs_strings = tf.strings.as_string(mfccs)

    final_string = ""
    for t in mfccs_strings:
        for s in t:
            final_string += s

    # print(mfccs_strings)
    # np.save(path_to_mccs, mfccs.numpy())
    tf.io.write_file(filename=path_to_mccs, contents=mfccs_strings)
    print("The size of the MFCCs' file is: ", os.path.getsize(path_to_mccs))

    if args.visualize_mfccs:
        image = tf.transpose(mfccs)  # Transpose the spectrogram to represent time on x-axis
        image = tf.expand_dims(image, -1)  # Add the 'channel' dimension
        image = tf.math.log(image + 1.e-6)  # take the log of the spectrogram for better visualize
        """For a grayscale images, the pixel value is a single number that represents the brightness of the pixel.
         The most common pixel format is the byte image, where this number is stored as an 8-bit integer giving a
         range of possible values from 0 to 255. Typically zero is taken to be black, and 255 is taken to be white.
         Let's apply min max normalization and then multiply per 255"""
        min_ = tf.reduce_min(image)
        max_ = tf.reduce_max(image)
        image = (image - min_) / (max_ - min_)
        image = image * 255  # We multiply because we got range[0,1] values in image
        image = tf.cast(image, tf.uint8)
        string_png_tensor = tf.io.encode_png(image=image)  # Let's convert the int8 tensor to a png image
        tf.io.write_file(filename="{}/{}".format(args.work_dir, args.mfccs_filename + "_image.png"),
                         contents=string_png_tensor)
        print('Your image is in the working dir!')


def main():
    parser = ArgumentParser()

    parser.add_argument('--work_dir', type=str, default=None, help='Set the working directory')
    parser.add_argument('--spectrogram_path', type=str, default=None,
                        help='Set the path to the file you want to process')
    parser.add_argument('--mel_bins', type=int, default=40, help='Set the mel bins')
    parser.add_argument('--lower_freq', type=int, default=20, help='Set the lower freq')
    parser.add_argument('--upper_freq', type=int, default=4000, help='Set the upper freq')
    parser.add_argument('--mfccs_filename', type=str, default=None)
    parser.add_argument('--visualize_mfccs', action='store_true')
    parser.add_argument('--sample_rate', type=int, default=16000)
    args = parser.parse_args()

    extract_mfcc(args)


if __name__ == '__main__':
    main()