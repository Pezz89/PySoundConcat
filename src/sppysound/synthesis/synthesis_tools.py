from __future__ import print_function, division
import numpy as np
from sppysound import AudioFile
import matplotlib.pyplot as plt
import pdb
import scipy


def convolve(input, impulse_response):
    out = np.zeros(len(input) + len(impulse_response) - 1)
    for input_ind, i in enumerate(input):
        for imp_ind, j in enumerate(impulse_response):
            out[input_ind+imp_ind] = out[input_ind+imp_ind] + i*j
    return out

def moving_average_filter_recursive(input, M, symetry = 'after'):
    '''
    Applies a moving average filter to the input.

    Arguments:
        input - the input signal to filter.
        symetry - ('before' or 'middle') defines how points will be
        averaged around the index
        M - the number of coefficients.
    '''
    # Calculate the filter coefficients
    filter_kernal = np.ones(M) / M
    # Get the pre-zero-padded input size.
    input_size = input.size
    # Pad end of input with zeros.
    if symetry == 'after':
        # Zero-pad input at end on input for averaging of end samples
        input = np.hstack((input, np.zeros(M)))
    elif symetry == 'middle':
        # M value must be odd to have an equal number of samples on each side.
        if not M % 2:
            raise ValueError("M must be odd for symetrical averaging")
        # Calculate the zero padding size.
        offset = np.floor(M/2.0)
        # Zero pad input on both sides to allow for averaging from first sample
        # to last sample
        input = np.hstack((np.zeros(offset), input, np.zeros(offset)))


    # Calculate the number of output samples.
    # y = np.zeros(input.size-M)

    y = np.zeros(input.size-M)
    # If averaging after first sample.
    if symetry == 'after':
        # For each sample in the input
        acc = 0

        i = 0
        while i < M:
            acc += input[i]
            i += 1
        y[0] = acc / M

        i = 1
        while i < input.size-M:
            acc += input[i+M-1] - input[i-1]
            y[i] = acc/M
            i += 1
        print(y)

    elif symetry == 'middle':
        # TODO: Make recursive
        i = 0
        # For all the input samples
        while i < input_size-offset:
            # The output sample is the average sample value for M samples.
            y[i] = np.sum(input[i:i+M] * filter_kernal)
            i += 1
    return y

def moving_average_filter(input, M, symetry = 'after'):
    '''
    Applies a moving average filter to the input.

    Arguments:
        input - the input signal to filter.
        symetry - ('before' or 'middle') defines how points will be
        averaged around the index
        M - the number of coefficients.
    '''
    # Calculate the filter coefficients
    filter_kernal = np.ones(M) / M
    # Get the pre-zero-padded input size.
    input_size = input.size
    # Pad end of input with zeros.
    if symetry == 'after':
        # Zero-pad input at end on input for averaging of end samples
        input = np.hstack((input, np.zeros(M)))
    elif symetry == 'middle':
        # M value must be odd to have an equal number of samples on each side.
        if not M % 2:
            raise ValueError("M must be odd for symetrical averaging")
        # Calculate the zero padding size.
        offset = np.floor(M/2.0)
        # Zero pad input on both sides to allow for averaging from first sample
        # to last sample
        input = np.hstack((np.zeros(offset), input, np.zeros(offset)))


    # Calculate the number of output samples.
    y = np.zeros(input.size-M)

    # If averaging after first sample.
    if symetry == 'after':
        i = 0
        # For each sample in the input
        while i < input_size:
            y[i] = np.sum(input[i:i+M] / M)
            i += 1
    # If averaging symetrically
    elif symetry == 'middle':
        i = 0
        # For all the input samples
        while i < input_size-offset:
            # The output sample is the average sample value for M samples.
            y[i] = np.sum(input[i:i+M] / M)
            i += 1
    return y

def blackman_filter(input, window_size, freq):
    '''
    Create a blackman windowed-sinc filter.

    freq - The cutoff frequency of the filter specified as a proportion of the
    samplerate of the signal.
    '''
    # TODO: Check the definition of freq is correct.

    i = np.arange(window_size)
    # Create a sinc function of M length.
    # The output will be a sinc function shifted from -M/2 - M/2 to 0 - M.
    # This will result in a sinc function that can be used to create a filter
    # at the cutoff-frequency provided in freq.
    sinc_kernal = np.sin(2*np.pi*freq*(i-window_size/2))/(i-window_size/2)

    # Create a blackman window
    window = 0.42 - 0.5 * np.cos(2 * np.pi * (i / window_size)) + 0.08 * np.cos(4 * np.pi * (i / window_size))
    window_sinc = sinc_kernal * window

    # Number of samplepoints
    N = window_size
    # sample spacing
    T = 1.0 / 800.0
    yf = scipy.fftpack.fft(window_sinc)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)


    plt.subplot(311)
    plt.title('Blackman Window')
    plt.plot(window)
    plt.ylabel('Amplitude')
    plt.xlabel('sample')
    plt.subplot(312)
    plt.title('Window sinc function')
    plt.plot(sinc_kernal)
    plt.subplot(313)
    plt.title('FFT')
    plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
    plt.show()

if __name__ == "__main__":
    '''
    a = np.array([1, 0.5, 3, 1])
    b = np.array([1, 0, 0, 0])
    c = convolve(a, b)
    print(c)
    print(np.convolve(a, b))
    '''
    with AudioFile('./test_audio.aif', 'r') as test_audio:
        grain = test_audio.read_grain(0, -1)
    grain = np.arange(5000)
    filtered_grain = moving_average_filter(grain, 101)
    filtered_r_grain = moving_average_filter_recursive(grain, 101)

    blackman_filter(grain, 101, 0.14)

    '''
    # Plot test wave
    plt.subplot(211)
    plt.title('Original Wave')
    plt.plot(grain)
    plt.ylabel('Amplitude')
    plt.xlabel('sample')
    plt.subplot(212)
    plt.title('Filtered Wave')
    plt.plot(filtered_grain)
    plt.ylabel('Amplitude')
    plt.xlabel('sample')
    plt.show()
    '''
