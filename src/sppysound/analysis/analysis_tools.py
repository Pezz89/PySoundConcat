from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pdb

def my_first_DFT(input, window_size=4096):
    # Create arrays at half the size of the sample provided.
    # These will store the real and imaginary values as amplitudes of sine and
    # cosine waves for each frequency bin
    real_values = np.zeros(window_size/2)
    imaginary_values = np.zeros(window_size/2)

    # Create vectors for each index of the input and the output arrays.

    i = np.arange(window_size)
    k = np.arange(window_size/2)

    # For each index in the input array...
    for ind in i:
        # For each index in the output arrays...
        for ind2 in k:
            # The real value is an accumulation of the correlations between the
            # input sample and each 'basis function' (cosines at the frequency of the
            # current bin).
            real_values[ind2] += input[ind] * np.cos(2*np.pi*ind2*ind/window_size)
            # Imaginary values are similar to real values, but use sine waves
            # as the basis functions as oppsoed to cosine waves.
            imaginary_values[ind2] -= input[ind] * np.sin(2*np.pi*ind2*ind/window_size)

    # Values are returned as arrays of amplitudes between 0.0 and 1.0 for the
    # waves of each frequency bin.
    '''
    real_values = real_values / (window_size / 2+1)
    imaginary_values = imaginary_values / (window_size / 2+1)
    '''

    return (real_values, imaginary_values)

def my_first_fft(input, window_size=2048):
    X = np.array(input, dtype=complex)
    N = window_size

    # M is the base 2 logarithm of the window size.
    # M represents the number of stage required in the decomposition.
    M = int(np.log2(window_size))

    # For every index from index 1 up to the length of the input -2...
    i = np.arange(window_size-2)
    i += 1
    J = window_size/2
    for ind in i:
        # If the index is less than half the total number of input samples...
        if ind <= J:
            # Swap values at current index and index half the input value
            X[ind], X[J] = X[J], X[ind]
        # K is the length of the input divided by 2
        K = window_size/2
        # While K <= J...
        while K <= J:
            # J is it's current value - K's current value
            J -= K
            # Divide K by 2
            K /= 2
        #J is it's current value + K's current value
        J += K

    # For each stage of combining the N frequency spectra
    L = 1
    while L < M:
        LE = int(2**L)
        LE2 = LE/2
        U = 1+0j
        # Calculate the index to the power of 2
        # Divide the result by 2
        # Calculate the cosine using the index bin number
        S=np.cos(np.pi/float(LE2))-np.sin(np.pi/float(LE2))*1j

        # for each sub DFT...
        Jind = np.arange(LE2)
        for J in Jind:
            I = J
            while I < N-1:
                print(X)
                IP = I+LE2
                # Butterfly calculation
                T = X[I] + X[IP]
                X[IP] = (X[I] - X[IP]) * U
                X[I] = T
                I += LE
            U = U * S
        L+=1
        pdb.set_trace()
        return X



def window_input(input):

    window = np.hamming(window_size)
    return input*window

def my_first_IDFT(real_input, imaginary_input, window_size=4096):
    # Create an array to store accumulated values of sine and cosine waves
    # generated
    out = np.zeros(window_size)

    # Create vectors for each index of the input and the output arrays.
    i = np.arange(window_size)
    k = np.arange(window_size/2)

    # For each index in the output arrays...
    for ind in i:
        # For each index in the input arrays...
        for ind2 in k:
            # The output is equal to the accumulated values of sines and
            # cosines at the frequency of the current bin multiplied by the
            # amplitude of the current bin.
            out[ind] += (real_input[ind2] * np.cos(2*np.pi*ind2*ind/window_size) +
                         imaginary_input[ind2] * np.sin(2*np.pi*ind2*ind/window_size))
    return out

def rect_to_pol(real, imaginary):
    magnitude = np.sqrt(np.power(real, 2) + np.power(imaginary, 2))
    # calculate the phases through the equation: arctan(x/y)
    # the arctan2 function is used to account for the divide by zero problem
    # aswell as the incorrect arctan problem
    # encountered when using arctan. see: http://www.dspguide.com/ch8/9.htm
    # (Nuisnce 2, 3) for more info.
    phase = np.arctan2(imaginary, real)


    return magnitude, phase

def unwrap_phase(phase):

    # Create an array to store unwrapped phase values
    unwrapped_phase = np.zeros(phase.size)
    i = 1
    while i < phase.size:
        # Calculate the offset of 2*PI to use for value based on previous
        # value.
        offset = int((unwrapped_phase[i-1] - phase[i])/(2*np.pi))
        # Offset phase to unwrap
        unwrapped_phase[i] = phase[i] + offset*2*np.pi
        i += 1
    return unwrapped_phase


def pol_to_rect(magnitude, phase):
    real = magnitude * np.cos(phase)
    imaginary = magnitude * np.sin(phase)
    return real, imaginary

def amp_to_dB(a):
    return 20 * np.log10(a / 1.0)

if __name__ == "__main__":
    # Create a signal containing 2 cosine waves at 3hz and 9hz, and a single
    # sine wave at 5 hz

    input = np.arange(16)
    my_first_fft(input, window_size=16)

    samples = np.arange(512)
    freq = np.linspace(3, 10, samples.size)
    amp = np.linspace(1, 0, samples.size)
    test_wave = np.sin(2*np.pi*3*samples/samples.size)
    """
    test_wave += np.cos(2*np.pi*9*samples/samples.size)
    test_wave += np.sin(2*np.pi*5*samples/samples.size)
    test_wave += np.cos(2*np.pi*230*samples/samples.size)
    #test_wave = amp * test_wave
    #test_wave = np.sin(2*np.pi*freq*samples/samples.size)
    plt.subplot(311)
    plt.title('Wave Frequency Ramp')
    plt.plot(freq)
    plt.subplot(312)
    plt.title('Amplitude Ramp')
    plt.plot(amp)
    plt.subplot(313)
    plt.title('Test Wave')
    plt.ylabel('Amplitude')
    plt.xlabel('sample')
    plt.plot(test_wave)
    plt.show()


    # Plot test wave
    plt.subplot(311)
    plt.title('Test Wave')
    plt.plot(test_wave)
    plt.ylabel('Amplitude')
    plt.xlabel('sample')
    plt.show()
    """

    # Generate real and imaginary values by performing a DFT analysis. Store
    # these values as 2 arrays with size == samples/2
    # output = my_first_DFT(test_wave, window_size=samples.size)

    # impulse = my_first_IDFT(np.hstack((np.zeros(110), [1], np.zeros(145))), np.zeros(256), window_size=512)
    impulse = np.hstack((np.zeros(3), [1], np.zeros(60)))
    windowed_impulse = impulse*np.hamming(64)

    '''
    window_dft = my_first_DFT(windowed_impulse, window_size=512)
    polar_values = rect_to_pol(window_dft[0], window_dft[1])
    '''
    '''
    # Check phase and amplitude of an impulse
    dft = my_first_DFT(impulse, window_size=64)
    polar_values = rect_to_pol(dft[0], dft[1])

    amp_to_dB(polar_values[0])
    plt.subplot(411)
    plt.title('Magnitude')
    plt.plot(polar_values[0])
    plt.subplot(412)
    plt.title('Phase')
    plt.plot(polar_values[1])
    plt.subplot(413)
    plt.title('Real Part')
    plt.plot(dft[0])
    plt.subplot(414)
    plt.title('Imaginary Part')
    plt.plot(dft[1])
    plt.show()
    '''

    # Check phase and amplitude of an sinc impulse
    impulse = np.hstack((np.zeros(3), [1], np.zeros(508)))
    sinc_impulse = np.hstack((np.ones(32), np.zeros(449), np.ones(31)))
    dft = my_first_DFT(sinc_impulse, window_size=512)

    # amp_to_dB(polar_values[0])
    plt.subplot(413)
    plt.title('Real Part')
    plt.plot(dft[0])
    plt.subplot(414)
    plt.title('Imaginary Part')
    plt.plot(dft[1])
    polar_values = rect_to_pol(dft[0], dft[1])
    plt.subplot(411)
    plt.title('Magnitude')
    plt.plot(polar_values[0])
    plt.subplot(412)
    plt.title('Phase')
    plt.plot(unwrap_phase(polar_values[1]))
    print(polar_values[1])
    plt.show()
    exit()

    window_dft = my_first_DFT(impulse, window_size=2048)
    polar_values = rect_to_pol(window_dft[0], window_dft[1])
    plt.subplot(212)
    plt.plot(polar_values[0])
    plt.show()
    exit()

    # Plot output arrays to show
    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    plt.subplot(312)
    plt.title('Real values')
    plt.axis((0, output[0].size, -1.2, 1.2))
    plt.plot(output[0], color = 'r', linestyle='-', marker = 'x')
    plt.subplot(313)
    plt.title('Imaginary values')
    plt.axis((0, output[1].size, -1.2, 1.2))
    plt.plot(output[1], color = 'r', linestyle='-', marker = 'x')
    plt.show()

    polar_values = rect_to_pol(output[0], output[1])
    print(polar_values[0].size)
    unwrapped_phase = unwrap_phase(polar_values[1])
    unwrapped_phase = np.unwrap(polar_values[1])
    plt.subplot(211)
    plt.axis((0, output[1].size, -np.pi, np.pi))
    plt.plot(output[1])
    plt.title("Wrapped Phase")
    plt.subplot(212)
    plt.axis((0, unwrapped_phase.size, -np.pi, np.pi))
    plt.plot(unwrapped_phase)
    plt.title("Unwrapped Phase")
    plt.show()


    # Perform inverse DFT to produce the output in the time domain.
    # This output should be the same as the input.
    out = my_first_IDFT(output[0], output[1], window_size=samples.size)

    # Plot the output against the input to check they are the same.
    plt.subplot(211)
    plt.title('Output wave')
    plt.plot(out)
    plt.subplot(212)
    plt.title('Original wave')
    plt.plot(test_wave)
    plt.show()
