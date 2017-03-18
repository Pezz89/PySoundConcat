import numpy as np

def gen_wave(
    size,
    freq,
    wave_type,
    phase = 0.0,
    amplitude = 1.0,
    samplerate = 44100
):
    """
    Generates a numpy array of given size (seconds) containing a wave of given
    type and frequency at the samplerate specified
    Note: Waves generated are raw and not anti-aliased. For audio signals
    consider using other algorithms
    """

    def sine():
        samples = np.arange(0, size, 1. / samplerate)
        return amplitude * np.sin(2.0*np.pi*freq*samples)

    def square():
        return amplitude * np.sign(sine())

    def triangle():
        samples = np.arange(0, size, 1. / samplerate)
        return amplitude - (2 * np.abs(samples * (2 * freq) % (2*amplitude) - amplitude))

    def sawtooth():
        samples = np.arange(0, size, 1. / samplerate)
        return amplitude - (2 * np.abs((samples * freq) % amplitude - amplitude))

    def reverse_saw():
        samples = np.arange(0, size, 1. / samplerate)
        return amplitude - (2 * np.abs(((samples * freq) % amplitude)))

    options = {
        "sine" : sine,
        "square" : square,
        "tri" : triangle,
        "saw" : sawtooth,
        "rev_saw" : reverse_saw
    }

    return options[wave_type]()
