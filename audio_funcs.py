import os
from scipy import signal
import numpy as np
from pysndfile import *

class AudioFile(PySndfile):
    def __init__(self, filepath, mode, format=None, channels=None, samplerate=None):
        super(AudioFile, self).__init__(
                    filepath, 
                    mode = mode, 
                    format = format,
                    channels = channels,
                    samplerate = samplerate
                )
        self.filepath = filepath

    def audio_file_info(self): 
        """ Prints audio information """
        print "*************************************************"
        print "File:                    ", os.path.relpath(self.filepath)  
        print "No. channels:            ", self.channels()
        print "Samplerate:              ", self.samplerate()
        print "Format:                  ", self.format()
        print "No. Frames:              ", self.frames()
        print "Encoding string:         ", self.encoding_str()
        print "Major format string:     ", self.major_format_str()
        print "Seekable?:               ", bool(self.seekable())
        print "Errors?:                 ", self.error()
        print "*************************************************"

    def read_grain(start_index, grain_size):
        """
        Read a grain of audio from the file. if grain ends after the end of
        the file, the grain is padded with zeros.
        Audio object seeker is not changed
        """
        position = self.get_seek_position(audio)
        #Read grain
        index = self.seek(start_index, 0)
        if index + grain_size > self.frames():
            grain = self.read_frames(
                self.frames() - index
            )
            grain = np.pad(
                grain, 
                (0, index+grain_size-self.frames()), 
                'constant', 
                constant_values=(0,0)
            )

        else:
            grain = self.read_frames(grain_size)
        self.seek(position, 0)
        return grain

    def get_seek_position():
        """Returns the current seeker position in the file"""
        return self.seek(0, 1)

def gen_window(window_type, window_size, sym=True):
    """
    Generates a window function of given size and type
    Returns a 1D numpy array

    sym: Used in the triangle window generation. When True (default), generates a symmetric window, for use in filter design. When False, generates a periodic window, for use in spectral analysis
    """
    if window_type is "hanning":
        return np.hanning(window_size)
    elif window_type is "hamming":
        return np.hamming(window_size)
    elif window_type is "bartlett":
        return np.bartlett(window_size)
    elif window_type is "blackman":
        return np.blackman(window_size)
    elif window_type is "kaiser":
        return np.kaiser(window_size)
    elif window_type is "triangle":
        return signal.triang(window_size, sym=sym)
    else:
        raise ValueError
