import os
from scipy import signal
import numpy as np
from pysndfile import *

class AudioFile(PySndfile):
    """Object for storing and accessing basic information for an audio file"""
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

class AnalysedAudioFile(AudioFile):
    """Generates and stores analysis information for an audio file"""
    def __init__(self, filepath, mode, format=None, channels=None, samplerate=None):
        super(AnalysedAudioFile, self).__init__(
                    filepath, 
                    mode = mode, 
                    format = format,
                    channels = channels,
                    samplerate = samplerate
                )

    def create_rms_analysis(self, window_size=100, window_type="triangle"):
        """Generate an energy contour analysis by calculating the RMS values of windows segments of the audio file"""
        window_size = self.ms_to_samps(window_size) 
        window_function = self.gen_window(window_type, window_size) 
        i = 0
        while i < self.frames():
            #Read frames from audio file
            frames = self.read_grain(i, window_size)
            #Apply window function to frames
            frames = frames * window_function
            #Calculate the RMS value of the current window of frames
            rms = np.sqrt(np.mean(frames*frames))

    def init_database(location):
        """Creates the folder hierachy for the database of files to be stored in"""
        #Check if database already exists in the location
        #if it does then great
        #else create a new one
        #create main directory with the name of the audio file (minus extension)
        #create sub-directories for:
        #the wav file (.wav)
        #the rms file (.lab)

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
    
    def ms_to_samps(self, ms):
        """Converts milliseconds to samples based on the sample rate of the audio file"""
        seconds = ms / 1000.0
        return int(round(seconds*self.samplerate()))


