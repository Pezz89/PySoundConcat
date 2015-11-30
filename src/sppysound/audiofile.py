from __future__ import print_function, division
import os
import shutil
import collections
from scipy import signal
import numpy as np
import pysndfile
import matplotlib.pyplot as plt
import pdb
import sys
import traceback
import logging
import h5py

from fileops import pathops
import analysis.RMSAnalysis as RMSAnalysis
import analysis.AttackAnalysis as AttackAnalysis
import analysis.ZeroXAnalysis as ZeroXAnalysis
import analysis.FFTAnalysis as FFTAnalysis
import analysis.SpectralCentroidAnalysis as SpectralCentroidAnalysis
import analysis.SpectralSpreadAnalysis as SpectralSpreadAnalysis

logger = logging.getLogger(__name__).addHandler(logging.NullHandler())


class AudioFile(object):

    """Object for storing and accessing basic information for an audio file."""

    def __init__(self, wavpath, mode,
                 format=None,
                 channels=None,
                 samplerate=None,
                 name=None, *args, **kwargs):
        self.logger = logging.getLogger(__name__ + '.AudioFile')
        self.logger.info("Initialised AudioFile")

        self.wavpath = wavpath
        # TODO: If a name isn't provided then create a default name based on
        # the file name without an extension
        self.name = name
        self.mode = mode
        self.samplerate = samplerate
        self.format = format
        self.channels = channels

    def __enter__(self):
        """Allow AudioFile object to be opened by 'with' statements"""
        self.logger.info("Opening soundfile {0}".format(self.wavpath))
        if self.mode == 'r':
            if not os.path.exists(self.wavpath):
                raise IOError(
                    "Cannot open {0} for reading as it cannot be "
                    "found.".format(self.wavpath)
                )
            self.pysndfile_object = pysndfile.PySndfile(
                self.wavpath,
                mode=self.mode
            )
            self.samplerate = self.get_samplerate()
            self.format = self.get_format()
            self.channels = self.get_channels()
            return self
        else:
            self.pysndfile_object = pysndfile.PySndfile(
                self.wavpath,
                mode=self.mode,
                format=self.format,
                channels=self.channels,
                samplerate=self.samplerate
            )
            return self

    def open(self):
        self.logger.info("Opening soundfile {0}".format(self.wavpath))
        return self.__enter__()

    def close(self):
        self.logger.info("Closing soundfile {0}".format(self.wavpath))
        self.pysndfile_object = None

    def __exit__(self, type, value, traceback):
        """Closes sound file when exiting 'with' statement."""
        self.logger.info("Closing soundfile {0}".format(self.wavpath))
        self.pysndfile_object = None

    def __if_open(method):
        """Handles error from using methods when the audio file is closed"""
        def wrapper(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except AttributeError, err:
                raise IOError("{0}: Audio file isn't open.".format(err),
                              sys.exc_info()[2])

        return wrapper

    @__if_open
    def get_channels(self):
        """Return number of channels of sndfile."""
        return self.pysndfile_object.channels()

    @__if_open
    def encoding_str(self):
        """
        Return string representation of encoding (e.g. pcm16).

        See pysndfile.get_sndfile_encodings() for a list of available
        encoding strings that are supported by a given sndfile format
        """
        return self.pysndfile_object.encoding_str()

    @__if_open
    def error(self):
        """Report error numbers related to the current sound file."""
        return self.pysndfile_object.error()

    @__if_open
    def get_format(self):
        """Return raw format specification from sndfile."""
        return self.pysndfile_object.format()

    @__if_open
    def frames(self):
        """Return number of frames in file (number of samples per channel)."""
        return self.pysndfile_object.frames()

    @__if_open
    def get_strings(self):
        """
        get all stringtypes from the sound file.

        see stringtype_name_top_id.keys() for the list of strings that
        are supported by the libsndfile version you use.
        """
        return self.pysndfile_object.get_strings()

    @__if_open
    def major_format_str(self):
        """
        return short string representation of major format.

        (e.g. aiff) see pysndfile.get_sndfile_formats() for a complete
        lst of fileformats
        """
        return self.pysndfile_object.major_format_str()

    @__if_open
    def read_frames(self, nframes=-1, dtype=np.float64):
        """
        Read the given number of frames and fill numpy array.

        Parameters
        nframes: <int>
        number of frames to read (default = -1 -> read all).
        dtype: <numpy dtype>
        dtype of the returned array containing read data (see note).

        Notes
        One column per channel.
        """
        return self.pysndfile_object.read_frames(nframes, dtype)

    @__if_open
    def rewind(self, mode='rw'):
        """
        Rewind read/write/read and write position given by mode to
        start of file.
        """
        return self.pysndfile_object.format(mode)

    @__if_open
    def get_samplerate(self):
        """Return the samplerate of the file."""
        return self.pysndfile_object.samplerate()

    @__if_open
    def seek(self, offset, whence=0, mode='rw'):
        """
        Seek into audio file: similar to python seek function,
        but taking only audio data into account.

        Parameters
        offset: <int>
        the number of frames (eg two samples for stereo files) to move
        relatively to position set by whence.
        whence: <int>
        only 0 (beginning), 1 (current) and 2 (end of the file) are valid.
        mode: <string>
        If set to 'rw', both read and write pointers are updated. If 'r'
        is given, only read pointer is updated, if 'w', only the write one
        is (this may of course make sense only if you open the file in a
        certain mode).

        Returns
        offset: int the number of frames from the beginning of the file

        Notes
        Offset relative to audio data: meta-data are ignored.
        if an invalid seek is given (beyond or before the file), an IOError
        is raised; note that this is different from the seek method of a
        File object.
        """
        return self.pysndfile_object.seek(offset, whence, mode)

    @__if_open
    def seekable(self):
        """Return true for soundfiles that support seeking."""
        return self.seekable()

    @__if_open
    def set_auto_clipping(self, arg=True):
        """
        Enable auto clipping when reading/writing samples from/to sndfile.

        auto clipping is enabled by default. auto clipping is required by
        libsndfile to properly handle scaling between sndfiles with pcm
        encoding and float representation of the samples in numpy. When auto
        clipping is set to on reading pcm data into a float vector and writing
        it back with libsndfile will reproduce the original samples. If auto
        clipping is off, samples will be changed slightly as soon as the
        amplitude is close to the sample range because libsndfile applies
        slightly different scaling factors during read and write.
        """
        return self.pysndfile_object.set_auto_clipping(arg)

    @__if_open
    def set_string(self, stringtype_name, string):
        """
        Set one of the stringtypes to the strig given as argument.

        If you try to write a stringtype that is not supported byty the library
        a RuntimeError will be raised
        """
        return self.pysndfile_object.set_string(stringtype_name, string)

    @__if_open
    def strError(self):
        """Report error strings related to the current sound file."""
        return self.pysndfile_object.strError()

    @__if_open
    def writeSync(self):
        """
        Call the operating system's function to force the writing of all file
        cache buffers to disk.
        No effect if file is open as read
        """
        return self.pysndfile_object.writeSync()

    @__if_open
    def write_frames(self, input):
        """
        write 1 or 2 dimensional array into sndfile.

        Parameters
        input: <numpy array>
        containing data to write.
        Notes
        One column per channel.
        updates the write pointer.
        if the input type is float, and the file encoding is an integer type,
        you should make sure the input data are normalized normalized data
        (that is in the range [-1..1] - which will corresponds to the maximum
        range allowed by the integer bitwidth).
        """
        return self.pysndfile_object.write_frames(input)

    @__if_open
    def construct_format(self, major, encoding):
        """
        construct a format specification for libsndfile from major format
        string and encoding string
        """
        return self.pysndfile_object.construct_format(major, encoding)

    @__if_open
    def get_pysndfile_version(self):
        """return tuple describing the version of pysndfile"""
        return self.pysndfile_object.get_pysndfile_version()

    @__if_open
    def get_sndfile_version(self):
        """
        return a tuple of ints representing the version of the libsdnfile that
        is used
        """
        return self.pysndfile_object.get_sndfile_version()

    @__if_open
    def get_sndfile_formats(self):
        """
        Return lists of available file formats supported by libsndfile and
        pysndfile.
        """
        return self.pysndfile_object.get_sndfile_formats()

    @__if_open
    def get_sndfile_encodings(self, major):
        """
        Return lists of available encoding for the given sndfile format.

        Parameters
        major sndfile format for that the list of available fomramst should
        be returned. format should be specified as a string, using one of the
        straings returned by get_sndfile_formats()
        """
        return self.pysndfile_object.get_sndfile_encodings(major)

    def __iter__(self):
        """
        Allows the AudioFile object to be iterated over
        Each iteration returns a chunk of audio.
        Audio chunk size is based on the self.chunksize member
        """

    def audio_file_info(self):
        """ Prints audio information """
        print('*************************************************')
        print('File:                    ', os.path.relpath(self.wavpath))
        print('No. channels:            ', self.channels())
        print('Samplerate:              ', self.samplerate())
        print('Format:                  ', self.format())
        print('No. Frames:              ', self.frames())
        print('Encoding string:         ', self.encoding_str())
        print('Major format string:     ', self.major_format_str())
        print('Seekable?:               ', bool(self.seekable()))
        print('Errors?:                 ', self.error())
        print('*************************************************')

    @__if_open
    def read_grain(self, start_index=0, grain_size=None, padding=True):
        """
        Read a grain of audio from the file. if grain ends after the end of
        the file, the grain can be padded with zeros using the padding
        argument.
        Audio object seeker is not changed
        """
        self.switch_mode('r')
        if start_index < 0:
            start_index = self.frames() + start_index
        if not grain_size:
            grain_size = self.frames()
        position = self.get_seek_position()
        # Read grain
        index = self.pysndfile_object.seek(start_index, 0)
        if index + grain_size > self.frames():
            grain = self.read_frames(self.frames() - index)
            if padding:
                grain = np.pad(
                    grain,
                    (0, index + grain_size - self.frames()),
                    'constant',
                    constant_values=(0, 0)
                )
        else:
            grain = self.read_frames(grain_size)
        self.seek(position, 0)
        return grain

    def normalize_file(self, overwrite_original=False):
        """Normalizes the entire file"""
        # Get current file name and it's extension
        (current_filename, current_fileextension) = (
            os.path.splitext(self.wavpath)
        )
        # Create a seperate filepath to use for the mono file to be created
        normalized_filename = ''.join(
            (current_filename, ".norm", current_fileextension)
        )
        # If the mono audio file already exists then use that to replace the
        # stereo file, rather than computing again from scratch
        if os.path.exists(normalized_filename):
            self.replace_audiofile(normalized_filename)
            return None
        # Create the empty mono file object
        normalized_file = AudioFile(
            normalized_filename,
            mode='w',
            format=self.format,
            channels=1,
            samplerate=self.samplerate
        ).open()
        # Read current file in chunks and convert to mono by deviding all
        # samples by 2 and combining to create a single signal
        self.seek(0, 0)
        self.switch_mode('r')
        samples = self.pysndfile_object.read_frames()
        self.normalize_audio(samples)
        normalized_file.write_frames(samples)
        normalized_file.switch_mode('r')

        # If overwriting the original sound file, delete the original stereo
        # audio file from the system and replace the audio object with the mono
        # audio object created earlier. Re-name the mono audio file to be the
        # same as the audio file it was replacing
        if overwrite_original:
            self.replace_audiofile(normalized_filename)
            return None
        else:
            return normalized_file

    def check_mono(self):
        """Check that the audio file is a mono audio file"""
        if self.channels != 1:
            return False
        return True

    def replace_audiofile(self, replacement_filename):
        """
        Replace the current audiofile and audiofile object with the file
        specified.
        """
        pathops.file_must_exist(replacement_filename)
        self.close()
        os.remove(self.wavpath)
        os.rename(replacement_filename, self.wavpath)
        replacement_file = pysndfile.PySndfile(
            self.wavpath,
            mode='r',
        )
        self.channels = replacement_file.channels()
        self.samplerate = replacement_file.samplerate()
        self.pysndfile_object = replacement_file
        self.mode = 'r'

    def convert_to_mono(self, overwrite_original=False):
        # TODO: Implement mixdown for multi-channel audio other than 2 channel
        # stereo.

        # Get current file name and it's extension
        (current_filename, current_fileextension) = (
            os.path.splitext(self.wavpath)
        )
        # Create a seperate filepath to use for the mono file to be created
        mono_filename = ''.join(
            (current_filename, ".mono", current_fileextension)
        )
        # If the mono audio file already exists then use that to replace the
        # stereo file, rather than computing again from scratch
        if os.path.exists(mono_filename):
            self.replace_audiofile(mono_filename)
            return None
        # If the file is already mono then stop
        if self.channels == 1:
            return None
        # Create the empty mono file object
        mono_file = AudioFile(
            mono_filename,
            mode='w',
            format=self.get_format(),
            channels=1,
            samplerate=self.get_samplerate()
        ).open()
        # Read current file in chunks and convert to mono by deviding all
        # samples by 2 and combining to create a single signal
        self.seek(0, 0)
        i = 0
        chunk_size = 2048
        while i < self.frames():
            chunk = self.read_grain(i, chunk_size, padding=False)
            chunk = ((chunk[:, 0] * 0.5) + (chunk[:, 1] * 0.5))
            mono_file.write_frames(chunk)
            i += chunk_size
        # If overwriting the original sound file, delete the original stereo
        # audio file from the system and replace the audio object with the mono
        # audio object created earlier. Re-name the mono audio file to be the
        # same as the audio file it was replacing
        if overwrite_original:
            del mono_file
            self.replace_audiofile(mono_filename)
            return None
        else:
            return mono_file

    @__if_open
    def rename_file(self, filename):
        """
        Renames the audio file associated with the object to the name
        specified as an argument
        """
        # TODO: Consider the race condition here. Is this a problem?
        # Check name doesn't already exist
        if os.path.exists(filename):
            raise ValueError("The filepath: {0} is an already existing file")
        # Check name is a valid file path
        if not os.path.exists(os.path.dirname(filename)):
            raise ValueError("The filepath: {0} does not point to an existing "
                             "directory".format(filename))
        # Check name has the same extension as previous file
        old_ext = os.path.splitext(self.wavpath)[1]
        new_ext = os.path.splitext(filename)[1]
        if old_ext != new_ext:
            raise ValueError("The renamed file's extension ({0})"
                             "must be the same as the original extension"
                             "({1})".format(old_ext, new_ext))
        # Delete pysndfile object
        seek = self.get_seek_position()
        del self.pysndfile_object
        # Rename file
        os.rename(self.wavpath, filename)
        # Reinitialize pysndfile object
        self.pysndfile_object = pysndfile.PySndfile(
            filename,
            mode='r',
            format=self.format,
            samplerate=self.samplerate,
            channels=self.channels
        )
        self.wavpath = filename
        # Re-set seek position to previous position
        self.seek(seek, 0)

    def get_seek_position(self):
        """Returns the current seeker position in the file"""
        return self.seek(0, 1)

    def ms_to_samps(self, ms):
        """
        Converts milliseconds to samples based on the sample rate of the audio
        file
        """
        seconds = ms / 1000.0
        return int(round(seconds * self.samplerate))

    def secs_to_samps(self, seconds):
        """
        Converts seconds to samples based on the sample rate of the audio file
        """
        return int(round(seconds * self.samplerate))

    def samps_to_secs(self, samps):
        """
        Converts samples to seconds based on the sample rate of the audio
        file
        """
        return float(samps) / self.samplerate

    def samps_to_ms(self, samps):
        """
        Convert samples to milliseconds based on the sample rate of the audio
        file
        """
        return float(samps) / self.samplerate * 1000.0

    def plot_grain_to_graph(self, start_index, number_of_samps):
        """Use matplotlib to create a graph of the audio file."""
        samps = self.read_grain(start_index, self.ms_to_samps(number_of_samps))
        self.plot_array_to_graph(samps)

    def fade_audio(self, audio, position, fade_time, mode):
        """
        Fade the audio in or out linearly from the position specified over the
        time specified.
        audio: A numpy array of audio to manipulate
        start_position: The starting position to begin the fade from (ms)
        fade_time: The length of the fade (ms)
        mode: choose to fade the audio in or out (string: "in" or "out")
        """
        if mode == "in":
            # Calculate the amplitude values to multiply the audio by
            fade = np.linspace(0.0, 1.0, self.ms_to_samps(fade_time))
            position = self.ms_to_samps(position)
            # multiply samples by the fade values from the start position for
            # the duration of the fade
            audio[position:position+fade.size] *= fade
            # zero any samples before the fade in
            audio[:position] *= 0

        elif mode == "out":
            # Calculate the amplitude values to multiply the audio by
            fade = np.linspace(1.0, 0.0, self.ms_to_samps(fade_time))
            position = self.ms_to_samps(position)
            # multiply samples by the fade values from the start position for
            # the duration of the fade
            audio[position:position+fade.size] *= fade
            # zero any samples after the fade in
            audio[position+fade.size:] *= 0
        else:
            self.logger.exception("{0} is not a valid fade option. Use either \"in\" or "
                  "\"out\"".format(mode))
            raise ValueError
        return audio

    def check_not_empty(self):
        """Check that the file contains audio"""
        if self.frames() > 0:
            return True
        return False

    def check_valid(self, force_mono=False):
        """
        Test to make sure that the audio file is valid for use.
        ie mono, not empty
        """
        if not self.check_mono():
            if force_mono:
                self.convert_to_mono(overwrite_original=True)
                return True
            return False
        if not self.check_not_empty():
            return False
        return True

    def switch_mode(self, mode):
        assert mode == 'r' or mode == 'w'
        # Change mode only if it is different from the currently set mode
        if self.mode != mode:
            seek = self.get_seek_position()
            del self.pysndfile_object
            self.pysndfile_object = pysndfile.PySndfile(
                self.wavpath,
                mode=mode,
                format=self.format,
                channels=self.channels,
                samplerate=self.samplerate
            )
            self.pysndfile_object.seek(seek)
            self.mode = mode

    @staticmethod
    def gen_window(window_type, window_size, sym=True):
        """
        Generates a window function of given size and type
        Returns a 1D numpy array

        sym: Used in the triangle window generation. When True (default),
        generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis
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
            raise ValueError("'{0}' is not a valid window"
                             " type".format(window_type))

    @staticmethod
    def plot_array_to_graph(array):
        plt.plot(array, 'r')
        plt.xlabel('Time (samples)')
        plt.ylabel('sample value')
        plt.show()

    @staticmethod
    def normalize_audio(audio, maximum=1.0):
        """
        Normalize array of audio so that the maximum sample value == the
        maximum provided
        """
        if audio.size < 1:
            raise ValueError("Audio array is empty. Cannot be normalized""")
        max_sample = np.max(np.abs(audio))
        ratio = maximum / max_sample
        audio = audio * ratio
        return audio

    @staticmethod
    def mono_arrays_to_stereo(array1, array2):
        """
        Converts to horizontal numpy arrays to one concatenated verticaly
        stacked array that can be written to a stereo file.

        eg:
            array1 = np.array([0.0, 0.1, 0.2, 0.3])
            array2 = np.array([0.4, 0.5, 0.6, 0.7])

            result:
                np.array([[0.0, 0.4]
                          [0.1, 0.5]
                          [0.2, 0.6]
                          [0.3, 0.7]])
        """
        return np.hstack((np.vstack(array1), np.vstack(array2)))

    @staticmethod
    def gen_white_noise(length, gain):
        """
        Generate mono white noise of the number of samples specified.

        length (samples)
        gain (silence 0.0 - full volume 1.0)
        """
        return np.random.uniform(low=-gain, high=gain, size=length)

    @staticmethod
    def gen_tone(length, gain, frequency, wavetype):
        """
        Generate a wave form.

        length (ms)
        gain (silence 0.0 - full volume 1.0)
        frequency (hz)
        wavetype: "sine" "square" "triangle" "saw" "rev-saw"
        """
        pass

    @staticmethod
    def gen_ADSR_envelope(
        attack,
        decay,
        sustain,
        sustain_length,
        release,
        samplerate=44100,
        gain=1.0
    ):
        """
        generate an ADSR envelope and applies to the audio.

        attack:         (float) attack time in ms
        decay:          (float) decay time in ms
        sustain:        (float) sustain level (0.0-1.0)
        sustain_length: (float) length of sustain in ms
        release:        (float) release time in ms
        """
        sustain_array = np.empty(sustain_length*samplerate)
        sustain_array.fill(sustain)
        envelope = np.concatenate(
            (np.linspace(0.0, gain, attack*samplerate),
             np.linspace(gain, sustain, (decay*samplerate)+1)[1:],
             sustain_array,
             np.linspace(sustain, 0.0, (release*samplerate)+1)[1:]),
            axis=2
        )
        return envelope

    @staticmethod
    def gen_default_wav(path, overwrite_existing=False, mode='w', channels=1):
        """
        Convenience method that creates a wav file with the following spec at
        the path given:
            Samplerate: 44.1Khz
            Bit rate: 24Bit
        """
        if os.path.exists(path):
            if not overwrite_existing:
                raise IOError(
                    ''.join(("File: \"", path, "\" already exists."))
                )
            else:
                os.remove(path)

        return AudioFile(
            path,
            mode,
            format=pysndfile.construct_format("wav", "pcm24"),
            channels=channels,
            samplerate=44100
        ).open()

    def __repr__(self):
        return 'AudioFile(name={0}, wav={1})'.format(self.name, self.wavpath)


class AnalysedAudioFile(AudioFile):

    """Generates and stores analysis information for an audio file"""

    def __init__(self, *args, **kwargs):
        # Initialise the AudioFile parent class
        self.logger = logging.getLogger('audiofile.AnalysedAudioFile')
        super(AnalysedAudioFile, self).__init__(*args, **kwargs)

        # Initialise database variables
        # Stores the path to the database if object is part of a database.
        self.db_dir = kwargs.pop('db_dir', None)

        # Refferences the HDF5 file object to use for storing analysis data.
        analysis_file = kwargs.pop('data_file', None)

        self.analysis_group = self.create_analysis_group(analysis_file)

        # If True then files are re-analysed, discarding any previous analysis.
        self.force_analysis = kwargs.pop('reanalyse', False)

        # Analysis members. If an analysis is specified either as a tag, or as
        # a filepath, it will be generated and either saved at the path
        # specified or if one isn't specified, it will be created.
        # A set containing tags for analyses to be created for the file
        self.analyses = kwargs["analyses"]

    def create_analysis(self):
        # Create the analysis objects for analyses that have been specified in
        # the analyses member variable.
        if 'fft' in self.analyses:
            self.FFT = FFTAnalysis(self, self.analysis_group)
        else:
            self.logger.info("Skipping FFT analysis.")
            self.FFT = None

        if 'rms' in self.analyses:
            self.RMS = RMSAnalysis(self, self.analysis_group)
        else:
            self.logger.info("Skipping RMS analysis.")
            self.RMS = None

        # Create Zero crossing analysis
        if 'zerox' in self.analyses:
            self.ZeroX = ZeroXAnalysis(self, self.analysis_group)
        else:
            self.logger.info("Skipping zero-crossing analysis.")
            self.ZeroX = None

        if 'spccntr' in self.analyses:
            self.SpectralCentroid = SpectralCentroidAnalysis(self, self.analysis_group)
        else:
            self.logger.info("Skipping Spectral Centroid analysis.")
            self.SpectralCentroid = None

        if 'spcsprd' in self.analyses:
            self.SpectralSpread = SpectralSpreadAnalysis(self, self.analysis_group)
        else:
            self.logger.info("Skipping Spectral Spread analysis.")
            self.SpectralSpread = None

    def create_analysis_group(self, analysis_file):
        """
        Create group for object to store analyses for this audio file.

        Audio file analyses are organized in groups per audio file.
        This function creates a group in the analysis HDF5 file with the name
        of the audio file. Analyses of this file are stored in analysis
        sub-groups.
        """
        # If an analysis file object is not provided, try to create on based on
        # the object's name and audio file location.
        if not analysis_file:
            if self.db_dir:
                # Raise error as database should have analysis file.
                raise IOError("Database doesn't have an analysis file.")
            else:
                # Attempt to create a new analysis file using the name of the
                # audiofile.
                path = os.path.split(self.wavpath)[0]
                name = '_'.join((os.path.splitext(self.name)[0], 'analysis_data.hdf5'))
                datapath = os.path.join(path, name)
                analysis_file = h5py.File(datapath, 'a')
        # Create a group to store analyses for this file in
        try:
            return analysis_file.create_group(self.name)
        except ValueError:
            self.logger.warning("A file with the same name ({0}) already "
                                "exists in the analysis data. Using data from "
                                "this file.".format(self.name))
            return analysis_file[self.name]

    def __enter__(self):
        super(AnalysedAudioFile, self).__enter__()
        if not self.check_valid(force_mono=True):
            raise IOError(
                "File isn't valid: {0}\nCheck that file is mono and isn't "
                "empty".format(self.name))

        return self

    def open(self):
        return self

    def plot_rms_to_graph(self):
        """
        Uses matplotlib to create a graph of the audio file and the generated
        RMS values
        """
        # Get audio samples from the audio file
        audio_array = self.read_frames()[:(44100 * 5)]
        # Create an empty array which will contain rms frame number and value
        # pairs
        rms_contour = self.get_rms_from_file(start=0, end=(44100 * 5))
        plt.plot(audio_array, 'b', rms_contour, 'r')
        plt.xlabel("Time (samples)")
        plt.ylabel("sample value")
        plt.show()

    # -------------------------------------------------------------------------
    # GENERAL ANALYSIS METHODS

    def __repr__(self):
        return ('AnalysedAudioFile(name={0}, wav={1}, '
                'rms={2}, attack={3}, zerox={4})'.format(self.name,
                                                         self.wavpath,
                                                         self.RMS.rmspath,
                                                         self.Attack.attackpath,
                                                         self.zeroxpath))


class AudioDatabase:

    """A class for encapsulating a database of AnalysedAudioFile objects."""

    def __init__(
        self,
        audio_dir,
        db_dir=None,
        analysis_list=["rms", "atk", "zerox", "fft", "spccntr", "spcsprd"],
        reanalyse=False
    ):
        """
        Create the folder hierachy for the database of files to be stored in.

        Adds any pre existing audio files and analyses to the object
        automatically.
        audio_dir:
        db_dir:
        analysis_list:
        """
        self.logger = logging.getLogger(__name__ + '.AudioDatabase')
        # TODO: Check that analysis strings in analysis_list are valid analyses

        # Check that all analysis list args are valid
        valid_analyses = {'rms', 'zerox', 'fft', 'spccntr', 'spcsprd'}
        for analysis in analysis_list:
            if analysis not in valid_analyses:
                raise ValueError("\'{0}\' is not a valid analysis type".format(analysis))

        analysis_list = set(analysis_list)

        self.logger.info("Initialising Database...")
        # define a list of sub-directory names for each of the analysis
        # parameters

        # Create a dictionary to store reference to the content of the database
        self.audio_file_list = []
        """
        db_content = collections.defaultdict(
            lambda: {i: None for i in {'audio', 'data'}}
        )
        """

        # Check that audio directory exists
        if not os.path.exists(audio_dir):
            raise IOError("The audio directory provided ({0}) doesn't "
                          "exist").format(audio_dir)

        # If the database directory isnt specified then the directory where the
        # audio files are stored will be used
        if not db_dir:
            db_dir = audio_dir

        # Check to see if the database directory already exists
        # Create if not
        pathops.dir_must_exist(db_dir)

        def initialise_subdir(dirkey, db_dir):
            """
            Create a subdirectory in the database with the name of the key
            provided.
            Returns the path to the created subdirectory.
            """
            # Make sure database subdirectory exists
            directory = os.path.join(db_dir, dirkey)
            try:
                # If it doesn't, Create it.
                os.mkdir(directory)
                self.logger.info(''.join(("Created directory: ", directory)))
            except OSError as err:
                # If it does exist, add it's content to the database content
                # dictionary.
                if os.path.exists(directory):
                    self.logger.warning("\'{0}\' directory already exists:"
                    " {1}".format(dirkey, os.path.relpath(directory)))
                    if dirkey == 'audio':
                        for item in pathops.listdir_nohidden(directory):
                            self.audio_file_list.append(item)
                    """
                    for item in pathops.listdir_nohidden(directory):
                        db_content[os.path.splitext(item)[0]][dirkey] = (
                            os.path.join(directory, item)
                        )
                    """
                else:
                    raise err
            return directory

        # Create a sub directory for every key in the analysis list
        # store reference to this in dictionary
        self.logger.info("Creating sub-directories...")
        directory_set = {'audio', 'data'}
        subdir_paths = {
            key: initialise_subdir(key, db_dir) for key in directory_set
        }

        self.logger.info("Moving any audio to sub directory...")

        valid_filetypes = {'.wav', '.aif', '.aiff'}
        # Move audio files to database
        # For all files in the audio dirctory...
        for item in pathops.listdir_nohidden(audio_dir):
            # If the file is a valid file type...
            if os.path.splitext(item)[1] in valid_filetypes:
                self.logger.debug(''.join(("File added to database content: ", item)))
                # Get the full path for the file
                wavpath = os.path.join(audio_dir, item)
                # If the file isn't already in the database...
                if not os.path.isfile(
                    '/'.join((subdir_paths["audio"], os.path.basename(wavpath)))
                ):
                    # Copy the file to the database
                    shutil.copy2(wavpath, subdir_paths["audio"])
                    self.logger.info(''.join(("Moved: ", item, "\tTo directory: ",
                          subdir_paths["audio"], "\n")))
                else:
                    self.logger.info(''.join(("File:  ", item, "\tAlready exists at: ",
                          subdir_paths["audio"])))
                # Add the file's path to the database content dictionary
                self.audio_file_list.append(
                    os.path.join(subdir_paths["audio"], item)
                )

        # Create data file for storing analysis data for the database
        datapath = os.path.join(subdir_paths['data'], 'analysis_data.hdf5')
        self.data = h5py.File(datapath, 'a')
        self.analysed_audio_list = []

        for item in self.audio_file_list:
            filepath = os.path.join(subdir_paths['audio'], item)
            print("--------------------------------------------------")
            # if there is no wav file then skip
            try:
                with AnalysedAudioFile(
                    filepath,
                    'r',
                    analyses=analysis_list,
                    name=os.path.basename(item),
                    db_dir=db_dir,
                    data_file=self.data,
                    reanalyse=reanalyse
                ) as AAF:
                    AAF.create_analysis()
                    self.analysed_audio_list.append(AAF)
            except IOError as err:
                # Skip any audio file objects that can't be analysed
                self.logger.warning("File cannot be analysed: {0}\nReason: {1}\n"
                      "Skipping...".format(item, err))
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          file=sys.stdout)
                continue
        print("--------------------------------------------------")
        self.logger.debug("Analysis Finished.")
        for i in self.analysed_audio_list:
            with i as AAF:
                print(i.SpectralCentroid.analysis['frames'])
        # with self.analysed_audio_list[48] as AAF:
            # AAF.FFT.plotstft(AAF.read_grain(), AAF.samplerate, binsize=AAF.ms_to_samps(100))

    def close(self):
        self.data.close()
    def __enter__(self):
        return self
    def __exit__(self):
        self.close()

