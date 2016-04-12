from __future__ import print_function, division
import os
import shutil
import collections
from scipy import signal
import numpy as np
import pysndfile
import pdb
import sys
import traceback
import logging
import h5py
import multiprocessing as mp
from collections import namedtuple, defaultdict
import gc
from functools import wraps

from fileops import pathops
import analysis.RMSAnalysis as RMSAnalysis
import analysis.PeakAnalysis as PeakAnalysis
import analysis.AttackAnalysis as AttackAnalysis
import analysis.ZeroXAnalysis as ZeroXAnalysis
import analysis.FFTAnalysis as FFTAnalysis
import analysis.SpectralCentroidAnalysis as SpectralCentroidAnalysis
import analysis.SpectralSpreadAnalysis as SpectralSpreadAnalysis
import analysis.SpectralFluxAnalysis as SpectralFluxAnalysis
import analysis.SpectralCrestFactorAnalysis as SpectralCrestFactorAnalysis
import analysis.SpectralFlatnessAnalysis as SpectralFlatnessAnalysis
import analysis.CentroidAnalysis as CentroidAnalysis
import analysis.F0Analysis as F0Analysis
import analysis.VarianceAnalysis as VarianceAnalysis
import analysis.KurtosisAnalysis as KurtosisAnalysis
import analysis.SkewnessAnalysis as SkewnessAnalysis
import analysis.F0HarmRatioAnalysis as F0HarmRatioAnalysis

logger = logging.getLogger(__name__).addHandler(logging.NullHandler())

class AudioFile(object):

    """
    Object for storing and accessing basic information for an audio file.

    This object is a wrapper for the pysndfile audio object. It provides
    additional functionality alongside the ability to open and close audiofiles
    without deleting their containing object.

    Arguments:

    - filepath: path to the file to be opened/written to

    - mode: mode to open file in. either 'r' (read) or 'w' (write)

    - format: the file format to use when opening a new file for writing. see
      get_sndfile_formats() for more information.

    - channels: the number of audio channels to use.

    - samplerate: the samplerate in herts to use.

    - name: the sound file object name.
    """

    def __init__(
        self,
        filepath,
        mode,
        format=None,
        channels=None,
        samplerate=None,
        name=None,
        *args,
        **kwargs
    ):
        self.logger = logging.getLogger(__name__ + '.AudioFile')
        self.logger.debug("Initialised AudioFile")

        self.filepath = filepath
        # TODO: If a name isn't provided then create a default name based on
        # the file name without an extension
        self.name = name
        self.mode = mode
        self.samplerate = samplerate
        self.format = format
        self.channels = channels
        self.frames = None
        self.times = None

    def __enter__(self):
        """Allow AudioFile object to be opened by 'with' statements"""
        self.logger.debug("Opening soundfile {0}".format(self.filepath))
        if self.mode == 'r':
            if not os.path.exists(self.filepath):
                raise IOError(
                    "Cannot open {0} for reading as it cannot be "
                    "found.".format(self.filepath)
                )
            self.pysndfile_object = pysndfile.PySndfile(
                self.filepath,
                mode=self.mode
            )
            self.samplerate = self.get_samplerate()
            self.format = self.get_format()
            self.channels = self.get_channels()
            self.frames = self.get_frames()
            return self
        else:
            self.pysndfile_object = pysndfile.PySndfile(
                self.filepath,
                mode=self.mode,
                format=self.format,
                channels=self.channels,
                samplerate=self.samplerate
            )
            return self

    def open(self):
        """Use for opening the associated audio file outside of a with statement"""
        self.logger.debug("Opening soundfile {0}".format(self.filepath))
        return self.__enter__()

    def close(self):
        """Use for closing the associated audio file outside of a with statement"""
        self.logger.debug("Closing soundfile {0}".format(self.filepath))
        self.pysndfile_object = None

    def __exit__(self, type, value, traceback):
        """Closes sound file when exiting 'with' statement."""
        self.logger.debug("Closing soundfile {0}".format(self.filepath))
        self.pysndfile_object = None

    def __if_open(method):
        """Handles error from using methods when the audio file is closed"""
        @wraps(method)
        def wrapper(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except AttributeError, err:
                raise IOError("{0}: Audio file isn't open.".format(err), sys.exc_info()[2])

        return wrapper

    @__if_open
    def get_channels(self):
        """Return number of channels of sndfile."""
        self.channels = self.pysndfile_object.channels()
        return self.channels

    @__if_open
    def encoding_str(self):
        """
        Return string representation of encoding (e.g. pcm16).

        See get_sndfile_encodings() for a list of available encoding strings
        that are supported by a given sndfile format
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
    def get_frames(self):
        """Return number of frames in file (number of samples per channel)."""
        self.frames = self.pysndfile_object.frames()
        return self.frames

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

        (e.g. aiff) see get_sndfile_formats() for a complete
        list of file formats
        """
        return self.pysndfile_object.major_format_str()

    @__if_open
    def read_frames(self, nframes=-1, dtype=np.float64):
        """
        Read the given number of frames and fill numpy array.

        Arguments

        - nframes: <int>

        - number of frames to read (default = -1 -> read all).

        - dtype: <numpy dtype>

        - dtype of the returned array containing read data (see note).

        Notes:
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

        Arguments

        - offset: <int>
          the number of frames (eg two samples for stereo files) to move
          relatively to position set by whence.

        - whence: <int>
          only 0 (beginning), 1 (current) and 2 (end of the file) are valid.

        - mode: <string>
          If set to 'rw', both read and write pointers are updated. If 'r' is
          given, only read pointer is updated, if 'w', only the write one is
          (this may of course make sense only if you open the file in a certain
          mode).

        Returns
        offset: int the number of frames from the beginning of the file

        Notes:
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
        Set one of the stringtypes to the string given as argument.

        If you try to write a stringtype that is not supported by the library
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

        Arguments:

        - input: <numpy array>
        containing data to write.

        Notes:

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

        Arguments:

        - major: Major sndfile format for the list of available formats.
          format should be specified as a string, using one of the strings
          returned by get_sndfile_formats()
        """
        return self.pysndfile_object.get_sndfile_encodings(major)

    @__if_open
    def read_grain(self, start_index=0, grain_size=None, padding=True):
        """
        Read a grain of audio from the file. if grain ends after the end of
        the file, the grain can be padded with zeros using the padding
        argument.

        Audio object seeker is not changed

        Arguments:

        - start_index: the index in samples to read from.

        - grain_size: The size of the grain (in samples) to read

        - padding: if the end of the audio file is reaches, the grain will be
          padded with additional zeros.
        """
        self.switch_mode('r')
        if start_index < 0:
            start_index = self.get_frames() + start_index
        if not grain_size:
            grain_size = self.get_frames()
        grain_size = int(grain_size)
        position = self.get_seek_position()
        # Read grain
        index = self.pysndfile_object.seek(start_index, 0)
        if index + grain_size > self.get_frames():
            grain = self.read_frames(self.get_frames() - index)
            if padding:
                grain = np.pad(
                    grain,
                    (0, index + grain_size - self.get_frames()),
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
            os.path.splitext(self.filepath)
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
        os.remove(self.filepath)
        os.rename(replacement_filename, self.filepath)
        self.mode = 'r'
        self.__enter__()

    def convert_to_mono(self, overwrite_original=False):
        """
        Converts stereo audiofiles to mono.

        Arguments:

        - overwrite_original: If True then the current object will be
          reloaded as the mono file. Otherwise, the new mono file will be
          returned as a new AudioFile object.
        """
        # TODO: Implement mixdown for multi-channel audio other than 2 channel
        # stereo.

        # Get current file name and it's extension
        (current_filename, current_fileextension) = (
            os.path.splitext(self.filepath)
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
        while i < self.get_frames():
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

        Arguments:

        - filename: the new path of the audio file.
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
        old_ext = os.path.splitext(self.filepath)[1]
        new_ext = os.path.splitext(filename)[1]
        if old_ext != new_ext:
            raise ValueError("The renamed file's extension ({0})"
                             "must be the same as the original extension"
                             "({1})".format(old_ext, new_ext))
        # Delete pysndfile object
        seek = self.get_seek_position()
        del self.pysndfile_object
        # Rename file
        os.rename(self.filepath, filename)
        # Reinitialize pysndfile object
        self.pysndfile_object = pysndfile.PySndfile(
            filename,
            mode='r',
            format=self.format,
            samplerate=self.samplerate,
            channels=self.channels
        )
        self.filepath = filename
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

    def fade_audio(self, audio, position, fade_time, mode):
        """
        Fade the audio in or out linearly from the position specified over the
        time specified.

        Arguments:

        - audio: A numpy array of audio to manipulate

        - start_position: The starting position to begin the fade from (ms)

        - fade_time: The length of the fade (ms)

        - mode: choose to fade the audio in or out (string: "in" or "out")
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
        if self.get_frames() > 0:
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
        """
        Switch audiofile to mode specified.

        This allows for convenient reading and writing of audiofiles without
        direct closing and opening of the underlying pysndfile object.
        """
        assert mode == 'r' or mode == 'w'
        # Change mode only if it is different from the currently set mode
        if self.mode != mode:
            seek = self.get_seek_position()
            del self.pysndfile_object
            self.mode = mode
            self.__enter__()
            self.pysndfile_object.seek(seek)

    def generate_grain_times(self, grain_length, overlap, save_times=False):
        """
        Generates an array of start and finish pairs based on overlapping
        frames at the grain length specified.

        Note that only full grains within the size of the sample are returned.
        incomplete grains found at the end of files are ignored.

        Arguments:

        - grain_length: length of each grain in seconds.

        - overlap: the factor by which grains overlap (integer)
        """
        length = self.samps_to_ms(self.frames)
        hop_size = grain_length / overlap
        grain_count = int(length / hop_size) - 1
        times = np.arange(grain_count).reshape(-1, 1)
        times = np.hstack((times, times)).astype(np.dtype('float64'))
        times *= hop_size
        times[:, 1] += grain_length
        if save_times:
            # Save grain times as a member variable for later refference.
            self.times = times
        return times

    def __getitem__(self, key):
        """
        Allow for grains to be retreived by indexing after grain times have been generated.
        """
        if self.times == None:
            raise IndexError("AudioFile object grain times must be generated "
                             "before grains can be accesed by index. Try running "
                             "AnalysedAudioFile.generate_grain_times(grain_size, "
                                                                    "overlap, save_times=True)")
        grain_times = self.times[key].copy()
        grain_times *= (self.samplerate / 1000)
        return self.read_grain(start_index=grain_times[0], grain_size=grain_times[1]-grain_times[0])


    @staticmethod
    def gen_window(window_type, window_size, sym=True):
        """
        Generates a window function of given size and type
        Returns a 1D numpy array

        sym: Used in the triangle window generation. When True (default),
        generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis

        Available window types:

        - hanning

        - hamming

        - bartlett

        - blackman

        - kaiser

        - triangle
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
    def normalize_audio(audio, maximum=1.0):
        """
        Normalize array of audio so that the maximum sample value is equal to
        the maximum provided
        """
        if audio.size < 1:
            raise ValueError("Audio array is empty. Cannot be normalized""")
        max_sample = np.max(np.abs(audio))
        audio *= maximum / max_sample
        return audio

    @staticmethod
    def mono_arrays_to_stereo(array1, array2):
        """
        Converts two horizontal numpy arrays to one concatenated verticaly
        stacked array that can be written to a stereo file.
        """
        return np.hstack((np.vstack(array1), np.vstack(array2)))

    @staticmethod
    def gen_white_noise(length, gain):
        """
        Generate mono white noise of the number of samples specified.

        Arguments:

        - length (samples)

        - gain (silence 0.0 - full volume 1.0)
        """
        return np.random.uniform(low=-gain, high=gain, size=length)

    @staticmethod
    def gen_default_wav(path, overwrite_existing=False, mode='w', channels=1):
        """
        Convenience method that creates a wav file with the following spec at
        the path given:

        - Samplerate: 44.1Khz

        - Bit rate: 24Bit
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
        return 'AudioFile(name={0}, wav={1})'.format(self.name, self.filepath)


class AnalysedAudioFile(AudioFile):

    """
    Generates and stores analysis information for an audio file.

    Arguments:

    - (All AudioFile arguments)

    - db_dir: if the object is part of a database, this is the path to the root
      of the database.

    - config: The config module used for configuration settings. See examples
      for further details.

    - data_file: the path to the HDF5 file used to store analyses for this
      audio file.

    - reanalyse: bool for whether to overwrite any previously created analyses
      for this audio file.

    - available_analyses: a list of strings for each analyses to be generated.
      ie. [\'f0\', \'rms\']
    """

    def __init__(self, *args, **kwargs):
        # Initialise the AudioFile parent class
        self.logger = logging.getLogger('audiofile.AnalysedAudioFile')
        super(AnalysedAudioFile, self).__init__(*args, **kwargs)

        # Initialise database variables
        # Stores the path to the database if object is part of a database.
        self.db_dir = kwargs.pop('db_dir', None)

        # Store configuration file used for various settings.
        self.config = kwargs.pop('config', None)

        # Refferences the HDF5 file object to use for storing analysis data.
        analysis_file = kwargs.pop('data_file', None)

        self.analysis_storage = self.create_analysis_group(analysis_file)

        # If True then files are re-analysed, discarding any previous analysis.
        self.force_analysis = kwargs.pop('reanalyse', False)

        # Analysis members. If an analysis is specified either as a tag, or as
        # a filepath, it will be generated and either saved at the path
        # specified or if one isn't specified, it will be created.
        # A set containing tags for analyses to be created for the file
        self.available_analyses = kwargs.pop("analyses", None)

    def create_analysis(self):
        """Generate all analyses that have been set in the self.available_analyses member."""
        analysis_object = namedtuple("AnalysisObject", "name, analysis_object")
        analysis_object_list = [
            analysis_object("fft", FFTAnalysis),
            analysis_object("rms", RMSAnalysis),
            analysis_object("zerox", ZeroXAnalysis),
            analysis_object("spccntr", SpectralCentroidAnalysis),
            analysis_object("spcsprd", SpectralSpreadAnalysis),
            analysis_object("spcflux", SpectralFluxAnalysis),
            analysis_object("spccf", SpectralCrestFactorAnalysis),
            analysis_object("spcflatness", SpectralFlatnessAnalysis),
            analysis_object("f0", F0Analysis),
            analysis_object("peak", PeakAnalysis),
            analysis_object("centroid", CentroidAnalysis),
            analysis_object("variance", VarianceAnalysis),
            analysis_object("kurtosis", KurtosisAnalysis),
            analysis_object("skewness", SkewnessAnalysis),
            analysis_object("harm_ratio", F0HarmRatioAnalysis)
        ]

        self.analyses = defaultdict(None)

        # Create the analysis objects for analyses that have been specified in
        # the analyses member variable.
        for analysis in analysis_object_list:
            if analysis.name in self.available_analyses:
                self.analyses[analysis.name] = analysis.analysis_object(self, self.analysis_storage, config=self.config)
        self.analysis_storage.file.flush()
        gc.collect()


    def create_analysis_group(self, analysis_file):
        """
        Create HDF5 group for object to store analyses for this audio file.

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
                path = os.path.split(self.filepath)[0]
                name = '_'.join((os.path.splitext(self.name)[0], 'analysis_data.hdf5'))
                datapath = os.path.join(path, name)
                analysis_file = h5py.File(datapath, 'a')
        # Create a group to store analyses for this file in
        group_name = ''.join(("analysis/", self.name))
        try:
            analysis_file.create_group(group_name)
        except ValueError:
            self.logger.warning("A file with the same name ({0}) already "
                                "exists in the analysis data. Using data from "
                                "this file.".format(group_name))
        analysis_file[group_name].attrs['filepath'] = self.filepath
        return analysis_file[group_name]

    def __enter__(self):
        """Allow AudioFile object to be opened by 'with' statements"""
        super(AnalysedAudioFile, self).__enter__()
        if not self.check_valid(force_mono=True):
            raise IOError(
                "File isn't valid: {0}\nCheck that file is mono and isn't "
                "empty".format(self.name))

        return self

    def open(self):
        return self

    def analysis_data_grains(self, times, analysis, *args, **kwargs):
        """
        retrieve data for analysis within start and end time pairs in the format specified.

        Arguments:

        - times: an array of start and end times to retrieve analysis from (np.array)

        - analysis: analysis string specifying analysis to retrieve
        """
        format_type = kwargs.pop("format", None)

        analysis_object = self.analyses[analysis]

        if len(times.shape) != 2:
            times = np.array([times])
        analysis_frames, selection = analysis_object.get_analysis_grains(times[:, 0], times[:, 1])

        if format_type:
            analysis_frames = analysis_object.analysis_formatter(analysis_frames, selection, format_type)

        return analysis_frames, selection

    def __repr__(self):
        return ('AnalysedAudioFile(name={0})'.format(self.name))
