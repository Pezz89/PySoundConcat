import os
import shutil
import collections
from scipy import signal
import numpy as np
import math
import pysndfile
import matplotlib.pyplot as plt

import fileops.pathops as pathops
import analysis.RMSAnalysis as RMSAnalysis
import analysis.AttackAnalysis as AttackAnalysis
import analysis.ZeroXAnalysis as ZeroXAnalysis


class AudioFile:
    """Object for storing and accessing basic information for an audio file"""
    def __init__(self, wavpath, mode,
                 format=None,
                 channels=None,
                 samplerate=None,
                 name=None, *args, **kwargs):

        self.wavpath = wavpath
        # TODO: If a name isn't provided then create a default name based n the
        # file name without an extension
        self.name = name
        self.pysndfile_object = pysndfile.PySndfile(
            wavpath,
            mode=mode,
            format=format,
            channels=channels,
            samplerate=samplerate
        )
    def channels(self):
        """return number of channels of sndfile"""
        return self.pysndfile_object.channels()
    def encoding_str(self):
        """
        return string representation of encoding (e.g. pcm16)
        see pysndfile.get_sndfile_encodings() for a list of available
        encoding strings that are supported by a given sndfile format
        """
        return self.pysndfile_object.encoding_str()
    def error(self):
        """report error numbers related to the current sound file"""
        return self.pysndfile_object.error()
    def format(self):
        """return raw format specification from sndfile"""
        return self.pysndfile_object.format()
    def frames(self):
        """return number for frames (number of samples per channel)"""
        return self.pysndfile_object.frames()
    def get_strings(self):
        """
        get all stringtypes from the sound file.
        see stringtype_name_top_id.keys() for the list of strings that
        are supported by the libsndfile version you use.
        """
        return self.pysndfile_object.get_strings()
    def major_format_str(self):
        """
        return short string representation of major format
        (e.g. aiff) see pysndfile.get_sndfile_formats() for a complete
        lst of fileformats
        """
        return self.pysndfile_object.major_format_str()
    def read_frames(self, nframes=-1, dtype=np.float64):
        """
        Read the given number of frames and put the data into a numpy
        array of the requested dtype.

        Parameters
        nframes: <int>
        number of frames to read (default = -1 -> read all).
        dtype: <numpy dtype>
        dtype of the returned array containing read data (see note).

        Notes
        One column per channel.
        """
        return self.pysndfile_object.read_frames(nframes, dtype)
    def rewind(self, mode='rw'):
        """
        rewind read/write/read and write position given by mode to
        start of file
        """
        return self.pysndfile_object.format(mode)
    def samplerate(self):
        """return samplerate"""
        return self.pysndfile_object.samplerate()
    def seek(self, offset, whence=0, mode='rw'):
        """
        Seek into audio file: similar to python seek function,
        taking only in account audio data.

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
    def seekable(self):
        """return true for soundfiles that support seeking"""
        return self.seekable()
    def set_auto_clipping(self, arg=True):
        """
        enable auto clipping when reading/writing samples from/to sndfile.

        auto clipping is enabled by default. auto clipping is required by
        libsndfile to properly handle scaling between sndfiles with pcm encoding
        and float representation of the samples in numpy. When auto clipping
        is set to on reading pcm data into a float vector and writing it back
        with libsndfile will reproduce the original samples. If auto clipping
        is off, samples will be changed slightly as soon as the amplitude is
        close to the sample range because libsndfile applies slightly different
        scaling factors during read and write
        """
        return self.pysndfile_object.set_auto_clipping(arg)
    def set_string(self, stringtype_name, string):
        """
        set one of the stringtypes to the strig given as argument. If you try
        to write a stringtype that is not supported byty the library a RuntimeError
        will be raised
        """
        return self.pysndfile_object.set_string(stringtype_name, string)
    def strError(self):
        """report error strings related to the current sound file"""
        return self.pysndfile_object.strError()
    def writeSync(self):
        """
        call the operating system's function to force the writing of all file
        cache buffers to disk the file.
        No effect if file is open as read
        """
        return self.pysndfile_object.writeSync()
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
        you should make sure the input data are normalized normalized data (that
        is in the range [-1..1] - which will corresponds to the maximum range
        allowed by the integer bitwidth).
        """
        return self.pysndfile_object.write_frames(input)
    def construct_format(self, major, encoding):
        """
        construct a format specification for libsndfile from major format string
        and encoding string
        """
        return self.pysndfile_object.construct_format(major, encoding)
    def get_pysndfile_version(self):
        """return tuple describing the version of pysndfile"""
        return self.pysndfile_object.get_pysndfile_version()
    def get_sndfile_version(self):
        """return a tuple of ints representing the version of the libsdnfile that is used"""
        return self.pysndfile_object.get_sndfile_version()
    def get_sndfile_formats(self):
        """Return lists of available file formats supported by libsndfile and pysndfile."""
        return self.pysndfile_object.get_sndfile_formats()
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
        print '*************************************************'
        print 'File:                    ', os.path.relpath(self.wavpath)
        print 'No. channels:            ', self.channels()
        print 'Samplerate:              ', self.samplerate()
        print 'Format:                  ', self.format()
        print 'No. Frames:              ', self.frames()
        print 'Encoding string:         ', self.encoding_str()
        print 'Major format string:     ', self.major_format_str()
        print 'Seekable?:               ', bool(self.seekable())
        print 'Errors?:                 ', self.error()
        print '*************************************************'

    def read_grain(self, start_index, grain_size):
        """
        Read a grain of audio from the file. if grain ends after the end of
        the file, the grain is padded with zeros.
        Audio object seeker is not changed
        """
        position = self.get_seek_position()
        # Read grain
        index = self.pysndfile_object.seek(start_index, 0)
        if index + grain_size > self.frames():
            grain = self.read_frames(self.frames() - index)
            grain = np.pad(grain, (0, index + grain_size - self.frames()),
                           'constant',
                           constant_values=(0, 0))

        else:
            grain = self.read_frames(grain_size)
        self.seek(position, 0)
        return grain

    def normalize_audio(self, maximum=1.0):
        """Normalize frames so that the maximum sample value == the maximum provided"""
        if self.mode != 'rw':
            raise ValueError("AudioFile object must be in read/write mode to"
                             "normalize audio")
        frames = self.read_frames()
        max_sample = np.max(frames)
        ratio = maximum / max_sample
        frames = frames * ratio
        self.write_frames(frames)

    def get_seek_position(self):
        """Returns the current seeker position in the file"""
        return self.seek(0, 1)

    def ms_to_samps(self, ms):
        """
        Converts milliseconds to samples based on the sample rate of the audio
        file
        """
        seconds = ms / 1000.0
        return int(round(seconds * self.samplerate()))

    def secs_to_samps(self, seconds):
        """
        Converts seconds to samples based on the sample rate of the audio file
        """
        return int(round(seconds * self.samplerate()))

    def samps_to_secs(self, samps):
        """
        Converts samples to seconds based on the sample rate of the audio
        file
        """
        return float(samps) / self.samplerate()

    def samps_to_ms(self, samps):
        """
        Converts samples to milliseconds based on the sample rate of the audio
        file
        """
        return float(samps) / self.samplerate() * 1000.0

    def plot_grain_to_graph(self, start_index, number_of_samps):
        """
        Uses matplotlib to create a graph of the audio file
        """
        samps = self.read_grain(start_index, self.ms_to_samps(4000))
        plt.plot(samps, 'r')
        plt.xlabel('Time (samples)')
        plt.ylabel('sample value')
        plt.show()

    def fade_audio(self, audio, position, fade_time, mode):
        """
        Fades the in or out linearly from the position specified over the time specified.
        audio: A numpy array of audio to manipulate
        start_position: The starting position to begin the fade from (ms)
        fade_time: The length of the fade (ms)
        mode: choose to fade the audio in or out (string: "in" or "out")
        """
        if mode == "in":
            #Calculate the amplitude values to multiply the audio by
            fade = np.linspace(0.0, 1.0, self.ms_to_samps(fade_time))
            position = self.ms_to_samps(position)
            #multiply samples by the fade values from the start position for
            #the duration of the fade
            audio[position:fade.size] *= fade
            #zero any samples before the fade in
            audio[:position] *= 0

        elif mode == "out":
            #Calculate the amplitude values to multiply the audio by
            fade = np.linspace(1.0, 0.0, self.ms_to_samps(fade_time))
            position = self.ms_to_samps(position)
            #multiply samples by the fade values from the start position for
            #the duration of the fade
            audio[position:position-fade.size] *= fade
            #zero any samples after the fade in
            audio[position-fade.size:] *= 0
        else:
            print mode, " is not a valid fade option. Use either \"in\" or \"out\""
            raise ValueError

        return audio

    def check_mono(self):
        """Check that the audio file is a mono audio file"""
        if self.channels() != 1:
            return False
        return True

    def check_not_empty(self):
        """Check that the file contains audio"""
        if self.frames() > 0:
            return True
        return False

    def check_valid(self):
        """Test to make sure that the audio file is valid for use. ie mono, not empty"""
        if not self.check_mono():
            return False
        if not self.check_not_empty():
            return False
        return True

    def switch_mode(self):
        inst = self.__new__(
            AudioFile,
            self.wavpath,
            mode='w',
            format=self.format(),
            channels=self.channels(),
            samplerate=self.samplerate()
        )

        super(self, inst).__init__(
        )

    @staticmethod
    def gen_white_noise(length, gain):
        """
        Generates a numpy array of white noise of the length specified
        length (ms)
        gain (silence 0.0 - full volume 1.0)
        """
        pass

    @staticmethod
    def gen_tone(length, gain, frequency, wavetype):
        """
        Generate a wave form
        length (ms)
        gain (silence 0.0 - full volume 1.0)
        frequency (hz)
        wavetype: "sine" "square" "triangle" "saw" "rev-saw"
        """
        pass

    @staticmethod
    def gen_ADSR_envelope(audio, attack, decay, sustain, release):
        """
        generates an ADSR envelope and applies to the audio
        audio: A numpy array of audio to manipulate
        attack:
        decay:
        sustain:
        release:
        """
        pass

    @staticmethod
    def gen_default_wav(path, overwrite_existing=False):
        """
        Convenience method that creates a wav file with the following spec at the path given:
            Samplerate: 44.1Khz
            Bit rate: 24Bit
        """
        if os.path.exists(path) and not overwrite_existing:
            raise IOError(''.join(("File: \"", path, "\" already exists.")))
        return AudioFile(
            path,
            "rw",
            format=pysndfile.construct_format("wav", "pcm24"),
            channels=1,
            samplerate=44100
        )

    def __repr__(self):
        return 'AudioFile(name={0}, wav={1})'.format(self.name, self.wavpath)

class AnalysedAudioFile(AudioFile):

    """Generates and stores analysis information for an audio file"""

    def __init__(self, *args, **kwargs):
        # Initialise the AudioFile parent class
        super(AnalysedAudioFile, self).__init__(*args, **kwargs)

        # Initialise database variables
        # Stores the path to the database
        self.db_dir = kwargs.pop('db_dir', None)

        if not self.check_valid():
            raise IOError("File isn't valid: {0}\nCheck that file is mono and isn't empty".format(self.name))

        #---------------
        # Initialise f0 variables
        # Stores the path to the f0 file
        self.f0path = kwargs.pop('f0path', None)

        # Create RMS analysis object if file has an rms path or is part of a
        # database
        if "rmspath" in kwargs or self.db_dir:
            self.RMS = RMSAnalysis(self, kwargs.pop('rmspath', None))
        else:
            print "No RMS path for: {0}".format(self.name)
            self.RMS = None

        # Create attack estimation analysis
        if "atkpath" in kwargs or self.db_dir:
            self.Attack = AttackAnalysis(self, kwargs.pop('atkpath', None))
        else:
            print "No Attack path for: {0}".format(self.name)
            self.Attack = None

        # Create Zero crossing analysis
        if "zeroxpath" in kwargs or self.db_dir:
            self.ZeroX = ZeroXAnalysis(self, kwargs.pop('zeroxpath', None))
        else:
            print "No Zero crossing path for: {0}".format(self.name)
            self.ZeroX = None


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



    #-------------------------------------------------------------------------
    # GENERAL ANALYSIS METHODS
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

    def __repr__(self):
        return ('AnalysedAudioFile(name={0}, wav={1}, '
                'rms={2}, attack={3}, zerox={4})'.format(self.name,
                                                         self.wavpath,
                                                         self.rmspath,
                                                         self.attackpath,
                                                         self.zeroxpath))


class AudioDatabase:
    """A class for encapsulating a database of AnalysedAudioFile objects"""

    def __init__(self, audio_dir, db_dir=None, analysis_list=["wav", "rms", "atk", "zerox"]):
        """
        Creates the folder hierachy for the database of files to be stored in
        Adds any pre existing audio files and analyses to the object automatically.
        audio_dir:
        db_dir:
        analysis_list:
        """

        # TODO: Check that analysis strings in analysis_list are valid analyses

        print "*****************************************"
        print "Initialising Database..."
        print "*****************************************"
        print ""
        # define a list of sub-directory names for each of the analysis
        # parameters

        # Create a dictionary to store reference to the content of the database
        db_content = collections.defaultdict(
            lambda: {i: None for i in analysis_list}
        )

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
                print "Created directory: ", directory
            except OSError as err:
                # If it does exist, add it's content to the database content
                # dictionary.
                if os.path.exists(directory):
                    print "{0} directory already exists:\t\t{1}".format(dirkey,
                                                                        os.path.relpath(directory))
                    for item in pathops.listdir_nohidden(directory):
                        db_content[os.path.splitext(item)[0]][dirkey] = (
                            os.path.join(directory, item)
                        )
                else:
                    raise err
            return directory

        # Create a sub directory for every key in the analysis list
        # store reference to this in dictionary
        print "*****************************************"
        print "Creating sub-directories..."
        print "*****************************************"
        subdir_paths = {key: initialise_subdir(key, db_dir) for key in analysis_list}

        print ""
        print "*****************************************"
        print "Moving any audio to sub directory..."
        print "*****************************************"
        # Move audio files to database
        if os.path.exists(audio_dir):
            for item in pathops.listdir_nohidden(audio_dir):
                if os.path.splitext(item)[1] == ".wav":
                    wavpath = os.path.join(audio_dir, item)
                    if not os.path.isfile('/'.join((subdir_paths["wav"], os.path.basename(wavpath)))):
                        shutil.copy2(wavpath, subdir_paths["wav"])
                        print "Moved: ", item, "\tTo directory: ", subdir_paths["wav"]
                    else:
                        print "File:  ", item, "\tAlready exists at: ", subdir_paths["wav"]
                    db_content[os.path.splitext(item)[0]]["wav"] = (
                        os.path.join(subdir_paths["wav"], item)
                    )

        # TODO: Create a dictionary of anlyses to be passed to the
        # AnalysedAudioFile objects that determines which analyses will be
        # produced
        self.analysed_audio_list = []
        for key in db_content.viewkeys():
            # if there is no wav file then skip
            if not db_content[key]["wav"]:
                continue
            try:
                self.analysed_audio_list.append(
                    AnalysedAudioFile(db_content[key]["wav"], 'r',
                                    rmspath=db_content[key]["rms"],
                                    zeroxpath=db_content[key]["zerox"],
                                    name=key,
                                    db_dir=db_dir))
            except IOError:
                # Skip any audio file objects that can't be analysed
                continue
