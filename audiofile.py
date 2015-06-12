import os
import shutil
import collections
from scipy import signal
import numpy as np
import math
from pysndfile import PySndfile
import matplotlib.pyplot as plt

import fileops.pathops as pathops
import analysis.rms.RMSAnalysis as RMSAnalysis

class AudioFile(PySndfile):
    """Object for storing and accessing basic information for an audio file"""

    def __new__(cls, filename, mode, **kwargs):
        inst = PySndfile.__new__(cls, filename, mode)
        return inst

    def __init__(self, wavpath, mode,
                 format=None,
                 channels=None,
                 samplerate=None,
                 name=None, *args, **kwargs):
        self.wavpath = wavpath
        self.name = name

        super(AudioFile, self).__init__(wavpath,
                                        mode=mode,
                                        format=format,
                                        channels=channels,
                                        samplerate=samplerate)

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
        index = self.seek(start_index, 0)
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

    def __repr__(self):
        return 'AudioFile(name={0}, wav={1})'.format(self.name, self.wavpath)

class AnalysedAudioFile(AudioFile):

    """Generates and stores analysis information for an audio file"""

    def __init__(self, *args, **kwargs):
        # Initialise the AudioFile parent class
        super(AnalysedAudioFile, self).__init__(*args, **kwargs)

        #---------------
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
        if kwargs["rmspath"] or self.db_dir:
            self.RMS = RMSAnalysis.RMSAnalysis(self, kwargs.pop('rmspath', None))
        else:
            self.RMS = None

        #---------------
        # Initialise Attack estimation variables
        self.attackpath = kwargs.pop('atkpath', None)
        self.attack_start = None
        self.attack_end = None
        self.attack_size = None

        #---------------
        # Initialise zero-crossing variables
        self.zeroxpath = kwargs.pop('zeroxpath', None)

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
    # ATTACK ESTIMATION METHODS
    def scale_to_range(self, array, high=1.0, low=0.0):
        mins = np.min(array)
        maxs = np.max(array)
        rng = maxs - mins
        return high - (((high - low) * (maxs - array)) / rng)

    def create_attack_analysis(self, multiplier=3):
        """
        Estimate the start and end of the attack of the audio
        Adaptive threshold method (weakest effort method) described here:
        http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf
        """
        # Make sure RMS has been calculated
        if not self.rmspath:
            raise IOError("RMS analysis is required to estimate attack")
        if not self.attackpath:
            if not self.db_dir:
                raise IOError("Analysed Audio object must have an atk file path"
                              "or be part of a database")
            self.attackpath = os.path.join(
                self.db_dir,
                "atk",
                self.name +
                ".lab")
        with open(self.attackpath, 'w') as attackfile:
            print "Creating attack estimation file:\t", os.path.relpath(self.attackpath)
            rms_contour = self.get_rms_from_file()
            rms_contour = self.scale_to_range(rms_contour)
            thresholds = np.arange(1, 11) * 0.1
            thresholds = thresholds.reshape(-1, 1)
            # Find first index of rms that is over the threshold for each
            # thresholds
            threshold_inds = np.argmax(rms_contour >= thresholds, axis=1)

            # TODO:Need to make sure rms does not return to a lower threshold after
            # being > a threshold.

            # Calculate the time difference between each of the indexes
            ind_diffs = np.ediff1d(threshold_inds)
            # Find the average (mean?) time between thresholds
            mean_ind_diff = np.mean(ind_diffs)
            # Calculate the start threshold by finding the first threshold that
            # goes below the average time * the multiplier
            if np.any(ind_diffs < mean_ind_diff * multiplier):
                attack_start_ind = threshold_inds[
                    np.argmax(
                        ind_diffs < mean_ind_diff *
                        multiplier)]
            else:
                attack_end_ind = threshold_inds[0]
            # Calculate the end threshold by thr same method except looking above
            # the average time * the multiplier
            if np.any(ind_diffs > mean_ind_diff * multiplier):
                attack_end_ind = threshold_inds[
                    np.argmax(
                        ind_diffs > mean_ind_diff *
                        multiplier)]
            else:
                attack_end_ind = threshold_inds[-1]
            # Refine position by searching for local min and max of these values
            self.attack_start = self.samps_to_secs(attack_start_ind)
            self.attack_end = self.samps_to_secs(attack_end_ind)
            attackfile.write(
                "{0}\t0\tAttack_start\n{1}\t0\tAttack_end".format(
                    self.attack_start,
                    self.attack_end))

    def calc_log_attack_time(self):
        """
        Calculate the logarithm of the time duration between the time the
        signal starts to the time that the signal reaches it's stable part
        Described here:
        http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf
        """
        if not self.attack_start or not self.attack_end:
            raise ValueError("Attack times must be calculated before calling"
                             "the log attack time method")
        self.logattacktime = math.log10(self.attackend-self.attackstart)

    #-------------------------------------------------------------------------
    # ZERO-CROSSING DETECTION METHODS

    def create_zerox_analysis(self, window_size=25):
        """Generate zero crossing detections for windows of the signal"""
        self.zeroxpath = os.path.join(self.db_dir, "zerox", self.name + ".lab")
        with open(self.zeroxpath, 'w') as zeroxfile:
            print "Creating zero-crossing file:\t\t", os.path.relpath(self.zeroxpath)
            i = 0
            while i < self.frames():
                zero_crossings = np.where(
                    np.diff(
                        np.sign(
                            self.read_grain(
                                i,
                                window_size))))[0].size
                zeroxfile.write(
                    "{0} {1} {2}\n".format(
                        self.samps_to_secs(i),
                        self.samps_to_secs(
                            i+window_size),
                        zero_crossings))
                i += window_size

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
                    shutil.copy2(wavpath, subdir_paths["wav"])
                    print "Moved: ", item, "\nTo directory: ", subdir_paths["wav"]
                    print "---------------------------------------------\n"
                    db_content[os.path.splitext(item)[0]]["wav"] = (
                        os.path.join(subdir_paths["wav"], item)
                    )

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

    def generate_analyses(self):
        print "*****************************************"
        print "Analysing audio files in database..."
        print "*****************************************"
        for audiofile in self.analysed_audio_list:
            print audiofile.name, ":"
            audiofile.create_attack_analysis()
            audiofile.create_zerox_analysis(window_size=11025/8)
            print "---------------------------------------------\n"
