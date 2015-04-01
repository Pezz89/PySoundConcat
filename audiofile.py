
import os
import shutil
import collections
from scipy import signal
import numpy as np
from pysndfile import PySndfile

import matplotlib.pyplot as plt


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
        super(AudioFile, self).__init__(wavpath,
                                        mode=mode,
                                        format=format,
                                        channels=channels,
                                        samplerate=samplerate)
        self.wavpath = wavpath
        self.name = name

    def audio_file_info(self):
        """ Prints audio information """
        print "*************************************************"
        print "File:                    ", os.path.relpath(self.wavpath)
        print "No. channels:            ", self.channels()
        print "Samplerate:              ", self.samplerate()
        print "Format:                  ", self.format()
        print "No. Frames:              ", self.frames()
        print "Encoding string:         ", self.encoding_str()
        print "Major format string:     ", self.major_format_str()
        print "Seekable?:               ", bool(self.seekable())
        print "Errors?:                 ", self.error()
        print "*************************************************"

    def read_grain(self, start_index, grain_size):
        """
        Read a grain of audio from the file. if grain ends after the end of
        the file, the grain is padded with zeros.
        Audio object seeker is not changed
        """
        position = self.get_seek_position()
        #Read grain
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

    def get_seek_position(self):
        """Returns the current seeker position in the file"""
        return self.seek(0, 1)


class AnalysedAudioFile(AudioFile):
    """Generates and stores analysis information for an audio file"""

    def __init__(self, *args, **kwargs):
        #---------------
        #Initialise database variables
        #Stores the path to the database
        self.db_dir = kwargs.pop('db_dir', None)

        #---------------
        #Initialise f0 variables
        #Stores the path to the f0 file
        self.f0path = kwargs.pop('f0path', None)

        #---------------
        #Initialise RMS variables
        #Stores the path to the RMS lab file
        self.rmspath = kwargs.pop('rmspath', None)
        #Stores the number of RMS window values when calculating the RMS contour
        self.rms_window_count = None
        #If an RMS file is provided then count the number of lines (1 for each
        #window)
        if self.rmspath:
            with open(self.rmspath, 'r') as rmsfile:
                self.rms_window_count = sum(1 for line in rmsfile)

        #---------------
        #Initialise Attack estimation variables
        self.attackpath = kwargs.pop('atkpath', None)

        #Initialise the AudioFile parent class
        super(AnalysedAudioFile, self).__init__(*args, **kwargs)

    #-------------------------------------------------------------------------
    #RMS ESTIMATION METHODS
    def create_rms_analysis(self,
                            window_size=25,
                            window_type="triangle",
                            window_overlap=8):
        """Generate an energy contour analysis by calculating the RMS values of windows segments of the audio file"""
        window_size = self.ms_to_samps(window_size)
        window_function = self.gen_window(window_type, window_size)
        if not self.rmspath:
            if not self.db_dir:
                raise IOError("Analysed Audio object must have an RMS file path"
                              "or be part of a database")
            self.rmspath = os.path.join(self.db_dir, "rms", self.name + ".lab")
        i = 0
        try:
            with open(self.rmspath, 'w') as rms_file:
                #Count the number of windows analysed
                self.rms_window_count = 0
                while i < self.frames():
                    #Read frames from audio file
                    frames = self.read_grain(i, window_size)
                    #Apply window function to frames
                    frames = frames * window_function
                    #Calculate the RMS value of the current window of frames
                    rms = np.sqrt(np.mean(np.square(frames)))
                    #Write data to RMS .lab file
                    rms_file.write("{0} {1:6f}\n".format(
                        i + int(round(window_size / 2.0)), rms))
                    #Iterate frame
                    i += int(round(window_size / window_overlap))
                    self.rms_window_count += 1
            return self.rmspath
        except IOError:
            return False


    def get_rms_from_file(self, start=0, end=-1):
        """
        Read values from RMS file between start and end points provided (in
        samples)
        """
        #Convert -1 index to final window index
        if end == -1:
            end = self.rms_window_count
        #Create empty array with a size equal to the maximum possible RMS values
        rms_array = np.empty((2, self.rms_window_count))
        #Open the RMS file
        with open(self.rmspath, 'r') as rmsfile:
            i = 0
            for line in rmsfile.xreadlines():
                #Split the values and convert to their correct types
                time, value = line.split()
                time = int(time)
                value = float(value)
                #If the values are within the desired range, add them to the
                #array
                if time >= start and time <= end:
                    #The first value will be rounded down to the start
                    if i == 0:
                        time = start
                    rms_array[0][i] = time
                    rms_array[1][i] = value
                    i += 1
            #The last value will be rounded up to the end
            rms_array[0][i] = end
        rms_array = rms_array[:, start:start+i]
        rms_contour = np.interp(np.arange(end), rms_array[0], rms_array[1])
        return rms_contour

    def plot_rms_to_graph(self):
        """
        Uses matplotlib to create a graph of the audio file and the generated
        RMS values
        """
        #Get all audio samples from the audio file
        audio_array = self.read_frames()[:(44100 * 5)]
        #Create an empty array which will contain rms frame number and value
        #pairs
        rms_contour = self.get_rms_from_file(start=0, end=(44100 * 5))
        plt.plot(audio_array, 'b', rms_contour, 'r')
        plt.xlabel("Time (samples)")
        plt.ylabel("sample value")
        plt.show()

    #-------------------------------------------------------------------------
    #ATTACK ESTIMATION METHODS
    def estimate_attack(self):
        """
        Estimate the start and end of the attack of the audio
        Adaptive threshold method (weakest effort method) described here:
        http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf

        """
        #Make sure RMS has been calculated
        if not self.rmspath:
            raise IOError("RMS analysis is required to estimate attack")
        if not self.attackpath:
            if not self.db_dir:
                raise IOError("Analysed Audio object must have an atk file path"
                              "or be part of a database")
            self.attackpath = os.path.join(self.db_dir, "atk", self.name + ".mrk")
        with open(self.attackpath, 'w') as attackfile:
            rms_contour = self.get_rms_from_file()
            max_rms = np.max(rms_contour[1])
            thresholds = np.arange(10, 0, step=-1) * (max_rms / 10.0) 
            print thresholds
    
    #-------------------------------------------------------------------------
    #GENERAL ANALYSIS METHODS
    def gen_window(self, window_type, window_size, sym=True):
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
            raise ValueError("'{0}' is not a valid window"
                             " type".format(window_type))

    def ms_to_samps(self, ms):
        """Converts milliseconds to samples based on the sample rate of the audio file"""
        seconds = ms / 1000.0
        return int(round(seconds * self.samplerate()))

    def __repr__(self):
        return ('AnalysedAudioFile(name={0}, wav={1}, '
                'rms={2})'.format(self.name, self.wavpath, self.rmspath))


class AudioDatabase:
    """A class for encapsulating a database of AnalysedAudioFile objects"""

    def __init__(self, audio_dir, db_dir=None):
        """Creates the folder hierachy for the database of files to be stored in"""

        #Create a dictionary to store reference to the content of the database
        db_content = collections.defaultdict(lambda: {"wav": None, "rms": None,
            "atk": None})

        #If the database directory isnt specified then the directory where the audio files are stored will be used
        if not db_dir:
            db_dir = audio_dir

        try:
            os.mkdir(db_dir)
        except OSError as err:
            if os.path.exists(db_dir):
                print "database directory already exists"
            else:
                raise err

        #Make sure wav directory exists
        wav_dir = os.path.join(db_dir, "wav")
        try:
            os.mkdir(wav_dir)
            print "Created directory: ", wav_dir
        except OSError as err:
            if os.path.exists(os.path.join(db_dir, "wav")):
                print "wav directory already exists"
                for item in os.listdir(wav_dir):
                    db_content[os.path.splitext(item)[0]]["wav"] = (
                        os.path.join(wav_dir, item)
                    )
            else:
                raise err

        #Make sure rms directory exists
        rms_dir = os.path.join(db_dir, "rms")
        try:
            os.mkdir(rms_dir)
            print "Created directory: ", rms_dir
        except OSError as err:
            if os.path.exists(os.path.join(rms_dir)):
                print "rms directory already exists"
                for item in os.listdir(rms_dir):
                    db_content[os.path.splitext(item)[0]]["rms"] = (
                        os.path.join(rms_dir, item)
                    )
            else:
                raise err

        #Make sure rms directory exists
        atk_dir = os.path.join(db_dir, "atk")
        try:
            os.mkdir(atk_dir)
            print "Created directory: ", atk_dir
        except OSError as err:
            if os.path.exists(os.path.join(atk_dir)):
                print "atk directory already exists"
                for item in os.listdir(atk_dir):
                    db_content[os.path.splitext(item)[0]]["atk"] = (
                        os.path.join(atk_dir, item)
                    )
            else:
                raise err
        
        #Move audio files from directory to database
        if os.path.exists(audio_dir):
            for item in os.listdir(audio_dir):
                if os.path.splitext(item)[1] == ".wav":
                    wavpath = os.path.join(audio_dir, item)
                    shutil.move(wavpath, wav_dir)
                    print "Moved: ", item, "\nTo directory: ", wav_dir
                    db_content[os.path.splitext(item)[0]]["wav"] = (
                        os.path.join(wav_dir, item)
                    )

        self.analysed_audio_list = []
        for key in db_content.viewkeys():
            if not db_content[key]["wav"]:
                continue
            self.analysed_audio_list.append(
                AnalysedAudioFile(db_content[key]["wav"], 'r',
                                  rmspath=db_content[key]["rms"],
                                  name=key,
                                  db_dir=db_dir))

    def generate_analyses(self):
        for audiofile in self.analysed_audio_list:
            audiofile.create_rms_analysis()
            audiofile.estimate_attack()
