#Embedded file name: audiofile.py
import os
import shutil
import collections
from scipy import signal
import numpy as np
import math
from pysndfile import PySndfile
import matplotlib.pyplot as plt
import fileops
from progressbar import ProgressBar

class AudioFile(PySndfile):
    """Object for storing and accessing basic information for an audio file"""

    def __new__(cls, filename, mode, **kwargs):
        inst = PySndfile.__new__(cls, filename, mode)
        return inst

    def __init__(self, wavpath, mode, format = None, channels = None, samplerate = None, name = None, *args, **kwargs):
        super(AudioFile, self).__init__(wavpath, mode=mode, format=format, channels=channels, samplerate=samplerate)
        self.wavpath = wavpath
        self.name = name

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
        index = self.seek(start_index, 0)
        if index + grain_size > self.frames():
            grain = self.read_frames(self.frames() - index)
            grain = np.pad(grain, (0, index + grain_size - self.frames()), 'constant', constant_values=(0, 0))
        else:
            grain = self.read_frames(grain_size)
        self.seek(position, 0)
        return grain

    def normalize_audio(self, maximum = 1.0):
        """Normalize frames so that the maximum sample value == the maximum provided"""
        if self.mode != 'rw':
            raise ValueError('AudioFile object must be in read/write mode tonormalize audio')
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
        if mode == 'in':
            fade = np.linspace(0.0, 1.0, self.ms_to_samps(fade_time))
            position = self.ms_to_samps(position)
            audio[position:fade.size] *= fade
            audio[:position] *= 0
        elif mode == 'out':
            fade = np.linspace(1.0, 0.0, self.ms_to_samps(fade_time))
            position = self.ms_to_samps(position)
            audio[position:position - fade.size] *= fade
            audio[position - fade.size:] *= 0
        else:
            print mode, ' is not a valid fade option. Use either "in" or "out"'
            raise ValueError
        return audio

    def fade_audio_ends(self, audio, fade_time):
        """
        Fade both ends of the audio file using the fade_audio function
        audio: A numpy array of audio to manipulate
        fade_time: The length of the fades (ms)
        """
        audio_length = audio.size
        fade_time = self.ms_to_samps(fade_time)
        if fade_time * 2.0 > audio_length:
            fade_time = math.floor(audio_length / 2.0)
        audio = self.fade_audio(audio, 0, fade_time, mode='in')
        audio = self.fade_audio(audio, audio_length, fade_time, mode='out')
        return audio

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
        self.db_dir = kwargs.pop('db_dir', None)
        self.f0path = kwargs.pop('f0path', None)
        self.rmspath = kwargs.pop('rmspath', None)
        self.rms_window_count = None
        if self.rmspath:
            with open(self.rmspath, 'r') as rmsfile:
                self.rms_window_count = sum((1 for line in rmsfile))
        self.attackpath = kwargs.pop('atkpath', None)
        self.attack_start = None
        self.attack_end = None
        self.attack_size = None
        self.zeroxpath = kwargs.pop('zeroxpath', None)
        super(AnalysedAudioFile, self).__init__(*args, **kwargs)

    def create_rms_analysis(self, window_size = 25, window_type = 'triangle', window_overlap = 8):
        """Generate an energy contour analysis by calculating the RMS values of windows segments of the audio file"""
        window_size = self.ms_to_samps(window_size)
        window_function = self.gen_window(window_type, window_size)
        if not self.rmspath:
            if not self.db_dir:
                raise IOError('Analysed Audio object must have an RMS file pathor be part of a database')
            self.rmspath = os.path.join(self.db_dir, 'rms', self.name + '.lab')
        i = 0
        try:
            with open(self.rmspath, 'w') as rms_file:
                print 'Creating RMS file:\t\t\t', os.path.relpath(self.rmspath)
                self.rms_window_count = 0
                pbar = ProgressBar(maxval=self.frames())
                while i < self.frames():
                    frames = self.read_grain(i, window_size)
                    frames = frames * window_function
                    rms = np.sqrt(np.mean(np.square(frames)))
                    rms_file.write('{0} {1:6f}\n'.format(i + int(round(window_size / 2.0)), rms))
                    i += int(round(window_size / window_overlap))
                    self.rms_window_count += 1
                    pbar.update(i)

            pbar.finish()
            return self.rmspath
        except IOError:
            return False

    def get_rms_from_file(self, start = 0, end = -1):
        """
        Read values from RMS file between start and end points provided (in
        samples)
        """
        if end == -1:
            end = self.frames()
        rms_array = np.empty((2, self.rms_window_count))
        with open(self.rmspath, 'r') as rmsfile:
            i = 0
            for line in rmsfile:
                time, value = line.split()
                time = int(time)
                value = float(value)
                if time >= start and time <= end:
                    if i == 0:
                        time = start
                    rms_array[0][i] = time
                    rms_array[1][i] = value
                    i += 1

            rms_array[0][i] = end
        rms_array = rms_array[:, start:start + i]
        rms_contour = np.interp(np.arange(end), rms_array[0], rms_array[1])
        return rms_contour

    def plot_rms_to_graph(self):
        """
        Uses matplotlib to create a graph of the audio file and the generated
        RMS values
        """
        audio_array = self.read_frames()[:220500]
        rms_contour = self.get_rms_from_file(start=0, end=220500)
        plt.plot(audio_array, 'b', rms_contour, 'r')
        plt.xlabel('Time (samples)')
        plt.ylabel('sample value')
        plt.show()

    def scale_to_range(self, array, high = 1.0, low = 0.0):
        mins = np.min(array)
        maxs = np.max(array)
        rng = maxs - mins
        return high - (high - low) * (maxs - array) / rng

    def create_attack_analysis(self, multiplier = 3):
        """
        Estimate the start and end of the attack of the audio
        Adaptive threshold method (weakest effort method) described here:
        http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf
        """
        if not self.rmspath:
            raise IOError('RMS analysis is required to estimate attack')
        if not self.attackpath:
            if not self.db_dir:
                raise IOError('Analysed Audio object must have an atk file pathor be part of a database')
            self.attackpath = os.path.join(self.db_dir, 'atk', self.name + '.lab')
        with open(self.attackpath, 'w') as attackfile:
            print 'Creating attack estimation file:\t', os.path.relpath(self.attackpath)
            rms_contour = self.get_rms_from_file()
            rms_contour = self.scale_to_range(rms_contour)
            thresholds = np.arange(1, 11) * 0.1
            thresholds = thresholds.reshape(-1, 1)
            threshold_inds = np.argmax(rms_contour >= thresholds, axis=1)
            ind_diffs = np.ediff1d(threshold_inds)
            mean_ind_diff = np.mean(ind_diffs)
            if np.any(ind_diffs < mean_ind_diff * multiplier):
                attack_start_ind = threshold_inds[np.argmax(ind_diffs < mean_ind_diff * multiplier)]
            else:
                attack_end_ind = threshold_inds[0]
            if np.any(ind_diffs > mean_ind_diff * multiplier):
                attack_end_ind = threshold_inds[np.argmax(ind_diffs > mean_ind_diff * multiplier)]
            else:
                attack_end_ind = threshold_inds[-1]
            self.attack_start = self.samps_to_secs(attack_start_ind)
            self.attack_end = self.samps_to_secs(attack_end_ind)
            attackfile.write('{0}\t0\tAttack_start\n{1}\t0\tAttack_end'.format(self.attack_start, self.attack_end))

    def calc_log_attack_time(self):
        """
        Calculate the logarithm of the time duration between the time the
        signal starts to the time that the signal reaches it's stable part
        Described here:
        http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf
        """
        if not self.attack_start or not self.attack_end:
            raise ValueError('Attack times must be calculated before callingthe log attack time method')
        self.logattacktime = math.log10(self.attackend - self.attackstart)

    def create_zerox_analysis(self, window_size = 25):
        """Generate zero crossing detections for windows of the signal"""
        self.zeroxpath = os.path.join(self.db_dir, 'zerox', self.name + '.lab')
        with open(self.zeroxpath, 'w') as zeroxfile:
            print 'Creating zero-crossing file:\t\t', os.path.relpath(self.zeroxpath)
            i = 0
            while i < self.frames():
                zero_crossings = np.where(np.diff(np.sign(self.read_grain(i, window_size))))[0].size
                zeroxfile.write('{0} {1} {2}\n'.format(self.samps_to_secs(i), self.samps_to_secs(i + window_size), zero_crossings))
                i += window_size

    @staticmethod
    def gen_window(window_type, window_size, sym = True):
        """
        Generates a window function of given size and type
        Returns a 1D numpy array

        sym: Used in the triangle window generation. When True (default),
        generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis
        """
        if window_type is 'hanning':
            return np.hanning(window_size)
        if window_type is 'hamming':
            return np.hamming(window_size)
        if window_type is 'bartlett':
            return np.bartlett(window_size)
        if window_type is 'blackman':
            return np.blackman(window_size)
        if window_type is 'kaiser':
            return np.kaiser(window_size)
        if window_type is 'triangle':
            return signal.triang(window_size, sym=sym)
        raise ValueError("'{0}' is not a valid window type".format(window_type))

    def __repr__(self):
        return 'AnalysedAudioFile(name={0}, wav={1}, rms={2}, attack={3}, zerox={4})'.format(self.name, self.wavpath, self.rmspath, self.attackpath, self.zeroxpath)


class AudioDatabase():
    """A class for encapsulating a database of AnalysedAudioFile objects"""

    def __init__(self, audio_dir, db_dir = None):
        """Creates the folder hierachy for the database of files to be stored in"""
        print '\nInitialising Database...'
        subdir_list = ['wav',
         'rms',
         'atk',
         'zerox']
        db_content = collections.defaultdict(lambda : {i:None for i in subdir_list})
        if not db_dir:
            db_dir = audio_dir
        fileops.must_exist(db_dir, msg='Database directory already exists.')

        def initialise_subdir(dirkey, db_dir):
            """
            Create a subdirectory in the database with the name of the key
            provided.
            """
            directory = os.path.join(db_dir, dirkey)
            try:
                os.mkdir(directory)
                print 'Created directory: ', directory
            except OSError as err:
                if os.path.exists(directory):
                    print '{0} directory already exists:\t\t{1}'.format(dirkey, os.path.relpath(directory))
                    for item in fileops.listdir_nohidden(directory):
                        db_content[os.path.splitext(item)[0]][dirkey] = os.path.join(directory, item)

                else:
                    raise err

            return directory

        print '\nCreating sub-directories...'
        subdir_paths = {key:initialise_subdir(key, db_dir) for key in subdir_list}
        print '\nMoving any audio to sub directory...'
        if os.path.exists(audio_dir):
            for item in fileops.listdir_nohidden(audio_dir):
                if os.path.splitext(item)[1] == '.wav':
                    wavpath = os.path.join(audio_dir, item)
                    shutil.move(wavpath, subdir_paths['wav'])
                    print ('Moved: ',
                     item,
                     '\nTo directory: ',
                     subdir_paths['wav'])
                    db_content[os.path.splitext(item)[0]]['wav'] = os.path.join(subdir_paths['wav'], item)

        self.analysed_audio_list = []
        for key in db_content.viewkeys():
            if not db_content[key]['wav']:
                continue
            self.analysed_audio_list.append(AnalysedAudioFile(db_content[key]['wav'], 'r', rmspath=db_content[key]['rms'], zeroxpath=db_content[key]['zerox'], name=key, db_dir=db_dir))

    def generate_analyses(self):
        print '\nAnalysing audio files in database...'
        for audiofile in self.analysed_audio_list:
            print audiofile.name, ':'
            audiofile.create_rms_analysis()
            audiofile.create_attack_analysis()
            audiofile.create_zerox_analysis(window_size=11025 / 8)
            print ''
