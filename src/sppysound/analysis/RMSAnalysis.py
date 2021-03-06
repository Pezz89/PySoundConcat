from __future__ import print_function, division
import os
import numpy as np
import logging
from scipy import signal
from numpy.lib import stride_tricks
import pdb
from scipy.signal import butter, lfilter


from AnalysisTools import ButterFilter
from fileops import pathops

from Analysis import Analysis

logger = logging.getLogger(__name__)


class RMSAnalysis(Analysis):

    """
    RMS descriptor class for generation of RMS audio analysis.

    This descriptor calculates the Root Mean Square analysis for overlapping
    grains of an AnalysedAudioFile object.  A full definition of RMS analysis
    can be found in the documentation.

    Arguments:

    - analysis_group: the HDF5 file group to use for the storage of the
      analysis.

    - config: The configuration module used to configure the analysis
    """

    def __init__(self, AnalysedAudioFile, frames, analysis_group, config=None):
        super(RMSAnalysis, self).__init__(AnalysedAudioFile,frames, analysis_group, 'RMS')
        self.logger = logging.getLogger(__name__+'.{0}Analysis'.format(self.name))
        # Store reference to the file to be analysed
        self.AnalysedAudioFile = AnalysedAudioFile

        if config:
            self.window_size = config.rms["window_size"] * self.AnalysedAudioFile.samplerate / 1000
            self.overlap = 1. / config.rms["overlap"]
        else:
            self.window_size=512
            self.overlap = 0.5

        self.analysis_group = analysis_group
        self.logger.info("Creating RMS analysis for {0}".format(self.AnalysedAudioFile.name))
        self.create_analysis(frames, self.AnalysedAudioFile.samplerate, window_size=self.window_size, overlapFac=self.overlap, )

    @staticmethod
    def create_rms_analysis(
        frames,
        samplerate,
        window_size=512,
        window=signal.hanning,
        overlapFac=0.5
    ):
        """
        Generate RMS contour analysis.

        Calculate the RMS values of windowed segments of the audio file and
        save to disk.
        """
        if hasattr(frames, '__call__'):
            frames = frames()
        def butter_lowpass(cutoff, fs, order=5):
            # red: taken from http://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
            return b, a
        def butter_lowpass_filter(data, cutoff, fs, order=5):
            # red: taken from http://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
            b, a = butter_lowpass(cutoff, fs, order=order)
            y = lfilter(b, a, data)
            return y


        # Calculate the period of the window in hz
        lowest_freq = 1.0 / (window_size / samplerate)
        frames = butter_lowpass_filter(frames, lowest_freq, samplerate)


        # Generate a window function to apply to rms windows before analysis
        hopSize = int(window_size - np.floor(overlapFac * window_size))

        # zeros at beginning (thus center of 1st window should be for sample nr. 0)
        samples = np.append(np.zeros(np.floor(window_size/2.0)), frames)

        # cols for windowing
        cols = np.ceil((len(samples) - window_size) / float(hopSize)) + 1
        # zeros at end (thus samples can be fully covered by frames)
        samples = np.append(samples, np.zeros(window_size))

        frames = stride_tricks.as_strided(
            samples,
            shape=(cols, window_size),
            strides=(samples.strides[0]*hopSize, samples.strides[0])
        ).copy()

        if window:
            win = window(window_size)
            frames *= win
        rms = np.sqrt(np.mean(np.square(np.abs(frames)), axis=1))

        return rms


    def hdf5_dataset_formatter(self, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.
        '''
        samplerate = self.AnalysedAudioFile.samplerate
        rms = self.create_rms_analysis(*args, **kwargs)
        rms_times = self.calc_rms_frame_times(rms, args[0], samplerate)
        return ({'frames': rms, 'times': rms_times}, {})

    @staticmethod
    def calc_rms_frame_times(rmsframes, sample_frames, samplerate):

        """Calculate times for frames using sample size and samplerate."""

        if hasattr(sample_frames, '__call__'):
            sample_frames = sample_frames()
        # Get number of frames for time and frequency
        timebins = rmsframes.shape[0]
        # Create array ranging from 0 to number of time frames
        scale = np.arange(timebins+1)
        # divide the number of samples by the total number of frames, then
        # multiply by the frame numbers.
        rms_times = (float(sample_frames.shape[0])/float(timebins)) * scale[:-1].astype(float)
        # Divide by the samplerate to give times in seconds
        rms_times = rms_times / samplerate
        return rms_times
