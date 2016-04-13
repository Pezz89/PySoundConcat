from __future__ import print_function, division
import os
import numpy as np
import logging
from scipy import signal
from numpy.lib import stride_tricks
import pdb

from fileops import pathops

from Analysis import Analysis

logger = logging.getLogger(__name__)


class SkewnessAnalysis(Analysis):

    """
    Skewness descriptor class for generation of temporal skewness audio analysis.

    This descriptor calculates thetemporal skewness for overlapping grains of
    an AnalysedAudioFile object.  A full definition of skewness analysis can be
    found in the documentation.

    Arguments:

    - analysis_group: the HDF5 file group to use for the storage of the
      analysis.

    - config: The configuration module used to configure the analysis
    """

    def __init__(self, AnalysedAudioFile, frames, analysis_group, config=None):
        super(SkewnessAnalysis, self).__init__(AnalysedAudioFile,frames, analysis_group, 'skewness')
        self.logger = logging.getLogger(__name__+'.{0}Analysis'.format(self.name))
        # Store reference to the file to be analysed
        self.AnalysedAudioFile = AnalysedAudioFile

        if config:
            self.window_size = config.skewness["window_size"] * self.AnalysedAudioFile.samplerate / 1000
            self.overlap = 1. / config.skewness["overlap"]

        try:
            variance = self.AnalysedAudioFile.analyses["variance"]
        except KeyError:
            raise KeyError("Variance analysis is required for skewness "
                             "analysis.")

        self.analysis_group = analysis_group
        self.logger.info("Creating skewness analysis for {0}".format(self.AnalysedAudioFile.name))
        self.create_analysis(frames, variance.analysis['frames'][:], self.window_size, overlapFac=self.overlap)

    @staticmethod
    def create_skewness_analysis(
        frames,
        variance,
        window_size=512,
        window=signal.hanning,
        overlapFac=0.5
    ):
        """
        Calculate the skewness values of windowed segments of the audio file and
        save to disk.
        """
        if hasattr(frames, '__call__'):
            frames = frames()
        # Calculate the period of the window in hz
        # lowest_freq = 1.0 / window_size
        # Filter frequencies lower than the period of the window
        # filter = ButterFilter()
        # filter.design_butter(lowest_freq, self.AnalysedAudioFile.samplerate)
        # TODO: Fix filter
        # frames = filter.filter_butter(frames)

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

        frame_mean = np.mean(frames, axis=1)

        variance_cubed = np.sqrt(variance)**3

        a =  ((1 / window_size)) * np.sum(((frames-np.vstack(frame_mean))**3), axis=1)
        skewness = a / variance_cubed

        return skewness

    def hdf5_dataset_formatter(self, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.
        '''
        samplerate = self.AnalysedAudioFile.samplerate
        skewness = self.create_skewness_analysis(*args, **kwargs)
        skewness_times = self.calc_skewness_frame_times(skewness, args[0], samplerate)
        return ({'frames': skewness, 'times': skewness_times}, {})

    @staticmethod
    def calc_skewness_frame_times(skewnessframes, sample_frames, samplerate):

        """Calculate times for frames using sample size and samplerate."""

        if hasattr(sample_frames, '__call__'):
            sample_frames = sample_frames()
        # Get number of frames for time and frequency
        timebins = skewnessframes.shape[0]
        # Create array ranging from 0 to number of time frames
        scale = np.arange(timebins+1)
        # divide the number of samples by the total number of frames, then
        # multiply by the frame numbers.
        skewness_times = (float(sample_frames.shape[0])/float(timebins)) * scale[:-1].astype(float)
        # Divide by the samplerate to give times in seconds
        skewness_times = skewness_times / samplerate
        return skewness_times
