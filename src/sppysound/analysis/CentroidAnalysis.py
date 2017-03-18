
from __future__ import print_function, division
import os
import numpy as np
import logging
from scipy import signal
from numpy.lib import stride_tricks
import pdb


from AnalysisTools import ButterFilter
from fileops import pathops

from Analysis import Analysis

logger = logging.getLogger(__name__)


class CentroidAnalysis(Analysis):

    """
    Temporal centroid descriptor class for generation of temporal centroid
    audio analysis.

    This descriptor calculates the temporal centroid for overlapping grains of
    an AnalysedAudioFile object.  A full definition of temporal centroid
    analysis can be found in the documentation.

    Arguments:

    - analysis_group: the HDF5 file group to use for the storage of the
      analysis.

    - config: The configuration module used to configure the analysis
    """

    def __init__(self, AnalysedAudioFile, frames, analysis_group, config=None):
        super(CentroidAnalysis, self).__init__(AnalysedAudioFile, frames, analysis_group, 'Centroid')
        self.logger = logging.getLogger(__name__+'.{0}Analysis'.format(self.name))
        # Store reference to the file to be analysed
        self.AnalysedAudioFile = AnalysedAudioFile

        self.analysis_group = analysis_group
        self.logger.info("Creating Centroid analysis for {0}".format(self.AnalysedAudioFile.name))
        self.create_analysis(frames)

    @staticmethod
    def create_centroid_analysis(frames, window_size=512,
                            window=signal.triang,
                            overlapFac=0.5):
        """
        Calculate the Centroid values of windowed segments of the audio file and
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

        # Generate a window function to apply to centroid windows before analysis
        win = window(window_size)
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

        frames *= win
        weighted_sum = np.sum((np.arange(frames.shape[1])+1) * frames, axis=1)

        centroid = weighted_sum / np.sum(frames, axis=1)

        return centroid

    def hdf5_dataset_formatter(self, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.
        '''
        samplerate = self.AnalysedAudioFile.samplerate
        centroid = self.create_centroid_analysis(*args, **kwargs)
        centroid_times = self.calc_centroid_frame_times(centroid, args[0], samplerate)
        return ({'frames': centroid, 'times': centroid_times}, {})

    @staticmethod
    def calc_centroid_frame_times(centroidframes, sample_frames, samplerate):

        """Calculate times for frames using sample size and samplerate."""

        if hasattr(sample_frames, '__call__'):
            sample_frames = sample_frames()
        # Get number of frames for time and frequency
        timebins = centroidframes.shape[0]
        # Create array ranging from 0 to number of time frames
        scale = np.arange(timebins+1)
        # divide the number of samples by the total number of frames, then
        # multiply by the frame numbers.
        centroid_times = (float(sample_frames.shape[0])/float(timebins)) * scale[:-1].astype(float)
        # Divide by the samplerate to give times in seconds
        return centroid_times
