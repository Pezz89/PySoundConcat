from __future__ import print_function, division
import numpy as np
import logging
from numpy.lib import stride_tricks
from Analysis import Analysis
import pdb

logger = logging.getLogger(__name__)


class ZeroXAnalysis(Analysis):

    """
    Zero-corssing descriptor class for generation of zero-crossing rate
    analysis.

    This descriptor calculates the zero-crossing rate for overlapping grains of
    an AnalysedAudioFile object.  A full definition of zero-crossing analysis
    can be found in the documentation.

    Arguments:

    - analysis_group: the HDF5 file group to use for the storage of the
      analysis.

    - config: The configuration module used to configure the analysis
    """

    def __init__(self, AnalysedAudioFile, frames, analysis_group, config=None):
        super(ZeroXAnalysis, self).__init__(AnalysedAudioFile,frames, analysis_group, 'ZeroCrossing')
        self.logger = logging.getLogger(__name__+'.{0}Analysis'.format(self.name))
        self.analysis_group = analysis_group
        self.logger.info("Creating zero crossing analysis for {0}".format(self.AnalysedAudioFile.name))
        self.create_analysis(frames)

    @staticmethod
    def create_zerox_analysis(
        frames,
        window_size=512,
        overlapFac=0.5,
        *args,
        **kwargs
    ):
        """Generate zero crossing value for window of the signal"""
        if hasattr(frames, '__call__'):
            frames = frames()
        hopSize = int(window_size - np.floor(overlapFac * window_size))

        # zeros at beginning (thus center of 1st window should be for sample nr. 0)
        samples = np.append(np.zeros(int(window_size/2.0)), frames)

        # cols for windowing
        cols = np.ceil((len(samples) - window_size) / float(hopSize)) + 1
        # zeros at end (thus samples can be fully covered by frames)
        samples = np.append(samples, np.zeros(window_size))

        # TODO: Better handeling of zeros based on previous sign would improve
        # accuracy.
        epsilon = np.finfo(float).eps
        samples[samples == 0.] += epsilon

        frames = stride_tricks.as_strided(
            samples,
            shape=(int(cols), window_size),
            strides=(samples.strides[0]*hopSize, samples.strides[0])
        ).copy()
        zero_crossing = np.sum(np.abs(np.diff(np.sign(frames))), axis=1)
        return zero_crossing

    @staticmethod
    def calc_zerox_frame_times(zerox_frames, sample_frames, samplerate):

        """Calculate times for frames using sample size and samplerate."""

        if hasattr(sample_frames, '__call__'):
            sample_frames = sample_frames()
        # Get number of frames for time and frequency
        timebins = zerox_frames.shape[0]
        # Create array ranging from 0 to number of time frames
        scale = np.arange(timebins+1)
        # divide the number of samples by the total number of frames, then
        # multiply by the frame numbers.
        zerox_times = (sample_frames.shape[0]/timebins) * scale[:-1]
        # Divide by the samplerate to give times in seconds
        return zerox_times

    def hdf5_dataset_formatter(self, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.
        '''
        samples = self.AnalysedAudioFile.read_grain()
        samplerate = self.AnalysedAudioFile.samplerate
        output = self.create_zerox_analysis(*args, **kwargs)
        times = self.calc_zerox_frame_times(output, args[0], samplerate)
        return ({'frames': output, 'times': times}, {})
