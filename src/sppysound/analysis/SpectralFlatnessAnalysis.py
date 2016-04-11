from __future__ import print_function, division
import scipy.stats as stats
import numpy as np
import logging
import pdb
import warnings

from Analysis import Analysis

class SpectralFlatnessAnalysis(Analysis):
    """
    Spectral flatness descriptor class for generation of spectral flatness
    audio analysis.

    This descriptor calculates the spectral flatness for overlapping grains of
    an AnalysedAudioFile object.  A full definition can be found in the
    documentation.

    Arguments:

    - analysis_group: the HDF5 file group to use for the storage of the
      analysis.

    - config: The configuration module used to configure the analysis
    """

    def __init__(self, AnalysedAudioFile, analysis_group, config=None):
        super(SpectralFlatnessAnalysis, self).__init__(AnalysedAudioFile, analysis_group, 'SpcFlatness')
        # Create logger for module
        self.logger = logging.getLogger(__name__+'.{0}Analysis'.format(self.name))
        # Store reference to the file to be analysed
        self.AnalysedAudioFile = AnalysedAudioFile
        self.nyquist_rate = self.AnalysedAudioFile.samplerate / 2.
        try:
            fft = self.AnalysedAudioFile.analyses["fft"]
        except KeyError:
            raise KeyError("FFT analysis is required for spectral spread "
                             "analysis.")

        self.analysis_group = analysis_group
        self.logger.info("Creating Spectral Flatness analysis for {0}".format(self.AnalysedAudioFile.name))
        self.create_analysis(
            self.create_spcflatness_analysis,
            fft.analysis['frames'][:],
        )
        self.spcflatness_window_count = None

    def hdf5_dataset_formatter(self, analysis_method, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.
        '''
        samplerate = self.AnalysedAudioFile.samplerate
        output = self.create_spcflatness_analysis(*args, **kwargs)
        times = self.calc_spcflatness_frame_times(output, self.AnalysedAudioFile.frames, samplerate)
        return ({'frames': output, 'times': times}, {})

    @staticmethod
    def create_spcflatness_analysis(fft):
        '''
        Calculate the spectral flatness of the fft frames.
        '''
        # Get the positive magnitudes of each bin.
        magnitudes = np.abs(fft)
        if not np.nonzero(magnitudes)[0].size:
            y = np.empty(magnitudes.shape[0])
            y.fill(np.nan)
            return y

        # Calculate the ratio between the two.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            # Calculate the geometric mean of magnitudes
            geo_mean = np.e**np.mean(np.log(magnitudes), axis=1)
            # Calculate the arithmetic mean of magnitudes
            arith_mean = np.mean(magnitudes, axis=1)
            spectral_flatness = geo_mean / arith_mean

        return spectral_flatness

    @staticmethod
    def calc_spcflatness_frame_times(spcflatness_frames, sample_frame_count, samplerate):

        """Calculate times for frames using sample size and samplerate."""

        # Get number of frames for time and frequency
        timebins = spcflatness_frames.shape[0]
        # Create array ranging from 0 to number of time frames
        scale = np.arange(timebins+1)
        # divide the number of samples by the total number of frames, then
        # multiply by the frame numbers.
        spcflatness_times = (float(sample_frame_count)/float(timebins)) * scale[:-1].astype(float)
        # Divide by the samplerate to give times in seconds
        spcflatness_times = spcflatness_times / samplerate
        return spcflatness_times

    def mean_formatter(self, data):
        """Calculate the mean value of the analysis data"""

        values = data[0]

        output = np.empty(len(values))
        for ind, i in enumerate(values):
            mean_i = np.mean(i)
            if mean_i == 0:
                output[ind] = np.nan
            else:
                output[ind] = np.log10(np.mean(i))/self.nyquist_rate
        return output

    def median_formatter(self, data):
        """Calculate the median value of the analysis data"""
        values = data[0]

        output = np.empty(len(data))
        for ind, i in enumerate(values):
            median_i = np.median(i)
            if median_i == 0:
                output[ind] = np.nan
            else:
                output[ind] = np.log10(np.median(i))/self.nyquist_rate
        return output
