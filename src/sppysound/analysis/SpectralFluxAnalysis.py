from __future__ import print_function, division
import numpy as np
import logging
import pdb

from Analysis import Analysis

class SpectralFluxAnalysis(Analysis):
    """
    Spectral flux descriptor class for generation of spectral flux audio
    analysis.

    This descriptor calculates the spectral flux for overlapping grains of an
    AnalysedAudioFile object.  A full definition of spectral flux analysis can
    be found in the documentation.

    Arguments:

    - analysis_group: the HDF5 file group to use for the storage of the
      analysis.

    - config: The configuration module used to configure the analysis
    """

    def __init__(self, AnalysedAudioFile, frames, analysis_group, config=None):
        super(SpectralFluxAnalysis, self).__init__(AnalysedAudioFile,frames, analysis_group, 'SpcFlux')
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
        self.logger.info("Creating Spectral Flux analysis for {0}".format(self.AnalysedAudioFile.name))
        self.create_analysis(
            self.create_spcflux_analysis,
            fft.analysis['frames'],
        )
        self.spcflux_window_count = None

    def hdf5_dataset_formatter(self, analysis_method, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.
        '''
        samplerate = self.AnalysedAudioFile.samplerate
        output = self.create_spcflux_analysis(*args, **kwargs)
        times = self.calc_spcflux_frame_times(output, self.AnalysedAudioFile.frames, samplerate)
        return ({'frames': output, 'times': times}, {})

    @staticmethod
    def create_spcflux_analysis(fft):
        '''
        Calculate the spectral flux of the fft frames.

        length: the length of the window used to calculate the FFT.
        output_format = Choose either "freq" for output in Hz or "ind" for bin
        index output
        '''
        fft = fft[:]
        # Get the positive magnitudes of each bin.
        magnitudes = np.abs(fft)
        if not np.nonzero(magnitudes)[0].size:
            y = np.empty(magnitudes.shape[0])
            y.fill(np.nan)
            return y
        # Roll magnitudes as flux is calculated using the difference between
        # consecutive magnitudes. Rolling allows for quick access to previous
        # magnitude.
        rolled_mags = np.roll(magnitudes, 1, axis=0)[1:]
        sum_of_squares = np.sum((magnitudes[1:]-rolled_mags)**2., axis=1)
        spectral_flux = np.sqrt(sum_of_squares) / (np.size(fft, axis=1))

        return spectral_flux

    @staticmethod
    def calc_spcflux_frame_times(spcflux_frames, sample_frame_count, samplerate):

        """Calculate times for frames using sample size and samplerate."""

        # Get number of frames for time and frequency
        timebins = spcflux_frames.shape[0]
        if not timebins:
            return np.array([])
        # Create array ranging from 0 to number of time frames
        scale = np.arange(timebins+1)
        # divide the number of samples by the total number of frames, then
        # multiply by the frame numbers.
        spcflux_times = (float(sample_frame_count)/float(timebins)) * scale[:-1].astype(float)
        # Divide by the samplerate to give times in seconds
        return spcflux_times

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
