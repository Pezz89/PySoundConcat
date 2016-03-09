from __future__ import print_function, division
import numpy as np
import logging
import pdb

from Analysis import Analysis

class SpectralCentroidAnalysis(Analysis):
    """
    An encapsulation of a spectral centroid analysis.
    """

    def __init__(self, AnalysedAudioFile, analysis_group, config=None):
        super(SpectralCentroidAnalysis, self).__init__(AnalysedAudioFile, analysis_group, 'SpcCntr')
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
        self.logger.info("Creating Spectral Centroid analysis for {0}".format(self.AnalysedAudioFile.name))
        self.create_analysis(
            self.create_spccntr_analysis,
            fft.analysis['frames'][:],
            fft.analysis.attrs['win_size'],
            self.AnalysedAudioFile.samplerate
        )
        self.spccntr_window_count = None

    def hdf5_dataset_formatter(self, analysis_method, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.
        '''
        samplerate = self.AnalysedAudioFile.samplerate
        output = self.create_spccntr_analysis(*args, **kwargs)
        times = self.calc_spccntr_frame_times(output, self.AnalysedAudioFile.frames, samplerate)
        return ({'frames': output, 'times': times}, {})

    @staticmethod
    def create_spccntr_analysis(fft, length, samplerate, output_format="freq"):
        '''
        Calculate the spectral centroid of the fft frames.

        length: the length of the window used to calculate the FFT.
        samplerate: the samplerate of the audio analysed.
        output_format = Choose either "freq" for output in Hz or "ind" for bin
        index output
        '''
        # Get the positive magnitudes of each bin.
        magnitudes = np.abs(fft)
        # Get the highest magnitude.
        mag_max = np.max(magnitudes)
        if not mag_max:
            y = np.empty(magnitudes.shape[0])
            y.fill(np.nan)
            return y
        # Calculate the centre frequency of each rfft bin.
        if output_format == "freq":
            freqs = np.fft.rfftfreq(length, 1.0/samplerate)
        elif output_format == "ind":
            freqs = np.arange(np.size(fft, axis=1))
        else:
            raise ValueError("\'{0}\' is not a valid output "
                             "format.".format(output_format))
        # Calculate the weighted mean
        y = np.sum(magnitudes*freqs, axis=1) / (np.sum(magnitudes, axis=1)+np.finfo(float).eps)

        # Convert from index to Hz
        return y

    @staticmethod
    def calc_spccntr_frame_times(spccntr_frames, sample_frame_count, samplerate):

        """Calculate times for frames using sample size and samplerate."""

        # Get number of frames for time and frequency
        timebins = spccntr_frames.shape[0]
        # Create array ranging from 0 to number of time frames
        scale = np.arange(timebins+1)
        # divide the number of samples by the total number of frames, then
        # multiply by the frame numbers.
        spccntr_times = (float(sample_frame_count)/float(timebins)) * scale[:-1].astype(float)
        # Divide by the samplerate to give times in seconds
        spccntr_times = spccntr_times / samplerate
        return spccntr_times

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
