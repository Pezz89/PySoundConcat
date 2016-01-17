from __future__ import print_function
import numpy as np
import logging
import pdb

from Analysis import Analysis

class SpectralCentroidAnalysis(Analysis):
    """
    An encapsulation of a spectral centroid analysis.
    """

    def __init__(self, AnalysedAudioFile, analysis_group):
        super(SpectralCentroidAnalysis, self).__init__(AnalysedAudioFile, analysis_group, 'SpcCntr')
        # Create logger for module
        self.logger = logging.getLogger(__name__+'.{0}Analysis'.format(self.name))
        # Store reference to the file to be analysed
        self.AnalysedAudioFile = AnalysedAudioFile
        if not self.AnalysedAudioFile.FFT:
            raise ValueError("FFT analysis is required for spectral centroid analysis.")

        self.analysis_group = analysis_group
        self.create_analysis(
            self.create_spccntr_analysis,
            self.AnalysedAudioFile.FFT.analysis['frames'][:],
            self.AnalysedAudioFile.FFT.analysis.attrs['win_size'],
            self.AnalysedAudioFile.samplerate
        )
        self.spccntr_window_count = None

    def hdf5_dataset_formatter(self, analysis_method, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.
        '''
        output = self.create_spccntr_analysis(*args, **kwargs)
        return ({'frames': output}, {})

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
        magnitudes = magnitudes / np.max(magnitudes)
        # Calculate the centre frequency of each rfft bin.
        if output_format == "freq":
            freqs = np.fft.rfftfreq(length, 1.0/samplerate)
        elif output_format == "ind":
            freqs = np.arange(np.size(fft, axis=1))
        else:
            raise ValueError("\'{0}\' is not a valid output "
                             "format.".format(output_format))
        # Calculate the weighted mean
        y = np.sum(magnitudes*freqs, axis=1) / np.sum(magnitudes, axis=1)
        # Convert from index to Hz
        return y
