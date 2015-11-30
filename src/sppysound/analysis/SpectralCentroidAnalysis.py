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
    def create_spccntr_analysis(fft, length, samplerate):
        '''
        Calculate the spectral centroid of the fft frames.

        length: the length of the window used to calculate the FFT.
        samplerate: the samplerate of the audio analysed.
        '''
        # Get the positive magnitudes of each bin.
        magnitudes = np.abs(fft)
        # Calculate the centre frequency of each rfft bin.
        freqs = np.fft.rfftfreq(length, 1.0/samplerate)
        # Calculate the weighted mean
        y = np.sum(magnitudes*freqs, axis=1) / np.sum(magnitudes, axis=1)
        return y
