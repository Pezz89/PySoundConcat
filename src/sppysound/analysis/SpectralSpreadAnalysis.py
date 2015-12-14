from __future__ import print_function
import numpy as np
import logging
import pdb

from Analysis import Analysis

class SpectralSpreadAnalysis(Analysis):
    """
    An encapsulation of a spectral centroid analysis.
    """

    def __init__(self, AnalysedAudioFile, analysis_group):
        super(SpectralSpreadAnalysis, self).__init__(AnalysedAudioFile, analysis_group, 'SpcSprd')
        self.logger = logging.getLogger(__name__+'.{0}Analysis'.format(self.name))
        # Store reference to the file to be analysed
        self.AnalysedAudioFile = AnalysedAudioFile
        if not self.AnalysedAudioFile.SpectralCentroid:
            raise ValueError("Spectral Centroid analysis is required for "
                             "spectral spread analysis.")
        if not self.AnalysedAudioFile.FFT:
            raise ValueError("FFT analysis is required for spectral spread "
                             "analysis.")

        self.analysis_group = analysis_group
        self.create_analysis(
            self.AnalysedAudioFile.FFT.analysis['frames'][:],
            self.AnalysedAudioFile.SpectralCentroid.analysis['frames'][:],
            self.AnalysedAudioFile.FFT.analysis.attrs['win_size'],
            self.AnalysedAudioFile.samplerate
        )
        self.spccntr_window_count = None

    def hdf5_dataset_formatter(self, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.
        '''
        output = self.create_spcsprd_analysis(*args, **kwargs)
        return ({'frames': output}, {})

    @staticmethod
    def create_spcsprd_analysis(fft, spectral_centroid, length, samplerate):
        '''
        Calculate the spectral centroid of the fft frames.

        fft: Real fft frames.
        spectral_centroid: spectral centroid frames.
        length: the length of the window used to calculate the FFT.
        samplerate: the samplerate of the audio analysed.
        '''
        # Get the positive magnitudes of each bin.
        magnitudes = np.abs(fft)
        # Calculate the centre frequency of each rfft bin.
        freqs = np.fft.rfftfreq(length, 1.0/samplerate)


        pdb.set_trace()
        a = np.power(freqs-spectral_centroid)
        mag_sqrd = np.power(magnitudes)
        # Calculate the weighted mean
        y = np.sqrt(np.sum(a*mag_sqrd, axis=1) / np.sum(mag_sqrd, axis=1))

        pdb.set_trace()
        return y
