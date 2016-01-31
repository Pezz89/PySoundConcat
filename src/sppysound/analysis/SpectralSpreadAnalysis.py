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
    def create_spcsprd_analysis(fft, spectral_centroid, length, samplerate, output_format = "ind"):
        '''
        Calculate the spectral centroid of the fft frames.

        fft: Real fft frames.
        spectral_centroid: spectral centroid frames (in index format).
        length: the length of the window used to calculate the FFT.
        samplerate: the samplerate of the audio analysed.
        '''
        # Get the positive magnitudes of each bin.
        magnitudes = np.abs(fft)
        magnitudes = magnitudes / np.max(magnitudes);
        # Get the index for each bin
        if output_format == "ind":
            freqs = np.arange(np.size(fft, axis=1))
        if output_format == "freq":
            freqs = np.fft.rfftfreq(length, 1.0/samplerate)
        else:
            raise ValueError("\'{0}\' is not a valid output "
                             "format.".format(output_format))

        spectral_centroid = np.vstack(spectral_centroid)

        a = np.power(freqs-spectral_centroid, 2)
        mag_sqrd = np.power(magnitudes, 2)
        # Calculate the weighted mean
        y = np.sqrt(np.sum(a*mag_sqrd, axis=1) / np.sum(mag_sqrd, axis=1))

        return y
