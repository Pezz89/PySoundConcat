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
        try:
            spccntr = self.AnalysedAudioFile.analyses["spccntr"]
        except KeyError:
            raise KeyError("Spectral Centroid analysis is required for "
                             "spectral spread analysis.")
        try:
            fft = self.AnalysedAudioFile.analyses["fft"]
        except KeyError:
            raise KeyError("FFT analysis is required for spectral spread "
                             "analysis.")

        self.analysis_group = analysis_group
        self.logger.info("Creating Spectral Spread analysis for {0}".format(self.AnalysedAudioFile.name))
        self.create_analysis(
            fft.analysis['frames'][:],
            spccntr.analysis['frames'][:],
            fft.analysis.attrs['win_size'],
            self.AnalysedAudioFile.samplerate
        )
        self.spccntr_window_count = None

    def get_analysis_grains(self, start, end):
        """
        Retrieve analysis frames for period specified in start and end times.
        arrays of start and end time pairs will produce an array of equivelant
        size containing frames for these times.
        """
        times = self.analysis_group["SpcSprd"]["times"][:]
        start = start / 1000
        end = end / 1000
        vtimes = times.reshape(-1, 1)

        selection = np.transpose((vtimes >= start) & (vtimes <= end))

        np.set_printoptions(threshold=np.nan)

        grain_data = []
        for grain in selection:
            grain_data.append((self.analysis_group["SpcSprd"]["frames"][grain], times[grain]))

        return grain_data

    def hdf5_dataset_formatter(self, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.
        '''
        samplerate = self.AnalysedAudioFile.samplerate
        output = self.create_spcsprd_analysis(*args, **kwargs)
        times = self.calc_spcsprd_frame_times(output, args[0], samplerate)
        return ({'frames': output, 'times': times}, {})

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
        elif output_format == "freq":
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

    @staticmethod
    def calc_spcsprd_frame_times(spcsprd_frames, sample_frames, samplerate):

        """Calculate times for frames using sample size and samplerate."""

        # Get number of frames for time and frequency
        timebins = spcsprd_frames.shape[0]
        # Create array ranging from 0 to number of time frames
        scale = np.arange(timebins+1)
        # divide the number of samples by the total number of frames, then
        # multiply by the frame numbers.
        spcsprd_times = (sample_frames.shape[0]/timebins) * scale[:-1]
        # Divide by the samplerate to give times in seconds
        spcsprd_times = spcsprd_times / samplerate
        return spcsprd_times
