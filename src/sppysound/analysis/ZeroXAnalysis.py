from __future__ import print_function
import numpy as np
import logging
from Analysis import Analysis
import pdb

logger = logging.getLogger(__name__)


class ZeroXAnalysis(Analysis):

    """Zero-crossing analysis class. """

    def __init__(self, AnalysedAudioFile, analysis_group):
        super(ZeroXAnalysis, self).__init__(AnalysedAudioFile, analysis_group, 'ZeroCrossing')
        self.logger = logging.getLogger(__name__+'.{0}Analysis'.format(self.name))
        self.analysis_group = analysis_group
        self.logger.info("Creating zero crossing analysis for {0}".format(self.AnalysedAudioFile.name))
        frames = self.AnalysedAudioFile.read_grain()
        self.create_analysis(frames)

    @staticmethod
    def create_zerox_analysis(samples, *args, **kwargs):
        """Generate zero crossing detections for windows of the signal"""
        # TODO: window across audiofile.
        zero_crossings = np.where(np.diff(np.sign(samples)))[0]
        return zero_crossings

    @staticmethod
    def calc_zerox_frame_times(zerox_frames, sample_frames, samplerate):

        """Calculate times for frames using sample size and samplerate."""

        # Get number of frames for time and frequency
        timebins = zerox_frames.shape[0]
        # Create array ranging from 0 to number of time frames
        scale = np.arange(timebins+1)
        # divide the number of samples by the total number of frames, then
        # multiply by the frame numbers.
        zerox_times = (sample_frames.shape[0]/timebins) * scale[:-1]
        # Divide by the samplerate to give times in seconds
        zerox_times = zerox_times / samplerate
        return zerox_times

    def get_analysis_grains(self, start, end):
        """
        Retrieve analysis frames for period specified in start and end times.
        arrays of start and end time pairs will produce an array of equivelant
        size containing frames for these times.
        """
        times = self.analysis_group["ZeroCrossing"]["times"][:]
        start = start / 1000
        end = end / 1000
        vtimes = times.reshape(-1, 1)

        selection = np.transpose((vtimes >= start) & (vtimes <= end))

        np.set_printoptions(threshold=np.nan)

        grain_data = []
        for grain in selection:
            grain_data.append((self.analysis_group["ZeroCrossing"]["frames"][grain], times[grain]))

        return grain_data

    def hdf5_dataset_formatter(self, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.
        '''
        samples = self.AnalysedAudioFile.read_grain()
        samplerate = self.AnalysedAudioFile.samplerate
        output = self.create_zerox_analysis(*args, **kwargs)
        times = self.calc_zerox_frame_times(output, args[0], samplerate)
        return ({'frames': output, 'times': times}, {})
