from __future__ import print_function
import os
import numpy as np
import logging
from scipy import signal
from numpy.lib import stride_tricks
import pdb


from AnalysisTools import ButterFilter
from fileops import pathops

logger = logging.getLogger(__name__)


class RMSAnalysis:

    """
    An encapsulation of the RMS analysis of an AnalysedAudioFile.

    On initialization, the RMS analysis is either created, or a pre existing
    file already exists.
    In either case, once the file is generated, it's values can be obtained
    through use of the get_rms_from_file method

    Note: Due to the large size of RMS analysis it is not stored in a class
    member as other such analyses are. Use get_rms_from_file.
    """

    def __init__(self, AnalysedAudioFile, analysis_group):
        self.logger = logging.getLogger(__name__ + '.RMSAnalysis')
        # Store reference to the file to be analysed
        self.AnalysedAudioFile = AnalysedAudioFile

        # Stores the number of RMS window values when calculating the RMS
        # contour
        self.rms_window_count = None

        try:
            self.analysis_data = analysis_group.create_group('rms')
        except ValueError:
            self.logger.warning("rms analysis group already exists")
            self.analysis_data = analysis_group['rms']

        # If forcing new analysis creation then delete old analysis and create
        # a new one
        if self.AnalysedAudioFile.force_analysis:
            self.logger.warning("Force re-analysis is enabled. "
                                "deleting: {0}".format(self.analysis_data.name))
            # Delete all pre-existing data in database.
            for i in self.analysis_data.iterkeys():
                del self.analysis_data[i]
            self.rms_analysis = self.create_rms_analysis()
        else:
            # Check if analysis file already exists.
            try:
                self.rms_analysis = self.analysis_data['data']
                self.logger.info("Analysis already exists. "
                                 "Reading from: {0}".format(self.analysis_data.name))
                # If an RMS file is provided then count the number of lines
                # (1 for each window)
                self.logger.info(''.join(("Reading RMS data: (HDF5 File)",
                                          self.analysis_data.name)))
                self.rms_window_count = self.rms_analysis.size
                self.logger.debug(''.join(("RMS Window Count: ",
                                           str(self.rms_window_count))))
            except KeyError:
                # If it doesn't then generate a new file
                self.rms_analysis = self.create_rms_analysis()


    def create_rms_analysis(self, window_size=100,
                            window=signal.triang,
                            overlapFac=0.5):
        """
        Generate an energy contour analysis.

        Calculate the RMS values of windowed segments of the audio file and
        save to disk.
        """
        # Calculate the period of the window in hz
        lowest_freq = 1.0 / window_size
        # Filter frequencies lower than the period of the window
        # filter = ButterFilter()
        # filter.design_butter(lowest_freq, self.AnalysedAudioFile.samplerate)

        window_size = self.AnalysedAudioFile.ms_to_samps(window_size)
        # Generate a window function to apply to rms windows before analysis

        frames = self.AnalysedAudioFile.read_grain()
        # TODO: Fix filter
        # frames = filter.filter_butter(frames)

        win = window(window_size)
        hopSize = int(window_size - np.floor(overlapFac * window_size))

        # zeros at beginning (thus center of 1st window should be for sample nr. 0)
        samples = np.append(np.zeros(np.floor(window_size/2.0)), frames)

        # cols for windowing
        cols = np.ceil((len(samples) - window_size) / float(hopSize)) + 1
        # zeros at end (thus samples can be fully covered by frames)
        samples = np.append(samples, np.zeros(window_size))

        frames = stride_tricks.as_strided(
            samples,
            shape=(cols, window_size),
            strides=(samples.strides[0]*hopSize, samples.strides[0])
        ).copy()

        frames *= win
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        rms_times = self.calc_rms_frame_times(
            rms,
            samples,
            self.AnalysedAudioFile.samplerate
        )

        self.analysis_data.create_dataset('data', data=rms)
        self.analysis_data.create_dataset('times', data=rms_times)

        return rms, rms_times

    def calc_rms_frame_times(self, rmsframes, sample_frames, samplerate):

        """Calculate times for frames using sample size and samplerate."""

        # Get number of frames for time and frequency
        timebins = rmsframes.shape[0]
        # Create array ranging from 0 to number of time frames
        scale = np.arange(timebins+1)
        # divide the number of samples by the total number of frames, then
        # multiply by the frame numbers.
        rms_times = (sample_frames.shape[0]/timebins) * scale[:-1]
        # Divide by the samplerate to give times in seconds
        rms_times = rms_times / samplerate
        return rms_times
