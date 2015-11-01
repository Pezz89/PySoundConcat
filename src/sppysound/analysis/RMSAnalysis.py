from __future__ import print_function
import os
import numpy as np
import logging

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

    def __init__(self, AnalysedAudioFile, rmspath):
        self.logger = logging.getLogger(__name__ + '.RMSAnalysis')
        # Store reference to the file to be analysed
        self.AnalysedAudioFile = AnalysedAudioFile

        # Store the path to the rms file if it already exists
        self.rmspath = rmspath

        # Stores the number of RMS window values when calculating the RMS
        # contour
        self.rms_window_count = None

        # Check that the class file has a path to write the rms file to
        if not self.rmspath:
            # If it doesn't then attampt to generate a path based on the
            # location of the database that the object is a part of.
            if not self.AnalysedAudioFile.db_dir:
                # If it isn't part of a database and doesn't have a path then
                # there is no where to write the rms data to.
                raise IOError('Analysed Audio object must have an RMS file'
                              'path or be part of a database')
            self.rmspath = os.path.join(
                self.AnalysedAudioFile.db_dir,
                'rms',
                self.AnalysedAudioFile.name + '.lab'
            )

        # If forcing new analysis creation then delete old analysis and create
        # a new one
        if self.AnalysedAudioFile.force_analysis:
            self.logger.warning("Force re-analysis is enabled. "
                                "deleting: {0}".format(self.rmspath))
            pathops.delete_if_exists(self.rmspath)
            self.rmspath = self.create_rms_analysis()
        else:
            # Check if analysis file already exists.
            try:
                with open(self.rmspath, 'r') as rmsfile:
                    self.logger.info("Analysis already exists. "
                                     "Reading from: {0}".format(self.rmspath))
                    # If an RMS file is provided then count the number of lines
                    # (1 for each window)
                    self.logger.info("Reading RMS file: ",
                                     os.path.relpath(self.rmspath))
                    self.rms_window_count = sum(1 for line in rmsfile)
            except IOError:
                # If it doesn't then generate a new file
                self.rmspath = self.create_rms_analysis()

    def create_rms_analysis(self, window_size=100, window_type='triangle',
                            window_overlap=1):
        """
        Generate an energy contour analysis.

        Calculate the RMS values of windowed segments of the audio file and
        save to disk.
        """
        # TODO: Add low pass filter that relates to the window size
        # Calculate the period of the window in hz
        lowest_freq = 1.0 / window_size
        filter = ButterFilter()
        filter.design_butter(lowest_freq, self.AnalysedAudioFile.samplerate)

        window_size = self.AnalysedAudioFile.ms_to_samps(window_size)
        # Generate a window function to apply to rms windows before analysis
        window_function = self.AnalysedAudioFile.gen_window(window_type,
                                                            window_size)
        i = 0
        try:
            with open(self.rmspath, 'w') as rms_file:
                self.logger.info('Creating RMS file: '+os.path.relpath(self.rmspath))
                self.rms_window_count = 0
                # For all frames in the file, read overlapping windows and
                # calculate the rms values for each window then write the data
                # to file
                while i < self.AnalysedAudioFile.frames():
                    frames = self.AnalysedAudioFile.read_grain(i, window_size)
                    frames = filter.filter_butter(frames)
                    frames = frames * window_function
                    rms = np.sqrt(np.mean(np.square(frames)))
                    rms_file.write('{0} {1:6f}\n'.format(
                        i + int(round(window_size / 2.0)), rms)
                    )
                    i += int(round(window_size / window_overlap))
                    self.rms_window_count += 1

            return self.rmspath
        # If the rms file couldn't be opened then raise an error
        except IOError:
            # TODO: Sort this. This isn't right I don't think.
            return False

    def get_rms_from_file(self, start=0, end=-1):
        """
        Read values from RMS file between start and end points provided (in
        samples)
        """
        self.logger.info("Reading RMS file: {0}".format(self.rmspath))
        # Convert negative numbers to the end of the file offset by that value
        if end < 0:
            end = self.AnalysedAudioFile.frames() + end + 1
        # Create empty array with a size equal to the maximum possible RMS
        # values
        rms_array = np.empty((2, self.rms_window_count))
        # Open the RMS file
        if os.stat(self.rmspath).st_size == 0:
            raise IOError("RMS file is empty")
        with open(self.rmspath, 'r') as rmsfile:
            i = 0
            for line in rmsfile:
                # Split the values and convert to their correct types
                time, value = line.split()
                time = int(time)
                value = float(value)
                # If the values are within the desired range, add them to the
                # array
                if time >= start and time <= end:
                    # The first value will be rounded down to the start
                    if i == 0:
                        time = start
                    rms_array[0][i] = time
                    rms_array[1][i] = value
                    i += 1
            # The last value will be rounded up to the end
            rms_array[0][i] = end
        rms_array = rms_array[:, start:start+i]
        # Interpolate between window values to get per-sample values
        rms_contour = np.interp(np.arange(end), rms_array[0], rms_array[1])
        return rms_contour
