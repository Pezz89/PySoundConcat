from __future__ import print_function
import os
import numpy as np
import logging

from fileops import pathops


class ZeroXAnalysis:

    """ """

    def __init__(self, AnalysedAudioFile, analysis_group):
        self.logger = logging.getLogger(__name__ + '.ZeroXAnalysis')
        self.AnalysedAudioFile = AnalysedAudioFile
        self.zerox_window_count = None

        # Store the path to the FFT file if it already exists
        try:
            self.analysis_data = analysis_group.create_group('zerox')
        except ValueError:
            self.logger.warning("Zero-crossing analysis group already exists")
            self.analysis_data = analysis_group['zerox']

        # If forcing new analysis creation then delete old analysis and create
        # a new one
        if self.AnalysedAudioFile.force_analysis:
            self.logger.warning("Force re-analysis is enabled. "
                                "deleting: {0}".format(self.analysis_data.name))
            # Delete all pre-existing data in database.
            for i in self.analysis_data.iterkeys():
                del self.analysis_data[i]
            self.zerox_analysis = self.create_zerox_analysis()
        else:
            # Check if analysis file already exists.
            try:
                self.zerox_analysis = self.analysis_data['data']
                self.logger.info("Analysis already exists. "
                                 "Reading from: {0}".format(self.analysis_data.name))
                # If an zero-crossing file is provided then count the number of lines
                # (1 for each window)
                self.logger.info(''.join(("Reading zero-crossing data: "
                                          "(HDF5 File)",
                                          self.analysis_data.name)))
                self.zerox_window_count = self.zerox_analysis.size
                self.logger.debug(''.join(("zero-crossing Window Count: ",
                                           str(self.zerox_window_count))))
            except KeyError:
                # If it doesn't then generate a new file
                self.zerox_analysis = self.create_zerox_analysis()


    def create_zerox_analysis(self):
        """Generate zero crossing detections for windows of the signal"""
        zero_crossings = np.where(np.diff(np.sign(
            self.AnalysedAudioFile.read_grain())))
        self.analysis_data.create_dataset('data', data=zero_crossings)
        return zero_crossings
