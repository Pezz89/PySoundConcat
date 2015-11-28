from __future__ import print_function
import os
import numpy as np
import logging
from Analysis import Analysis
import pdb

from fileops import pathops

logger = logging.getLogger(__name__)

class ZeroXAnalysis(Analysis):

    """ """

    def __init__(self, AnalysedAudioFile, analysis_group):
        super(ZeroXAnalysis, self).__init__(AnalysedAudioFile, analysis_group, 'name')
        self.logger = logging.getLogger('audiofile.{0}'.format(self.name))
        self.zerox_window_count = None
        self.logger.debug(''.join(("zero-crossing Window Count: ",
                                    str(self.zerox_window_count))))
        self.analysis_group = analysis_group
        self.create_analysis(self.create_zerox_analysis)

        self.zerox_window_count = self.analysis.size


    def create_zerox_analysis(self, *args, **kwargs):
        """Generate zero crossing detections for windows of the signal"""
        zero_crossings = np.where(np.diff(np.sign(
            self.AnalysedAudioFile.read_grain())))[0]
        self.analysis_data.create_dataset('data', data=zero_crossings)
        return zero_crossings
