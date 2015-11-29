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
        self.zerox_window_count = None
        self.logger.debug(''.join(("zero-crossing Window Count: ",
                                    str(self.zerox_window_count))))
        self.analysis_group = analysis_group
        self.create_analysis(self.create_zerox_analysis)

        self.zerox_window_count = self.analysis['data'].size


    def create_zerox_analysis(self, *args, **kwargs):
        """Generate zero crossing detections for windows of the signal"""
        zero_crossings = np.where(np.diff(np.sign(
            self.AnalysedAudioFile.read_grain())))[0]
        self.analysis.create_dataset('data', data=zero_crossings)
        return zero_crossings
