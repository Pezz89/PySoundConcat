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
        self.create_analysis(self.create_zerox_analysis)

    @staticmethod
    def create_zerox_analysis(samples, *args, **kwargs):
        """Generate zero crossing detections for windows of the signal"""
        # TODO: window across audiofile.
        zero_crossings = np.where(np.diff(np.sign(samples)))[0]
        return zero_crossings

    def hdf5_dataset_formatter(self, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.
        '''
        samples = self.AnalysedAudioFile.read_grain()
        frames = self.create_zerox_analysis(samples, *args, **kwargs)
        return ({'frames': frames}, {})
