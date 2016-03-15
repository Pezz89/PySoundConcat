from __future__ import print_function, division
import numpy as np
import logging
import pdb
import numpy as np
from numpy.lib import stride_tricks
from Analysis import Analysis
from scipy import signal
from numpy.fft import fft, ifft, fftshift
import warnings

from numpy import polyfit, arange

class F0HarmRatioAnalysis(Analysis):

    """
    The F0 HarmRatio analysis object is a placeholder to allow access to the
    harmonic ratio generated in the f0 analysis.  As a result it does not have
    it's own "create analysis method as other analyses do. it is designed to be
    used for the retreival of the f0 harmonic ratio analysis for matching.

    F0 analysis must be generated for the AnalysedAudioFile in order to use
    this object.
    """

    def __init__(self, AnalysedAudioFile, analysis_group, config=None):
        super(F0HarmRatioAnalysis, self).__init__(AnalysedAudioFile, analysis_group, 'F0HarmRatio')
        self.logger = logging.getLogger(__name__+'.{0}Analysis'.format(self.name))
        # Store reference to the file to be analysed
        self.AnalysedAudioFile = AnalysedAudioFile

        self.analysis_group = analysis_group
        self.logger.info("Initialising F0HarmRatio analysis for {0}".format(self.AnalysedAudioFile.name))

    def get_analysis_grains(self, start, end):
        """
        Retrieve analysis frames for period specified in start and end times.
        arrays of start and end time pairs will produce an array of equivelant
        size containing frames for these times.
        """
        times = self.analysis_group["F0"]["times"][:]
        hr = self.analysis_group["F0"]["harmonic_ratio"][:]
        start = start / 1000
        end = end / 1000
        vtimes = times.reshape(-1, 1)

        selection = np.transpose((vtimes >= start) & (vtimes <= end))

        return ((hr, times), selection)

    @staticmethod
    def calc_F0HarmRatio_frame_times(F0HarmRatioframes, sample_frames, samplerate):

        """Calculate times for frames using sample size and samplerate."""

        # Get number of frames for time and frequency
        timebins = F0HarmRatioframes.shape[0]
        # Create array ranging from 0 to number of time frames
        scale = np.arange(timebins+1)
        # divide the number of samples by the total number of frames, then
        # multiply by the frame numbers.
        F0HarmRatio_times = (sample_frames.shape[0]/timebins) * scale[:-1]
        # Divide by the samplerate to give times in seconds
        F0HarmRatio_times = F0HarmRatio_times / samplerate
        return F0HarmRatio_times

    def analysis_formatter(self, data, selection, format):
        """Calculate the average analysis value of the grain using the match format specified."""
        harm_ratio, times = data
        # Get indexes of all valid frames (that aren't nan)
        valid_inds = np.isfinite(harm_ratio)

        format_style_dict = {
            'mean': np.mean,
            'median': np.median,
            'log2_mean': self.log2_mean,
            'log2_median': self.log2_median,
        }

        # For debugging apply along axis:
        #for ind, i in enumerate(selection):
        #    output[ind] = self.formatter_func(i, frames, valid_inds, harm_ratio, formatter=format_style_dict[format])

        output = np.apply_along_axis(
            self.formatter_func,
            1,
            selection,
            harm_ratio,
            valid_inds,
            formatter=format_style_dict[format]
        )

        return output

