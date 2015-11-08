"""A collection of useful tools for multiple audio analyses."""

from __future__ import division
from scipy.signal import butter, lfilter
import logging

logger = logging.getLogger(__name__)


class ButterFilter:
    def __init__(self, *args, **kwargs):
        self.filtervalues = None
        self.logger = logging.getLogger(__name__ + '.ButterFilter')

    def design_butter(self, cutoff, fs, filtertype='high', order=5):
        """
        Generate a butterworth filter of type and order specified.

        Calculates the cutoff frequency based on the samplerate.
        """
        # Ref: This code has been adapted from:
        # http://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units

        # Calculate nyquist rate
        nyq = 0.5 * fs
        # Calculate the cutoff based on the nyquist rate
        normal_cutoff = cutoff / nyq
        # Calcuate filter coefficients based on parameters
        b, a = butter(order, normal_cutoff, btype=filtertype, analog=False)
        self.filtervalues = b, a


    def filter_butter(self, data):
        """Filter audio using a butterworth filter."""
        # Filter audio using coefficients generated
        y = lfilter(self.filtervalues[0], self.filtervalues[1], data)
        return y
