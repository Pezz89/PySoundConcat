from __future__ import print_function, division
import numpy as np
import logging
import pdb
import numpy as np
from numpy.lib import stride_tricks
from Analysis import Analysis
from scipy import signal
from numpy.fft import fft, ifft, fftshift
import multirate
import warnings
import matplotlib.pyplot as plt

import pYIN.pYINmain as pYINmain
from numpy import polyfit, arange

class F0Analysis(Analysis):

    """
    F0 analysis descriptor class for generation of fundamental frequency
    estimation.

    This descriptor calculates the fundamental frequency for overlapping grains
    of an AnalysedAudioFile object.  A full definition of F0 analysis can be
    found in the documentation.

    Arguments:

    - analysis_group: the HDF5 file group to use for the storage of the
      analysis.

    - config: The configuration module used to configure the analysis
    """

    def __init__(self, AnalysedAudioFile, frames, analysis_group, config=None):
        super(F0Analysis, self).__init__(AnalysedAudioFile,frames, analysis_group, 'F0')
        self.logger = logging.getLogger(__name__+'.{0}Analysis'.format(self.name))
        # Store reference to the file to be analysed
        self.AnalysedAudioFile = AnalysedAudioFile

        self.nyquist_rate = self.AnalysedAudioFile.samplerate / 2.

        if config:
            self.window_size = config.f0["window_size"]
            self.overlap = 1. / config.f0["overlap"]
            self.threshold = config.f0["ratio_threshold"]
        else:
            self.window_size=512
            self.overlap = 0.5
            self.threshold = 0.

        self.analysis_group = analysis_group
        self.logger.info("Creating F0 analysis for {0}".format(self.AnalysedAudioFile.name))

        self.hopSize = int(np.floor(self.overlap * self.window_size))
        self.pYinInst = pYINmain.PyinMain()
        self.pYinInst.initialise(channels = 1, inputSampleRate = self.AnalysedAudioFile.samplerate, stepSize = self.hopSize, blockSize = self.window_size,
                    lowAmp = 0.00, onsetSensitivity = 0.9, pruneThresh = 0.0)

        self.create_analysis(
            frames,
            self.AnalysedAudioFile.samplerate,
            window_size=self.window_size,
            overlapFac=self.overlap,
            threshold=config.f0["ratio_threshold"]
        )

    def get_analysis_grains(self, start, end):
        """
        Retrieve analysis frames for period specified in start and end times.
        arrays of start and end time pairs will produce an array of equivelant
        size containing frames for these times.
        """
        times = self.analysis_group["F0"]["times"][:]
        frames = self.analysis_group["F0"]["frames"][:]
        vtimes = times.reshape(-1, 1)

        selection = np.transpose((vtimes >= start) & (vtimes <= end))
        if not selection.any():
            frame_center = start + (end-start)/2.
            closest_frames = np.abs(vtimes-frame_center).argsort()[:2]
            selection[closest_frames] = True

        return ((frames, times), selection)

    def create_f0_analysis(
        self,
        frames,
        samplerate,
        window_size=512,
        overlapFac=0.5,
        threshold=0.0,
        m0=None,
        M=None,
    ):
        """
        Generate F0 contour analysis.

        Calculate the frequency and harmonic ratio values of windowed segments
        of the audio file and save to disk.
        """

        if hasattr(frames, '__call__'):
            frames = frames()
        if not M:
            M=int(round(0.016*samplerate))



        # zeros at beginning (thus center of 1st window should be for sample nr. 0)
        samples = frames
        #samples = np.concatenate((np.zeros(np.floor(window_size/2.0)), frames))

        # cols for windowing
        cols = np.ceil((len(samples) - window_size) / float(self.hopSize)) + 1
        # zeros at end (thus samples can be fully covered by frames)
        samples = np.concatenate((samples, np.zeros(window_size)))

        frames = stride_tricks.as_strided(
            samples,
            shape=(cols, window_size),
            strides=(samples.strides[0]*self.hopSize, samples.strides[0])
        )

        for ind, frame in enumerate(frames):
            fs = self.pYinInst.process(frame)
        output = self.pYinInst.getSmoothedPitchTrack()
        output[output < 0] = np.nan
        '''
        if self.AnalysedAudioFile.name == 'human.002.001.wav':
            pdb.set_trace()
        '''

        return output

    def hdf5_dataset_formatter(self, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.
        '''
        samplerate = self.AnalysedAudioFile.samplerate
        frames = args[0]
        # frames = multirate.interp(frames, 4)
        f0 = self.create_f0_analysis(frames, samplerate, **kwargs)
        f0_times = self.calc_f0_frame_times(f0, frames, samplerate)
        return ({'frames': f0, 'times': f0_times}, {})

    @staticmethod
    def calc_f0_frame_times(f0frames, sample_frames, samplerate):

        """Calculate times for frames using sample size and samplerate."""

        if hasattr(sample_frames, '__call__'):
            sample_frames = sample_frames()
        # Get number of frames for time and frequency
        timebins = f0frames.shape[0]
        # Create array ranging from 0 to number of time frames
        scale = np.arange(timebins+1)
        # divide the number of samples by the total number of frames, then
        # multiply by the frame numbers.
        f0_times = (sample_frames.shape[0]/timebins) * scale[:-1]
        # Divide by the samplerate to give times in seconds
        return f0_times

    def analysis_formatter(self, data, selection, format):
        """Calculate the average analysis value of the grain using the match format specified."""
        frames, times = data
        # Get indexes of all valid frames (that aren't nan)
        valid_inds = np.isfinite(frames)

        format_style_dict = {
            'mean': np.mean,
            'median': np.median,
            'log2_mean': self.log2_mean,
            'log2_median': self.log2_median,
        }

        if not selection.size:
            # TODO: Add warning here
            return np.nan

        #for ind, i in enumerate(selection):
        #    output[ind] = self.formatter_func(i, frames, valid_inds, harm_ratio, formatter=format_style_dict[format])

        try:
            output = np.apply_along_axis(
                self.formatter_func,
                1,
                selection,
                frames,
                valid_inds,
                formatter=format_style_dict[format]
            )/self.nyquist_rate
        except IndexError:
            pdb.set_trace()

        return output

