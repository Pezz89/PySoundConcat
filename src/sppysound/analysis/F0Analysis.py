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

class F0Analysis(Analysis):

    """
    An encapsulation of the F0 analysis of an AnalysedAudioFile.

    On initialization, the F0 analysis is either created, or a pre existing
    file already exists.
    In either case, once the file is generated, it's values can be obtained
    through use of the get_f0_from_file method

    Note: Due to the large size of F0 analysis it is not stored in a class
    member as other such analyses are. Use get_rms_from_file.
    """

    def __init__(self, AnalysedAudioFile, analysis_group, config=None):
        super(F0Analysis, self).__init__(AnalysedAudioFile, analysis_group, 'F0')
        self.logger = logging.getLogger(__name__+'.{0}Analysis'.format(self.name))
        # Store reference to the file to be analysed
        self.AnalysedAudioFile = AnalysedAudioFile

        self.nyquist_rate = self.AnalysedAudioFile.samplerate / 2.

        self.analysis_group = analysis_group
        frames = self.AnalysedAudioFile.read_grain()
        self.logger.info("Creating F0 analysis for {0}".format(self.AnalysedAudioFile.name))
        self.create_analysis(frames, self.AnalysedAudioFile.samplerate)
        if config:
            self.threshold = config.f0["threshold"]
        else:
            self.threshold = None

    def get_analysis_grains(self, start, end):
        """
        Retrieve analysis frames for period specified in start and end times.
        arrays of start and end time pairs will produce an array of equivelant
        size containing frames for these times.
        """
        times = self.analysis_group["F0"]["times"][:]
        start = start / 1000
        end = end / 1000
        vtimes = times.reshape(-1, 1)

        selection = np.transpose((vtimes >= start) & (vtimes <= end))

        grain_data = [[], [], []]
        for grain in selection:
            grain_data[0].append(self.analysis_group["F0"]["frames"][grain])
            grain_data[1].append(self.analysis_group["F0"]["harmonic_ratio"][grain])
            grain_data[2].append(times[grain])

        return grain_data

    @staticmethod
    def create_f0_analysis(
        frames,
        samplerate,
        window_size=512,
        overlapFac=0.5,
        m0=None,
        M=None,
    ):
        """
        # function [HR, f0, Gamma] = feature_harmonic(window, Fs, M, m0)
        # This function computes the harmonic ratio and fundamental frequency of a
        # window
        #
        # ARGUMENTS
        # - window: the samples of the window
        # - Fs:     the sampling frequency
        # - M:      the maximum T0 (optional)
        # - m0:     the minimum T0 (optional)
        #
        # RETURNS:
        # - HR:     harmonic ratio
        # - f0:     fundamental frequency
        #
        # (c) 2014 T. Giannakopoulos, A. Pikrakis
        """

        if not M:
            M=round(0.016*samplerate)

        hopSize = int(window_size - np.floor(overlapFac * window_size))

        # zeros at beginning (thus center of 1st window should be for sample nr. 0)
        samples = frames
        #samples = np.concatenate((np.zeros(np.floor(window_size/2.0)), frames))

        # cols for windowing
        cols = np.ceil((len(samples) - window_size) / float(hopSize)) + 1
        # zeros at end (thus samples can be fully covered by frames)
        samples = np.concatenate((samples, np.zeros(window_size)))

        frames = stride_tricks.as_strided(
            samples,
            shape=(cols, window_size),
            strides=(samples.strides[0]*hopSize, samples.strides[0])
        ).copy()

        def feature_zcr(window):
            # function  Z = feature_zcr(window);
            #
            # This function calculates the zero crossing rate of an audio frame.
            #
            # ARGUMENTS:
            # - window: 	an array that contains the audio samples of the input frame
            #
            # RETURN:
            # - Z:		the computed zero crossing rate value
            #

            window2 = np.zeros(window.size)
            window2[1:-1] = window[0:-2]
            Z = (1/(2*window.size)) * np.sum(np.abs(np.sign(window)-np.sign(window2)))
            return Z

        def parabolic(f, x):
            """
            #Quadratic interpolation for estimating the true position of an
            #inter-sample maximum when nearby samples are known.

            f is a vector and x is an index for that vector.

            Returns (vx, vy), the coordinates of the vertex of a parabola that
            goes through point x and its two neighbors.

            Example:
            Defining a vector f with a local maximum at index 3 (= 6), find
            local maximum if points 2, 3, and 4 actually defined a parabola.

            In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

            In [4]: parabolic(f, argmax(f))
            Out[4]: (3.2142857142857144, 6.1607142857142856)

            """
            if x >= f.size-1 or x <= 2:
                return x, f[x]

            xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
            yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
            return (xv, yv)

        def autocorr(x):
            """
            #FFT based autocorrelation function, which is faster than numpy.correlate
            #Operates on muti-dimensional arrays on a per element basis.
            #Ref: http://stackoverflow.com/questions/4503325/autocorrelation-of-a-multidimensional-array-in-numpy
            """
            length = np.size(x, axis=1)
            # x is supposed to be an array of sequences, of shape (totalelements, length)
            fftx = fft(x, n=(length*2-1), axis=1)
            ret = ifft(fftx * np.conjugate(fftx), axis=1).real
            ret = fftshift(ret, axes=1)
            return ret

        def per_frame_f0(frames, m0, M):
            if not frames.any():
                HR = 0
                f0 = 0
                return f0, HR

            R=autocorr([frames])
            R = R[0]
            g=R[frames.size]

            R=R[frames.size-1:]

            if not m0:
                # estimate m0 (as the first zero crossing of R)
                m0 = np.argmin(np.diff(np.sign(R[1:])))+1
            if m0 == 1:
                m0 = R.size
            if M > R.size:
                M = R.size
            Gamma = np.zeros(M)

            CSum = np.cumsum(frames*frames)

            Gamma[m0:M] = R[m0:M] / (np.sqrt([g*CSum[-m0:-M:-1]])+np.finfo(float).eps)

            Z = feature_zcr(Gamma)
            if Z > 0.15:
                HR = 0
                f0 = 0
            else:
                # compute T0 and harmonic ratio:
                if np.isnan(Gamma).any():
                    HR=0
                    f0 = 0
                else:
                    blag = np.argmax(Gamma)
                    HR = Gamma[blag]
                    interp, HR = parabolic(Gamma, blag)
                    if not interp:
                        f0 = 0
                        HR = 0
                    else:
                        # get fundamental frequency:
                        f0 = samplerate / interp
            if f0 > samplerate/2:
                raise ValueError("F0 value ({0}) is above the nyquist rate "
                                 "({1}). This shouldn't happen...".format(f0,
                                 samplerate/2))
            if HR >= 1:
                HR = 1
            return (f0, HR)

        output = np.empty((frames.shape[0], 2))
        for ind, i in enumerate(frames):
            output[ind] = per_frame_f0(i, m0, M)

        # output = np.apply_along_axis(per_frame_f0, 1, frames, m0, M)
        return output

    def hdf5_dataset_formatter(self, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.
        '''
        samplerate = self.AnalysedAudioFile.samplerate
        data = self.create_f0_analysis(*args, **kwargs)
        f0 = data[:, 0]
        harmonic_ratio = data[:, 1]
        f0_times = self.calc_f0_frame_times(f0, args[0], samplerate)
        return ({'frames': f0, 'harmonic_ratio': harmonic_ratio, 'times': f0_times}, {})

    @staticmethod
    def calc_f0_frame_times(f0frames, sample_frames, samplerate):

        """Calculate times for frames using sample size and samplerate."""

        # Get number of frames for time and frequency
        timebins = f0frames.shape[0]
        # Create array ranging from 0 to number of time frames
        scale = np.arange(timebins+1)
        # divide the number of samples by the total number of frames, then
        # multiply by the frame numbers.
        f0_times = (sample_frames.shape[0]/timebins) * scale[:-1]
        # Divide by the samplerate to give times in seconds
        f0_times = f0_times / samplerate
        return f0_times

    def mean_formatter(self, data):
        if not self.threshold:
            raise ValueError("Threshold not set for F0Analysis object.")
        frames = data[0]
        confidence = data[1]
        output = np.empty(len(frames))
        for i, (frame, conf) in enumerate(zip(frames, confidence)):
            if not frame.size:
                output[i] = 0
                continue
            med_conf = np.mean(conf)
            if med_conf > self.threshold:
                output[i] = np.log10(np.mean(frame[conf > self.threshold]))/self.nyquist_rate
            else:
                output[i] = 0

        return output

    def median_formatter(self, data):
        if not self.threshold:
            raise ValueError("Threshold not set for F0Analysis object.")
        frames = data[0]
        confidence = data[1]
        output = np.empty(len(frames))
        for i, (frame, conf) in enumerate(zip(frames, confidence)):
            if not frame.size:
                output[i] = 0
                continue
            med_conf = np.median(conf)
            if med_conf > self.threshold:
                output[i] = np.log10(np.median(frame[conf > self.threshold]))/self.nyquist_rate
            else:
                output[i] = 0

        return output
