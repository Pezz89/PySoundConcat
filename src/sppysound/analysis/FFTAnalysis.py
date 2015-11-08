"""
Module for creating an FFT analysis of audio.

Ref: Code adapted from:
http://www.frank-zalkow.de/en/code-snippets/create-audio-spectrograms-with-python.html?ckattempt=1
"""
from __future__ import print_function, division
import logging
from fileops import pathops
import numpy as np
from numpy.lib import stride_tricks
import os
from matplotlib import pyplot as plt
from AnalysisTools import ButterFilter
import pdb

logger = logging.getLogger(__name__)


class FFTAnalysis:
    """An encapsulation of the FFT analysis of audio."""

    def __init__(self, AnalysedAudioFile, fftpath):
        self.logger = logging.getLogger(__name__ + '.FFTAnalysis')
        # Store reference to the file to be analysed
        self.AnalysedAudioFile = AnalysedAudioFile

        # Store the path to the FFT file if it already exists
        self.fftpath = fftpath

        # Stores the number of FFT window values when calculating the FFT
        # contour
        self.fft_window_count = None

        # Check that the class file has a path to write the fft file to
        if not self.fftpath:
            # If it doesn't then attampt to generate a path based on the
            # location of the database that the object is a part of.
            if not self.AnalysedAudioFile.db_dir:
                # If it isn't part of a database and doesn't have a path then
                # there is no where to write the fft data to.
                raise IOError('Analysed Audio object must have an FFT file'
                              'path or be part of a database')
            self.fftpath = os.path.join(
                self.AnalysedAudioFile.db_dir,
                'fft',
                self.AnalysedAudioFile.name + '.npz'
            )

        # If forcing new analysis creation then delete old analysis and create
        # a new one
        if self.AnalysedAudioFile.force_analysis:
            self.logger.warning("Force re-analysis is enabled. "
                                "deleting: {0}".format(self.fftpath))
            pathops.delete_if_exists(self.fftpath)
            self.fftpath = self.create_fft_analysis()
            self.fft_analysis = np.load(self.fftpath, mmap_mode='r')
        else:
            # Check if analysis file already exists.
            try:
                self.fft_analysis = np.load(self.fftpath, mmap_mode='r')
                self.logger.info("Analysis already exists. "
                                 "Reading from: {0}".format(self.fftpath))
                # If an FFT file is provided then count the number of lines
                # (1 for each window)
                self.logger.info(''.join(("Reading FFT file: ",
                                    os.path.relpath(self.fftpath))))
                self.fft_window_count = self.fft_analysis.size
                self.logger.debug(''.join(("FFT Window Count: ",
                                           str(self.fft_window_count))))
            except IOError:
                # If it doesn't then generate a new file
                self.fftpath = self.create_fft_analysis()
                self.fft_analysis = np.load(self.fftpath, mmap_mode='r')

    def create_fft_analysis(self, window_size=100, window_overlap=2,
                            window_type='hanning'):
        """Create a spectral analysis for overlapping frames of audio."""
        # Calculate the period of the window in hz
        lowest_freq = 1.0 / window_size
        # Filter frequencies lower than the period of the window
        filter = ButterFilter()
        filter.design_butter(lowest_freq, self.AnalysedAudioFile.samplerate)

        window_size = self.AnalysedAudioFile.ms_to_samps(window_size)
        try:
            frames = self.AnalysedAudioFile.read_grain()
            # frames = filter.filter_butter(frames)
            stft = self.stft(frames, window_size, overlapFac=1/window_overlap)
            frame_times = self.calc_fft_frame_times(
                stft,
                frames,
                self.AnalysedAudioFile.samplerate
            )
            np.savez(self.fftpath, frames=stft, times=frame_times)

            return self.fftpath
        # If the fft file couldn't be opened then raise an error
        except IOError:
            # TODO: Sort this. This isn't right I don't think.
            return None

    def stft(self, sig, frameSize, overlapFac=0.5, window=np.hanning):
        """Short time fourier transform of audio signal."""
        win = window(frameSize)
        hopSize = int(frameSize - np.floor(overlapFac * frameSize))

        # zeros at beginning (thus center of 1st window should be for sample nr. 0)
        samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)
        # cols for windowing

        cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
        # zeros at end (thus samples can be fully covered by frames)
        samples = np.append(samples, np.zeros(frameSize))

        frames = stride_tricks.as_strided(
            samples,
            shape=(cols, frameSize),
            strides=(samples.strides[0]*hopSize, samples.strides[0])
        ).copy()

        frames *= win

        return np.fft.rfft(frames)

    def logscale_spec(self, spec, sr=44100, factor=20.):
        """Scale frequency axis logarithmically."""
        # Get a count of times and frequencies from fft frames
        timebins, freqbins = np.shape(spec)

        # Create array from 0 to 1 with values for each frequency bin.
        # Scale by a power of the factor provided.
        scale = np.linspace(0, 1, freqbins) ** factor
        # Scale to the number of frequency bins
        scale *= (freqbins-1)/max(scale)
        # Round to the nearest whole number and reduce to only unique numbers.
        scale = np.unique(np.round(scale))

        # Create a new complex number array with the number of time frames and
        # the new number of frequency bins
        newspec = np.complex128(np.zeros([timebins, len(scale)]))
        # For each of the frequency bins
        for i in range(0, len(scale)):
            # If it is the highest frequency bin...
            if i == len(scale)-1:
                # Sum all frequency bins from the scale index upwards
                newspec[:, i] = np.sum(spec[:, scale[i]:], axis=1)
            else:
                # Sum all frequency bins from the current scale index up to the
                # next scale index
                newspec[:, i] = np.sum(spec[:, scale[i]:scale[i+1]], axis=1)

        # List the center frequency of bins
        allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
        freqs = []
        # For each of the frequency bins
        for i in range(0, len(scale)):
            if i == len(scale)-1:
                freqs += [np.mean(allfreqs[scale[i]:])]
            else:
                freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

        return newspec, freqs

    def plotstft(self, samples, fs, binsize=2**10, plotpath=None,
                 colormap="jet"):
        """Plot spectrogram."""
        # Get all fft frames
        s = self.fft_analysis['frames'][:]

        print(fs)
        sshow, freq = self.logscale_spec(s, factor=1.0, sr=fs)

        # Amplitude to decibel
        ims = 20.*np.log10(np.abs(sshow)/10e-6)

        # Get the dimensions of the fft
        timebins, freqbins = np.shape(ims)

        plt.figure(figsize=(15, 7.5))
        plt.imshow(np.transpose(ims), origin="lower", aspect="auto",
                   cmap=colormap)
        # Add a colour bar to the side of the spectrogram.
        plt.colorbar()

        # Set spectrogram labels
        plt.xlabel("time (s)")
        plt.ylabel("frequency (hz)")
        plt.xlim([0, timebins-1])
        plt.ylim([0, freqbins])

        # Create an array of 5 values from 0 to the number of times
        xlocs = np.float32(np.linspace(0, timebins-1, 5))
        # Display time values at 5 points along the x axis of the graph
        plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/fs])
        # Display frequency values at 10 points along the y axis of the graph
        ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
        plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

        if plotpath:
            plt.savefig(plotpath, bbox_inches="tight")
        else:
            plt.show()

        plt.clf()

    def calc_fft_frame_times(self, fftframes, sample_frames, samplerate):
        """Calculate times for frames using sample size and samplerate."""

        # Get number of frames for time and frequency
        timebins, freqbins = np.shape(fftframes)
        # Create array ranging from 0 to number of time frames
        scale = np.arange(timebins+1)
        # divide the number of samples by the total number of frames, then
        # multiply by the frame numbers.
        fft_times = (sample_frames.shape[0]/timebins) * scale[:-1]
        # Divide by the samplerate to give times in seconds
        fft_times = fft_times / samplerate
        return fft_times


