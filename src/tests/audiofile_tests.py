"""A set of unit tests to check the correct operation of the pysound module."""
import unittest
import numpy as np
import sys

######
# Import sppysound module from directory above current one
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from sppysound import AudioFile, analysis
from sppysound.database import AudioDatabase, Matcher
######

import subprocess
from scipy import signal

from fileops import pathops
import pdb
import os
import config
import math


class NumericAssertions:
    """
    This class is following the UnitTest naming conventions.
    It is meant to be used along with unittest.TestCase like so :
    class MyTest(unittest.TestCase, NumericAssertions):
        ...
    It needs python >= 2.6
    """

    def assertIsNaN(self, value, msg=None):
        """
        Fail if provided value is not NaN
        """
        standardMsg = "%s is not NaN" % str(value)
        try:
            if not math.isnan(value):
                self.fail(self._formatMessage(msg, standardMsg))
        except:
            self.fail(self._formatMessage(msg, standardMsg))

    def assertIsNotNaN(self, value, msg=None):
        """
        Fail if provided value is NaN
        """
        standardMsg = "Provided value is NaN"
        try:
            if math.isnan(value):
                self.fail(self._formatMessage(msg, standardMsg))
        except:
            pass

class globalTests(unittest.TestCase):

    """Includes functions that are accesible to all audiofile tests."""

    def check_setup(self, Audio, channels=1, mode='r',
                    format=65539, samplerate=44100):
        """Check setup was correct."""
        self.assertEquals(Audio.channels, channels)
        self.assertEquals(Audio.pysndfile_object.channels(), channels)
        self.assertEquals(Audio.mode, mode)
        self.assertEquals(Audio.samplerate, samplerate)
        self.assertEquals(Audio.pysndfile_object.samplerate(), samplerate)
        self.assertEquals(Audio.format, format)
        self.assertEquals(Audio.pysndfile_object.format(), format)

    def check_result(self, Audio, channels, mode='r',
                     format=65539, samplerate=44100):
        """Check the output file is the correct size, format etc..."""

    def create_test_audio(self, filename="./.TestAudio.wav",  mode='w', channels=1):
        """Create a default audio file to test on"""
        return AudioFile.gen_default_wav(
            filename,
            overwrite_existing=True,
            mode=mode,
            channels=channels,
        )


class FileCreationTests(globalTests):

    """Audio file creation tests."""

    def test_CreateAudioFile(self):
        """Check the creation of default audio files."""
        self.assertFalse(os.path.exists("./.TestAudio.wav"))
        self.TestAudio = self.create_test_audio()
        self.assertTrue(os.path.exists("./.TestAudio.wav"))
        self.assertEquals(self.TestAudio.channels, 1)
        self.assertEquals(self.TestAudio.pysndfile_object.channels(), 1)
        self.assertEquals(self.TestAudio.mode, 'w')
        self.assertEquals(self.TestAudio.samplerate, 44100)
        self.assertEquals(self.TestAudio.pysndfile_object.samplerate(), 44100)
        self.assertEquals(self.TestAudio.format, 65539)
        self.assertEquals(self.TestAudio.pysndfile_object.format(), 65539)

    def test_ReadFail(self):
        """Check that opening a file that doesn't exist for reading fails."""
        with self.assertRaises(IOError):
            self.TestAudio = AudioFile.gen_default_wav(
                "./.TestAudio.wav",
                mode='r'
            )

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.

        For example, remove all temporary test audio files generated during the
        tests.
        """
        pathops.delete_if_exists("./.TestAudio.wav")


class SwitchModeTests(globalTests):

    """Test read/write mode switching functionality."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.TestAudio = self.create_test_audio()

    def test_SwitchMode(self):
        """Check that pysndfile object mode switching works as expected."""
        # Check setup was correct
        self.check_setup(self.TestAudio, mode='w')

        self.TestAudio.write_frames(np.linspace(-0.5, 0.5, 101))
        self.TestAudio.switch_mode('r')
        with self.assertRaises(IOError):
            self.TestAudio.write_frames(np.zeros(10))
        self.TestAudio.seek(0, 0)
        self.assertEqual(self.TestAudio.read_frames().size, 101)
        self.TestAudio.switch_mode('w')
        self.TestAudio.seek(0, 0)
        self.assertEqual(self.TestAudio.read_frames().size, 0)
        self.assertEqual(
            self.TestAudio.write_frames(np.linspace(-0.5, 0.5, 101)), 101
        )

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.

        For example, remove all temporary test audio files generated during the
        tests.
        """
        del self.TestAudio
        pathops.delete_if_exists("./.TestAudio.wav")


class ReadGrainTest(globalTests):

    """Test granular sample reading functionality."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.TestAudio = self.create_test_audio()
        self.TestAudio.write_frames(np.linspace(-0.5, 0.5, 101))
        self.TestAudio.switch_mode('r')

    def test_AllGrains(self):
        """Check all samples are read correctly if no arguments are given."""
        # TestAudio file has 100 samples
        self.check_setup(self.TestAudio)
        self.assertEqual(
            self.TestAudio.read_grain(0, -1).size, 101, "Read Grain - all "
            "grains test failed: Didn't Read all samples")

    def test_SliceStartToMiddle(self):
        """Check that start slicing is read from and too the correct sample."""
        self.check_setup(self.TestAudio)
        grain = self.TestAudio.read_grain(0, 51)
        self.assertEqual(grain.size, 51, "Read Grain - Start to middle test "
                         "failed: Didn't read correct number of samples.")
        self.assertEqual(grain[0], -0.5, "Read Grain - Start to middle test "
                         "failed: Didn't read from correct start index.")
        self.assertEqual(grain[-1], 0.0, "Read Grain - Start to middle test "
                         "failed: Didn't read to correct end index.")

    def test_NegativeIndexing(self):
        """Check that end slicing is read from and too the correct sample."""
        self.check_setup(self.TestAudio)
        grain = self.TestAudio.read_grain(-51, 51)
        self.assertEqual(grain.size, 51, "Read Grain - Negative indexing test "
                         "failed: Didn't return correct number of samples.")
        self.assertEqual(grain[0], 0.0, "Read Grain - Negative indexing test "
                         "failed: Didn't start at correct grain index.")
        self.assertEqual(grain[-1], 0.5, "Read Grain - Negative indexing test "
                         "failed: Didn't read to correct end index.")

    def test_ZeroPadding(self):
        """
        Check that reading samples further than the end sample results in zero
        padding after the last sample.
        """
        self.check_setup(self.TestAudio)
        grain = self.TestAudio.read_grain(-26, 50)
        self.assertEqual(grain.size, 50, "Read Grain - Zero padding test "
                         "failed: Didn't read correct number of samples.")
        self.assertEqual(grain[0], 0.25, "Read Grain - Zero padding test "
                         "failed: Didn't read from correct start index")
        self.assertEqual(grain[25:50].all(), 0, "Read Grain - Zero padding "
                         "test failed: Didn't pad zeroes to end of grain "
                         "correctly")

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.

        For example: delete temporary test audio files generated during the
        tests.
        """
        del self.TestAudio
        pathops.delete_if_exists("./.TestAudio.wav")


class GenerateWhiteNoiseTest(globalTests):

    """Test white noise generation."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.TestAudio = self.create_test_audio()

    def test_CreateNoise(self):
        """Test that white noise is generated corectly without clipping."""
        self.check_setup(self.TestAudio, mode='w',)
        samples = AudioFile.gen_white_noise(2 * self.TestAudio.samplerate, 0.7)

        # Check all samples are within the range of -1.0 to 1.0
        self.assertFalse((samples < -1.0).any() and (samples > 1.0).any())
        self.TestAudio.write_frames(samples)
        self.TestAudio.switch_mode('r')
        self.assertEqual(self.TestAudio.get_frames(), 88200)

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.

        For example: delete temporary test audio files generated during the
        tests.
        """
        del self.TestAudio
        pathops.delete_if_exists("./.TestAudio.wav")


class MonoDownmixTest(globalTests):

    """Test mixing of audio files from multi-channel to mono."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.TestAudio = self.create_test_audio(channels=2)

        samples = AudioFile.mono_arrays_to_stereo(
            np.linspace(-0.5, 0.5, 101),
            np.linspace(-0.5, 0.5, 101)
        )
        self.TestAudio.write_frames(
            samples
        )
        self.TestAudio.switch_mode('r')
        self.TestAudio.seek(0, 0)

    def test_MixToMono(self):
        """Test that mixing audio to mono works as expected."""
        self.check_setup(self.TestAudio, channels=2)

        mono_file = self.TestAudio.convert_to_mono(overwrite_original=False)
        mono_file.switch_mode('r')
        mono_file.seek(0, 0)
        stereo_frames = self.TestAudio.frames
        self.assertGreater(stereo_frames, 0)
        # Check new file exists
        self.assertTrue(os.path.exists("./.TestAudio.mono.wav"))
        # Check new file is mono
        self.assertEqual(mono_file.channels, 1)
        # Check original exists
        self.assertTrue(os.path.exists("./.TestAudio.wav"))
        # Check original is stereo
        self.assertEqual(self.TestAudio.channels, 2)
        # Check new file is the same length in frames as the original
        self.assertEqual(stereo_frames, mono_file.frames)
        # Check that if mono file already exists, it is used to replace the
        # original if the function is run again with overwrite_original set to
        # True
        self.TestAudio.convert_to_mono(overwrite_original=True)
        # Check that mono file no longer exists
        self.assertFalse(os.path.exists("./.TestAudio.mono.wav"))
        # Check new file is mono
        self.assertEqual(self.TestAudio.channels, 1)
        # Check original exists
        self.assertTrue(os.path.exists("./.TestAudio.wav"))
        # Check new file is the same length in frames as the original
        self.assertEqual(stereo_frames, mono_file.frames)
        self.TestAudio.seek(0, 0)
        samples = self.TestAudio.read_frames()
        self.TestAudio.switch_mode('r')
        self.TestAudio.seek(0, 0)
        samples = self.TestAudio.read_frames()
        self.assertEqual(samples.size, 101)

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.

        For example: delete temporary test audio files generated during the
        tests.
        """
        del self.TestAudio
        pathops.delete_if_exists("./.TestAudio.wav")
        pathops.delete_if_exists("./.TestAudio.mono.wav")


class NormalizeTest(globalTests):

    """Test the audio normalization functionality."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.TestAudio = self.create_test_audio()
        samples = np.linspace(-0.3, 0.3, 1000)
        self.TestAudio.write_frames(
            samples
        )
        self.TestAudio.seek(0, 0)

    def test_Normalize(self):
        """Test that audio normalization works as expected."""
        self.check_setup(self.TestAudio, mode='w')
        normalized_file = self.TestAudio.normalize_file(
            overwrite_original=False
        )
        normalized_frames = self.TestAudio.frames
        self.assertGreater(normalized_frames, 0)
        # Check new file exists
        self.assertTrue(os.path.exists("./.TestAudio.norm.wav"))
        # Check new file is normalized
        self.assertEqual(normalized_file.channels, 1)
        # Check original exists
        self.assertTrue(os.path.exists("./.TestAudio.wav"))
        # Check new file is the same length in frames as the original
        self.assertEqual(normalized_frames, normalized_file.frames)
        # Check that if normalized file already exists, it is used to replace
        # the original if the function is run again with overwrite_original set
        # to True
        self.TestAudio.normalize_file(overwrite_original=True)
        # Check that normalized file no longer exists
        self.assertFalse(os.path.exists("./.TestAudio.norm.wav"))
        # Check new file is normalized
        self.assertEqual(normalized_file.channels, 1)
        # Check original exists
        self.assertTrue(os.path.exists("./.TestAudio.wav"))
        # Check new file is the same length in frames as the original
        self.assertEqual(normalized_frames, normalized_file.frames)
        # Check that all samples have been normalized
        self.TestAudio.switch_mode('r')
        self.TestAudio.seek(0, 0)
        self.assertGreater(self.TestAudio.read_frames().all(), 0.9)
        self.assertEqual(self.TestAudio.frames, 1000)

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.

        For example: delete temporary test audio files generated during the
        tests.
        """
        del self.TestAudio
        pathops.delete_if_exists("./.TestAudio.wav")
        pathops.delete_if_exists("./.TestAudio.norm.wav")


class RenameFileTests(globalTests):

    """Test the audio file renaming functionality."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.TestAudio = self.create_test_audio()
        self.TestAudio.write_frames(np.linspace(-0.5, 0.5, 101))
        self.TestAudio.switch_mode('r')

    def test_RenameFile(self):
        """Check that file renaming function works correctly."""
        self.check_setup(self.TestAudio, mode='r')
        original_framecount = self.TestAudio.frames
        original_channels = self.TestAudio.channels
        # Check original file exists
        self.assertTrue(os.path.exists("./.TestAudio.wav"))
        # Rename the file
        self.TestAudio.rename_file("./.TestAudio.rename.wav")
        # Check file has new name
        self.assertTrue(os.path.exists("./.TestAudio.rename.wav"))
        self.assertEqual(self.TestAudio.filepath, "./.TestAudio.rename.wav")
        # Check new file has correct data
        self.assertEqual(self.TestAudio.frames, original_framecount)
        self.assertEqual(self.TestAudio.channels, original_channels)
        # Check original file no longer exists
        self.assertFalse(os.path.exists("./.TestAudio.wav"))
        self.assertEqual(self.TestAudio.frames, 101)

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.

        For example: delete temporary test audio files generated during the
        tests.
        """
        del self.TestAudio
        pathops.delete_if_exists("./.TestAudio.wav")
        pathops.delete_if_exists("./.TestAudio.rename.wav")


class ReplaceFileTests(globalTests):

    """Test audio file replacement functionality."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.TestAudio = self.create_test_audio()
        self.TestAudio.write_frames(np.zeros(101))
        self.TestAudio2 = self.create_test_audio(
            filename='./.TestAudio.replace.wav'
        )
        self.TestAudio2.write_frames(np.ones(50))
        del self.TestAudio2

    def test_ReplaceFile(self):
        """Check that file replacment function works correctly."""
        self.check_setup(self.TestAudio, mode='w')
        # Check that file to be replaced exists
        self.assertTrue(os.path.exists("./.TestAudio.wav"))
        # Check that file to replace file with exists
        self.assertTrue(os.path.exists("./.TestAudio.replace.wav"))
        # Check attempt to replace with a file that doesn't exists fails.
        with self.assertRaises(IOError):
            self.TestAudio.replace_audiofile("./.FileThatDoesntExist.wav")
        # Replace file with replacement file
        self.TestAudio.replace_audiofile("./.TestAudio.replace.wav")
        # Check that the replacement file no longer exists
        self.assertFalse(os.path.exists("./.TestAudio.replace.wav"))
        # Check that currently open object is now refferencing the replacement
        # file and that it has been renamed correctly
        self.assertTrue(os.path.exists("./.TestAudio.wav"))
        self.assertTrue(self.TestAudio.filepath, "./.TestAudio.wav")
        self.assertEqual(self.TestAudio.frames, 50)


class FadeAudioTest(globalTests):

    """Test audio fade in/out functionality."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        # Generate 2 second of ones at a samplerate of 44.1Khz
        self.test_audio = np.ones(88200)
        self.TestAudio = self.create_test_audio()

    def test_FadeIn(self):
        """Check that audio is faded in and out correctly and accuratley."""
        self.check_setup(self.TestAudio, mode='w')
        faded_audio = self.TestAudio.fade_audio(
            self.test_audio,
            500,
            1000,
            "in"
        )
        self.assertEquals(faded_audio[0], 0)
        self.assertEquals(faded_audio[self.TestAudio.ms_to_samps(500)], 0)
        self.assertEquals(faded_audio[-1], 1)
        self.assertEquals(faded_audio[self.TestAudio.ms_to_samps(1500)], 1)

        # Reset test array
        self.test_audio = np.ones(88200)
        faded_audio = self.TestAudio.fade_audio(
            self.test_audio,
            0,
            1000,
            "out"
        )
        self.assertEquals(faded_audio[0], 1)
        self.assertEquals(faded_audio[self.TestAudio.ms_to_samps(500)], 1)
        self.assertEquals(faded_audio[-1], 0)
        self.assertEquals(faded_audio[self.TestAudio.ms_to_samps(1500)], 0)

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.

        For example: delete temporary test audio files generated during the
        tests.
        """
        del self.TestAudio
        pathops.delete_if_exists("./.TestAudio.wav")
        pathops.delete_if_exists("./.TestAudio.rename.wav")


class RMSAnalysisTests(globalTests):

    """Tests RMS analysis generation"""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.silence = np.zeros(512)
        self.sine = np.sin(2*np.pi*5000*np.arange(512)/512)
        self.noise = np.random.uniform(low=-1.0, high=1.0, size=512)

    def test_GenerateRMS(self):
        """Check that RMS values generated are the expected values"""
        output = analysis.RMSAnalysis.create_rms_analysis(self.silence, 44100, window=None)
        output1 = analysis.RMSAnalysis.create_rms_analysis(self.noise, 44100, window=None)
        output2 = analysis.RMSAnalysis.create_rms_analysis(self.sine, 44100, window=None)

        np.testing.assert_array_equal(output, 0.0)
        np.testing.assert_almost_equal(output1[1], 1./np.sqrt(3), decimal=1)
        np.testing.assert_almost_equal(output2[1], 1./np.sqrt(2), decimal=1)

class ZeroXAnalysisTests(globalTests):

    """Tests Zero-Crossing analysis generation"""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.silence = np.zeros(512)
        self.sine = np.sin(2*np.pi*5000*np.arange(44100)/44100)
        self.noise = np.random.uniform(low=-1.0, high=1.0, size=512)

    def test_GenerateRMS(self):
        """Check that RMS values generated are the expected values"""
        output = analysis.ZeroXAnalysis.create_zerox_analysis(self.silence)
        output1 = analysis.ZeroXAnalysis.create_zerox_analysis(self.noise)
        output2 = analysis.ZeroXAnalysis.create_zerox_analysis(self.sine, window_size=44100)
        #TODO: Finish this test...


class PeakAnalysisTests(globalTests):

    """Tests Peak analysis generation"""

    def setUp(self):
        self.silence = np.zeros(512)
        self.positive_max = np.ones(512)
        self.negative_max = np.ones(512)-2

    def test_GeneratePeak(self):
        """Check that RMS values generated are the expected values"""
        output = analysis.PeakAnalysis.create_peak_analysis(self.silence)
        output1 = analysis.PeakAnalysis.create_peak_analysis(self.positive_max)
        output2 = analysis.PeakAnalysis.create_peak_analysis(self.negative_max)

        np.testing.assert_array_equal(output, 0)
        np.testing.assert_array_equal(output1, 1)
        np.testing.assert_array_equal(output2, 1)

class SpectralCentroidAnalysisTests(globalTests, NumericAssertions):
    """Tests Spectral Centroid analysis generation."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.silence = np.zeros(512)
        self.equal_mag = np.ones(512)
        self.peak = self.silence.copy()
        self.peak[256] = 1

    def test_GenerateSpectralCentroid(self):
        output = analysis.SpectralCentroidAnalysis.create_spccntr_analysis([self.silence], 44100)
        output1 = analysis.SpectralCentroidAnalysis.create_spccntr_analysis([self.peak], 44100)
        output2 = analysis.SpectralCentroidAnalysis.create_spccntr_analysis([self.equal_mag], 44100)
        self.assertIsNaN(output)
        self.assertEqual(output1, 256)
        self.assertEqual(output2, 255.5)

class SpectralSpreadAnalysisTests(globalTests, NumericAssertions):
    """Tests Spectral Spread analysis generation."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.silence = np.zeros(512)
        self.equal_mag = np.ones(512)
        self.peak = self.silence.copy()
        self.peak[256] = 1

    def test_GenerateSpectralSpread(self):
        output = analysis.SpectralCentroidAnalysis.create_spccntr_analysis([self.silence], 44100)
        output = analysis.SpectralSpreadAnalysis.create_spcsprd_analysis([self.silence], output, 44100)
        output1 = analysis.SpectralCentroidAnalysis.create_spccntr_analysis([self.equal_mag], 44100)
        output1 = analysis.SpectralSpreadAnalysis.create_spcsprd_analysis([self.equal_mag], output1, 44100)
        output2 = analysis.SpectralCentroidAnalysis.create_spccntr_analysis([self.peak], 44100)
        output2 = analysis.SpectralSpreadAnalysis.create_spcsprd_analysis([self.peak], output2, 44100)
        self.assertIsNaN(output)
        np.testing.assert_almost_equal(output1, 147.801387)
        self.assertEquals(output2, 0)

class SpectralFluxAnalysisTests(globalTests):
    """Tests Spectral Flux analysis generation."""

    def setUp(self):
        self.silence = np.zeros(512)
        self.equal_mag = np.ones(512)
        self.peak = self.silence.copy()
        self.peak[256] = 1

    def test_GenerateSpectralFlux(self):
        x = np.vstack((self.equal_mag, self.peak))
        output = analysis.SpectralFluxAnalysis.create_spcflux_analysis(x)
        x = np.vstack((self.peak, self.peak))
        output1 = analysis.SpectralFluxAnalysis.create_spcflux_analysis(x)
        self.assertTrue(output[0] > output1[0])


class SpectralCrestFactorAnalysisTests(globalTests, NumericAssertions):
    """Tests Spectral Crest Factor analysis generation."""

    def setUp(self):
        self.silence = np.zeros(512)
        self.equal_mag = np.ones(512)
        self.peak = self.silence.copy()
        self.peak[256] = 1

    def test_GenerateSpectralCrestFactor(self):
        output = analysis.SpectralCrestFactorAnalysis.create_spccf_analysis([self.silence])
        output1 = analysis.SpectralCrestFactorAnalysis.create_spccf_analysis([self.equal_mag])
        output2 = analysis.SpectralCrestFactorAnalysis.create_spccf_analysis([self.peak])

        self.assertIsNaN(output)
        self.assertEquals(output1, 2./1024.)
        self.assertEquals(output2, 1.)

class SpectralFlatnessAnalysisTests(globalTests, NumericAssertions):
    """Tests Spectral Crest Factor analysis generation."""

    def setUp(self):
        self.silence = np.zeros(512)
        self.equal_mag = np.ones(512)
        self.peak = self.silence.copy()
        self.peak[256] = 1

    def test_GenerateSpectralFlatness(self):
        output = analysis.SpectralFlatnessAnalysis.create_spcflatness_analysis([self.silence])
        output1 = analysis.SpectralFlatnessAnalysis.create_spcflatness_analysis([self.equal_mag])
        output2 = analysis.SpectralFlatnessAnalysis.create_spcflatness_analysis([self.peak])

        self.assertIsNaN(output)
        self.assertEqual(output1, 1.)
        self.assertEqual(output2, 0.)

class KurtosisAnalysisTests(globalTests):
    """Tests Kurtosis analysis generation."""

    def setUp(self):
        self.sr = 44100
        self.f = 440
        x = np.arange(44100)
        self.sine_wave = np.sin(2*np.pi*self.f/self.sr*x)
        self.white_noise = np.random.random(44100)
        self.silence = np.zeros(44100)
        self.gausian = signal.gaussian(512, std=7)

    def test_GenerateKurtosis(self):
        sine_variance = analysis.VarianceAnalysis.create_variance_analysis(self.sine_wave)
        sine_output = analysis.KurtosisAnalysis.create_kurtosis_analysis(self.sine_wave, sine_variance)

        noise_variance = analysis.VarianceAnalysis.create_variance_analysis(self.white_noise)
        noise_output = analysis.KurtosisAnalysis.create_kurtosis_analysis(self.white_noise, noise_variance)

        silence_variance = analysis.VarianceAnalysis.create_variance_analysis(self.silence)
        silence_output = analysis.KurtosisAnalysis.create_kurtosis_analysis(self.silence, silence_variance)
        # TODO: Write assertions for results. This isn't testing anything
        # otherwise...

class F0AnalysisTests(globalTests):
    """Tests Spectral Spread analysis generation."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.TestAudio = self.create_test_audio()
        # Specify frequency of the sine wave
        self.sr = 44100
        self.f = 440
        x = np.arange(88200)+1
        self.sine_wave = np.sin(2*np.pi*self.f/self.sr*x)
        self.white_noise = 2 * np.random.random(88200) - 1

    def test_Generatef0(self):
        output = analysis.F0Analysis.create_f0_analysis(self.sine_wave, self.sr)
        average_output = np.median(output[:, 0])
        a = (average_output >= 437) & (average_output <= 443)
        self.assertTrue(a.all())

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.

        For example, remove all temporary test audio files generated during the
        tests.
        """
        del self.TestAudio
        pathops.delete_if_exists("./.TestAudio.wav")

class DatabaseTests(globalTests):
    """Tests database creation and analysis."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        pathops.dir_must_exist("./.test_db")
        self.sine_audio = self.create_test_audio(filename="./.test_db/test_sine.wav")
        self.silent_audio = self.create_test_audio(filename="./.test_db/test_silent.wav")
        self.noise_audio = self.create_test_audio(filename="./.test_db/test_noise.wav")
        f = 440
        x = np.arange(self.sine_audio.samplerate*2)+1
        sine_wave = np.sin(2*np.pi*f/self.sine_audio.samplerate*x)
        silence = np.zeros(self.silent_audio.samplerate*2)
        white_noise = 2 * np.random.random(self.noise_audio.samplerate*2) - 1

        self.sine_audio.write_frames(sine_wave)
        del self.sine_audio

        self.noise_audio.write_frames(white_noise)
        del self.noise_audio

        self.silent_audio.write_frames(silence)
        del self.silent_audio

    def test_DatabaseAnalysis(self):
        # Create database object
        database = AudioDatabase(
            "./.test_db",
            analysis_list=["rms", "zerox", "fft", "spccntr", "spcsprd", "f0"],
            config=config
        )
        # Create/load a pre-existing database
        database.load_database(reanalyse=True)

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.

        For example, remove all temporary test audio files generated during the
        tests.
        """
        pathops.delete_if_exists("./.test_db")

class MatcherTests(globalTests):
    """Tests database creation and analysis."""

    # Create database 1
    def setUp(self):
        """Create matcher object to be tested"""
        pathops.dir_must_exist("./.test_db1")
        self.sine_audio = self.create_test_audio(filename="./.test_db1/test_sine.wav")
        self.silent_audio = self.create_test_audio(filename="./.test_db1/test_silent.wav")
        self.noise_audio = self.create_test_audio(filename="./.test_db1/test_noise.wav")
        f = 440
        x = np.arange(self.sine_audio.samplerate*2)+1
        sine_wave = np.sin(2*np.pi*f/self.sine_audio.samplerate*x)
        silence = np.zeros(self.silent_audio.samplerate*2)
        white_noise = 2 * np.random.random(self.noise_audio.samplerate*2) - 1

        self.sine_audio.write_frames(sine_wave)
        del self.sine_audio

        self.noise_audio.write_frames(white_noise)
        del self.noise_audio

        self.silent_audio.write_frames(silence)
        del self.silent_audio
        # Create database object
        self.database1 = AudioDatabase(
            "./.test_db1",
            config=config
        )
        # Create/load a pre-existing database
        self.database1.load_database(reanalyse=True)

        # Create database 2
        pathops.dir_must_exist("./.test_db2")
        self.sine_audio = self.create_test_audio(filename="./.test_db2/test_sine.wav")
        self.silent_audio = self.create_test_audio(filename="./.test_db2/test_silent.wav")
        self.noise_audio = self.create_test_audio(filename="./.test_db2/test_noise.wav")
        f = 440
        x = np.arange(self.sine_audio.samplerate*2)+1
        sine_wave = np.sin(2*np.pi*f/self.sine_audio.samplerate*x)
        silence = np.zeros(self.silent_audio.samplerate*2)
        white_noise = 2 * np.random.random(self.noise_audio.samplerate*2) - 1

        self.sine_audio.write_frames(sine_wave)
        del self.sine_audio

        self.noise_audio.write_frames(white_noise)
        del self.noise_audio

        self.silent_audio.write_frames(silence)
        del self.silent_audio
        # Create database object
        self.database2 = AudioDatabase(
            "./.test_db2",
            config=config
        )
        # Create/load a pre-existing database
        self.database2.load_database(reanalyse=True)
        self.matcher = Matcher(self.database1, self.database2, {}, config=config)

    def test_DistanceCalc(self):
        data1 = np.array([np.nan, 1,2,3,4, np.nan, np.nan, 7, 6, 5])
        data2 = np.array([1, np.nan,2,3,4, 6, np.nan, 7, np.nan, 5])
        output = self.matcher.distance_calc(data1, data2)
        expected_output = np.array([[ 39.6,   0. ,  39.6,  39.6,  39.6,  39.6,   0. ,  39.6,   0. , 39.6],
       [  0. ,  39.6,   1. ,   4. ,   9. ,  25. ,  39.6,  36. ,  39.6,  16. ],
       [  1. ,  39.6,   0. ,   1. ,   4. ,  16. ,  39.6,  25. ,  39.6,   9. ],
       [  4. ,  39.6,   1. ,   0. ,   1. ,   9. ,  39.6,  16. ,  39.6,   4. ],
       [  9. ,  39.6,   4. ,   1. ,   0. ,   4. ,  39.6,   9. ,  39.6,   1. ],
       [ 39.6,   0. ,  39.6,  39.6,  39.6,  39.6,   0. ,  39.6,   0. , 39.6],
       [ 39.6,   0. ,  39.6,  39.6,  39.6,  39.6,   0. ,  39.6,   0. , 39.6],
       [ 36. ,  39.6,  25. ,  16. ,   9. ,   1. ,  39.6,   0. ,  39.6,   4. ],
       [ 25. ,  39.6,  16. ,   9. ,   4. ,   0. ,  39.6,   1. ,  39.6,   1. ],
       [ 16. ,  39.6,   9. ,   4. ,   1. ,   1. ,  39.6,   4. ,  39.6,   0. ]])
        np.testing.assert_array_equal(output, expected_output)

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.

        For example, remove all temporary test audio files generated during the
        tests.
        """
        pathops.delete_if_exists("./.test_db1")
        pathops.delete_if_exists("./.test_db2")


ReadGrainSuite = unittest.TestLoader().loadTestsFromTestCase(ReadGrainTest)
SwitchModeSuite = unittest.TestLoader().loadTestsFromTestCase(SwitchModeTests)
FileCreationSuite = unittest.TestLoader().loadTestsFromTestCase(
    FileCreationTests
)

AllTestsSuite = unittest.TestSuite(
    [ReadGrainSuite, SwitchModeSuite, FileCreationSuite]
)
if __name__ == "__main__":
    unittest.main()
