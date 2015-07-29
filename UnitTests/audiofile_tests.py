"""A set of unit tests to check the correct operation of the pysound module."""
import unittest
import numpy as np
from pysound import AudioFile
from fileops import pathops
import os


class FileCreationTests(unittest.TestCase):

    """Audio file creation tests."""

    def test_CreateAudioFile(self):
        """Check the creation of default audio files."""
        self.assertFalse(os.path.exists("./.TestAudio.wav"))
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav",
            overwrite_existing=True
        )
        self.assertTrue(os.path.exists("./.TestAudio.wav"))
        self.assertEquals(self.TestAudio.channels, 1)
        self.assertEquals(self.TestAudio.pysndfile_object.channels(), 1)
        self.assertEquals(self.TestAudio.mode, 'w')
        self.assertEquals(self.TestAudio.samplerate, 44100)
        self.assertEquals(self.TestAudio.pysndfile_object.samplerate(), 44100)
        self.assertEquals(self.TestAudio.format, 65539)
        self.assertEquals(self.TestAudio.pysndfile_object.format(), 65539)
        del self.TestAudio
        os.remove("./.TestAudio.wav")

    def test_ReadFail(self):
        """Check that opening a file that doesn't exist for reading fails."""
        with self.assertRaises(IOError):
            self.TestAudio = AudioFile.gen_default_wav(
                "./.TestAudio.wav",
                mode='r'
            )


class SwitchModeTests(unittest.TestCase):

    """Test read/write mode switching functionality."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav",
            mode='w',
            overwrite_existing=True
        )

    def test_SwitchMode(self):
        """
        Read/write mode switching test.

        Check that the switch_mode() function can write frames, then read them,
        then switch to write mode and append to these frames, before switching
        back to read all written frames.
        """
        # Check setup was correct
        self.assertEquals(self.TestAudio.channels, 1)
        self.assertEquals(self.TestAudio.pysndfile_object.channels(), 1)
        self.assertEquals(self.TestAudio.mode, 'w')
        self.assertEquals(self.TestAudio.samplerate, 44100)
        self.assertEquals(self.TestAudio.pysndfile_object.samplerate(), 44100)
        self.assertEquals(self.TestAudio.format, 65539)
        self.assertEquals(self.TestAudio.pysndfile_object.format(), 65539)

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


class ReadGrainTest(unittest.TestCase):

    """Test granular sample reading functionality."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav", overwrite_existing=True
        )
        self.TestAudio.write_frames(np.linspace(-0.5, 0.5, 101))
        self.TestAudio.switch_mode('r')

    def check_setup(self):
        """Check setup was correct."""
        self.assertEquals(self.TestAudio.channels, 1)
        self.assertEquals(self.TestAudio.pysndfile_object.channels(), 1)
        self.assertEquals(self.TestAudio.mode, 'r')
        self.assertEquals(self.TestAudio.samplerate, 44100)
        self.assertEquals(self.TestAudio.pysndfile_object.samplerate(), 44100)
        self.assertEquals(self.TestAudio.format, 65539)
        self.assertEquals(self.TestAudio.pysndfile_object.format(), 65539)

    def test_AllGrains(self):
        """
        Read grain - Read all samples test.

        Check all samples are read correctly if no arguments are given.
        """
        # TestAudio file has 100 samples
        self.check_setup()
        self.assertEqual(
            self.TestAudio.read_grain(0, -1).size, 101, "Read Grain - all "
            "grains test failed: Didn't Read all samples")

    def test_SliceStartToMiddle(self):
        """
        Read grain - Slicing test.

        Check that slice from begining of audio is read from and too the
        correct sample.
        """
        self.check_setup()
        grain = self.TestAudio.read_grain(0, 51)
        self.assertEqual(grain.size, 51, "Read Grain - Start to middle test "
                         "failed: Didn't read correct number of samples.")
        self.assertEqual(grain[0], -0.5, "Read Grain - Start to middle test "
                         "failed: Didn't read from correct start index.")
        self.assertEqual(grain[-1], 0.0, "Read Grain - Start to middle test "
                         "failed: Didn't read to correct end index.")

    def test_NegativeIndexing(self):
        """
        Read grain - Negative indexing test.

        Check that slice from end is read from and too the correct sample.
        """
        self.check_setup()
        grain = self.TestAudio.read_grain(-51, 51)
        self.assertEqual(grain.size, 51, "Read Grain - Negative indexing test "
                         "failed: Didn't return correct number of samples.")
        self.assertEqual(grain[0], 0.0, "Read Grain - Negative indexing test "
                         "failed: Didn't start at correct grain index.")
        self.assertEqual(grain[-1], 0.5, "Read Grain - Negative indexing test "
                         "failed: Didn't read to correct end index.")

    def test_ZeroPadding(self):
        """
        Read grain - zero padding test.

        Check that reading samples further than the end sample results in zero
        padding after the last sample.
        """
        self.check_setup()
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


class GenerateWhiteNoiseTest(unittest.TestCase):

    """Test white noise generation."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav",
            overwrite_existing=True,
            channels=1
        )

    def check_setup(self):
        """Check setup was correct."""
        self.assertEquals(self.TestAudio.channels, 1)
        self.assertEquals(self.TestAudio.pysndfile_object.channels(), 1)
        self.assertEquals(self.TestAudio.mode, 'w')
        self.assertEquals(self.TestAudio.samplerate, 44100)
        self.assertEquals(self.TestAudio.pysndfile_object.samplerate(), 44100)
        self.assertEquals(self.TestAudio.format, 65539)
        self.assertEquals(self.TestAudio.pysndfile_object.format(), 65539)

    def test_CreateNoise(self):
        """Test that white noise is generated corectly without clipping."""
        self.check_setup()
        samples = AudioFile.gen_white_noise(2 * self.TestAudio.samplerate, 0.7)
        # Check all samples are within the range of -1.0 to 1.0
        self.assertFalse((samples < -1.0).any() and (samples > 1.0).any())
        self.TestAudio.write_frames(samples)
        self.TestAudio.switch_mode('r')
        self.assertEqual(self.TestAudio.frames(), 88200)

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.

        For example: delete temporary test audio files generated during the
        tests.
        """
        del self.TestAudio
        pathops.delete_if_exists("./.TestAudio.wav")


class MonoDownmixTest(unittest.TestCase):

    """Test mixing of audio files from multi-channel to mono."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav",
            overwrite_existing=True,
            channels=2
        )

        samples = AudioFile.mono_arrays_to_stereo(
            np.linspace(-0.5, 0.5, 101),
            np.linspace(-0.5, 0.5, 101)
        )
        self.TestAudio.write_frames(
            samples
        )
        self.TestAudio.switch_mode('r')
        self.TestAudio.seek(0, 0)

    def check_setup(self):
        """Check setup was correct."""
        self.assertEquals(self.TestAudio.channels, 2)
        self.assertEquals(self.TestAudio.pysndfile_object.channels(), 2)
        self.assertEquals(self.TestAudio.mode, 'r')
        self.assertEquals(self.TestAudio.samplerate, 44100)
        self.assertEquals(self.TestAudio.pysndfile_object.samplerate(), 44100)
        self.assertEquals(self.TestAudio.format, 65539)
        self.assertEquals(self.TestAudio.pysndfile_object.format(), 65539)

    def test_MixToMono(self):
        """Run the test."""
        self.check_setup()

        mono_file = self.TestAudio.convert_to_mono(overwrite_original=False)
        mono_file.switch_mode('r')
        mono_file.seek(0, 0)
        stereo_frames = self.TestAudio.frames()
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
        self.assertEqual(stereo_frames, mono_file.frames())
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
        self.assertEqual(stereo_frames, mono_file.frames())
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


class NormalizeTest(unittest.TestCase):

    """Test the audio normalization functionality."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav",
            overwrite_existing=True,
        )

        samples = np.linspace(-0.3, 0.3, 1000)
        self.TestAudio.write_frames(
            samples
        )
        self.TestAudio.seek(0, 0)

    def check_setup(self):
        """Check setup was correct."""
        self.assertEquals(self.TestAudio.channels, 1)
        self.assertEquals(self.TestAudio.pysndfile_object.channels(), 1)
        self.assertEquals(self.TestAudio.mode, 'w')
        self.assertEquals(self.TestAudio.samplerate, 44100)
        self.assertEquals(self.TestAudio.pysndfile_object.samplerate(), 44100)
        self.assertEquals(self.TestAudio.format, 65539)
        self.assertEquals(self.TestAudio.pysndfile_object.format(), 65539)

    def test_Normalize(self):
        """Run the test."""
        self.check_setup()
        normalized_file = self.TestAudio.normalize_file(
            overwrite_original=False
        )
        normalized_frames = self.TestAudio.frames()
        self.assertGreater(normalized_frames, 0)
        # Check new file exists
        self.assertTrue(os.path.exists("./.TestAudio.norm.wav"))
        # Check new file is normalized
        self.assertEqual(normalized_file.channels, 1)
        # Check original exists
        self.assertTrue(os.path.exists("./.TestAudio.wav"))
        # Check new file is the same length in frames as the original
        self.assertEqual(normalized_frames, normalized_file.frames())
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
        self.assertEqual(normalized_frames, normalized_file.frames())
        # Check that all samples have been normalized
        self.TestAudio.switch_mode('r')
        self.TestAudio.seek(0, 0)
        self.assertGreater(self.TestAudio.read_frames().all(), 0.9)
        self.assertEqual(self.TestAudio.frames(), 1000)

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.

        For example: delete temporary test audio files generated during the
        tests.
        """
        del self.TestAudio
        pathops.delete_if_exists("./.TestAudio.wav")
        pathops.delete_if_exists("./.TestAudio.norm.wav")


class RenameFileTests(unittest.TestCase):

    """Test the audio file renaming functionality."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav", overwrite_existing=True
        )
        self.TestAudio.write_frames(np.linspace(-0.5, 0.5, 101))
        self.TestAudio.switch_mode('r')

    def check_setup(self):
        """Check setup was correct."""
        self.assertEquals(self.TestAudio.channels, 1)
        self.assertEquals(self.TestAudio.pysndfile_object.channels(), 1)
        self.assertEquals(self.TestAudio.mode, 'r')
        self.assertEquals(self.TestAudio.samplerate, 44100)
        self.assertEquals(self.TestAudio.pysndfile_object.samplerate(), 44100)
        self.assertEquals(self.TestAudio.format, 65539)
        self.assertEquals(self.TestAudio.pysndfile_object.format(), 65539)

    def test_RenameFile(self):
        """Run the tests."""
        self.check_setup()
        original_framecount = self.TestAudio.frames()
        original_channels = self.TestAudio.channels
        # Check original file exists
        self.assertTrue(os.path.exists("./.TestAudio.wav"))
        # Rename the file
        self.TestAudio.rename_file("./.TestAudio.rename.wav")
        # Check file has new name
        self.assertTrue(os.path.exists("./.TestAudio.rename.wav"))
        self.assertEqual(self.TestAudio.wavpath, "./.TestAudio.rename.wav")
        # Check new file has correct data
        self.assertEqual(self.TestAudio.frames(), original_framecount)
        self.assertEqual(self.TestAudio.channels, original_channels)
        # Check original file no longer exists
        self.assertFalse(os.path.exists("./.TestAudio.wav"))
        self.assertEqual(self.TestAudio.frames(), 101)

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.

        For example: delete temporary test audio files generated during the
        tests.
        """
        del self.TestAudio
        pathops.delete_if_exists("./.TestAudio.wav")
        pathops.delete_if_exists("./.TestAudio.rename.wav")


class ReplaceFileTests(unittest.TestCase):

    """Test audio file replacement functionality."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav", overwrite_existing=True
        )
        self.TestAudio.write_frames(np.zeros(101))
        self.TestAudio2 = AudioFile.gen_default_wav(
            "./.TestAudio.replace.wav", overwrite_existing=True
        )
        self.TestAudio2.write_frames(np.ones(50))
        del self.TestAudio2

    def check_setup(self):
        """Check setup was correct."""
        self.assertEquals(self.TestAudio.channels, 1)
        self.assertEquals(self.TestAudio.pysndfile_object.channels(), 1)
        self.assertEquals(self.TestAudio.mode, 'w')
        self.assertEquals(self.TestAudio.samplerate, 44100)
        self.assertEquals(self.TestAudio.pysndfile_object.samplerate(), 44100)
        self.assertEquals(self.TestAudio.format, 65539)
        self.assertEquals(self.TestAudio.pysndfile_object.format(), 65539)

    def test_ReplaceFile(self):
        """Do the test."""
        self.check_setup()
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
        self.assertTrue(self.TestAudio.wavpath, "./.TestAudio.wav")
        self.assertEqual(self.TestAudio.frames(), 50)


class FadeAudioTest(unittest.TestCase):

    """Test audio fade in/out functionality."""

    def setUp(self):
        """Create functions and variables before each test is run."""
        # Generate 2 second of ones at a samplerate of 44.1Khz
        self.test_audio = np.ones(88200)
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav", overwrite_existing=True
        )

    def check_setup(self):
        """Check setup was correct."""
        self.assertEquals(self.TestAudio.channels, 1)
        self.assertEquals(self.TestAudio.pysndfile_object.channels(), 1)
        self.assertEquals(self.TestAudio.mode, 'w')
        self.assertEquals(self.TestAudio.samplerate, 44100)
        self.assertEquals(self.TestAudio.pysndfile_object.samplerate(), 44100)
        self.assertEquals(self.TestAudio.format, 65539)
        self.assertEquals(self.TestAudio.pysndfile_object.format(), 65539)

    def test_FadeIn(self):
        """Check that audio is faded in and out correctly and accuratley."""
        self.check_setup()
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
            500,
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

ReadGrainSuite = unittest.TestLoader().loadTestsFromTestCase(ReadGrainTest)
SwitchModeSuite = unittest.TestLoader().loadTestsFromTestCase(SwitchModeTests)
FileCreationSuite = unittest.TestLoader().loadTestsFromTestCase(
    FileCreationTests
)

AllTestsSuite = unittest.TestSuite(
    [ReadGrainSuite, SwitchModeSuite, FileCreationSuite]
)
