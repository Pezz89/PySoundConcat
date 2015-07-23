import unittest
import numpy as np
from pysound import AudioFile
from fileops import pathops
import os


class FileCreationTests(unittest.TestCase):
    def test_CreateAudioFile(self):
        """Check the creation of default audio files"""
        self.assertFalse(os.path.exists("./.TestAudio.wav"))
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav",
            overwrite_existing=True
        )
        self.assertTrue(os.path.exists("./.TestAudio.wav"))
        del self.TestAudio
        os.remove("./.TestAudio.wav")

    def test_ReadFail(self):
        """Check that opening a file that doesn't exist for reading fails"""
        with self.assertRaises(IOError):
            self.TestAudio = AudioFile.gen_default_wav(
                "./.TestAudio.wav",
                mode='r'
            )


class SwitchModeTests(unittest.TestCase):
    def setUp(self):
        """
        Create functions and variables that will be defined before each test
        is run.
        """
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav",
            mode='w',
            overwrite_existing=True
        )

    def test_SwitchMode(self):
        """
        Check that the switch_mode() function can write frames, then read them,
        then switch to write mode and append to these frames, before switching
        back to read all written frames.
        """
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
    def setUp(self):
        """
        Create functions and variables that will be defined before each test
        is run
        """
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav", overwrite_existing=True
        )
        self.TestAudio.write_frames(np.linspace(-0.5, 0.5, 101))
        self.TestAudio.switch_mode('r')

    def test_AllGrains(self):
        """Check all samples are read correctly if no arguments are given"""
        # TestAudio file has 100 samples
        self.assertEqual(
            self.TestAudio.read_grain(0, -1).size, 101, "Read Grain - all "
            "grains test failed: Didn't Read all samples")

    def test_SliceStartToMiddle(self):
        """
        Check that slice from begining of audio is read from and too the
        correct sample
        """
        grain = self.TestAudio.read_grain(0, 51)
        self.assertEqual(grain.size, 51, "Read Grain - Start to middle test "
                         "failed: Didn't read correct number of samples.")
        self.assertEqual(grain[0], -0.5, "Read Grain - Start to middle test "
                         "failed: Didn't read from correct start index.")
        self.assertEqual(grain[-1], 0.0, "Read Grain - Start to middle test "
                         "failed: Didn't read to correct end index.")

    def test_NegativeIndexing(self):
        """Check that slice from end is read from and too the correct sample"""
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
        padding after the last sample
        """
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
        ie. temporary test audio files generated during the tests.
        """
        del self.TestAudio
        pathops.delete_if_exists("./.TestAudio.wav")


class MonoDownmixTest(unittest.TestCase):
    def setUp(self):
        """
        Create functions and variables that will be defined before each test
        is run
        """
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

    def test_MixToMono(self):
        mono_file = self.TestAudio.convert_to_mono(overwrite_original=False)
        stereo_frames = self.TestAudio.frames()
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
        self.TestAudio.switch_mode('r')
        samples = self.TestAudio.read_frames()

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.
        ie. temporary test audio files generated during the tests.
        """
        del self.TestAudio
        pathops.delete_if_exists("./.TestAudio.wav")
        pathops.delete_if_exists("./.TestAudio.mono.wav")


class NormalizeTest(unittest.TestCase):
    def setUp(self):
        """
        Create functions and variables that will be defined before each test
        is run
        """
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav",
            overwrite_existing=True,
        )

        samples = np.linspace(-0.3, 0.3, 1000)
        self.TestAudio.write_frames(
            samples
        )
        self.TestAudio.seek(0, 0)

    def test_Normalize(self):
        normalized_file = self.TestAudio.normalize_file(
            overwrite_original=False
        )
        normalized_frames = self.TestAudio.frames()
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

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.
        ie. temporary test audio files generated during the tests.
        """
        del self.TestAudio
        pathops.delete_if_exists("./.TestAudio.wav")
        pathops.delete_if_exists("./.TestAudio.norm.wav")


class RenameFileTests(unittest.TestCase):
    def setUp(self):
        """
        Create functions and variables that will be defined before each test
        is run
        """
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav", overwrite_existing=True
        )
        self.TestAudio.write_frames(np.linspace(-0.5, 0.5, 101))

    def test_RenameFile(self):
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

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.
        ie. temporary test audio files generated during the tests.
        """
        del self.TestAudio
        pathops.delete_if_exists("./.TestAudio.wav")
        pathops.delete_if_exists("./.TestAudio.rename.wav")

class ReplaceFileTests(unittest.TestCase):
    def setUp(self):
        """
        Create functions and variables that will be defined before each test
        is run
        """
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav", overwrite_existing=True
        )
        self.TestAudio.write_frames(np.zeros(101))
        self.TestAudio2 = AudioFile.gen_default_wav(
            "./.TestAudio.replace.wav", overwrite_existing=True
        )
        self.TestAudio2.write_frames(np.ones(50))
        del self.TestAudio2

    def test_ReplaceFile(self):
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
    def setUp(self):
        """
        Create functions and variables that will be defined before each test
        is run
        """
        # Generate 2 second of ones at a samplerate of 44.1Khz
        self.test_audio = np.ones(88200)
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav", overwrite_existing=True
        )

    def test_FadeIn(self):
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
        ie. temporary test audio files generated during the tests.
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
