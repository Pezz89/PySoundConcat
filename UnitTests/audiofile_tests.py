import unittest
import numpy as np
from pysound import AudioFile
import os


class FileCreationTests(unittest.TestCase):
    def test_createAudioFile(self):
        """Check the creation of default audio files"""
        self.assertFalse(os.path.exists("./.TestAudio.wav"))
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav",
            overwrite_existing=True
        )
        self.assertTrue(os.path.exists("./.TestAudio.wav"))
        del self.TestAudio
        os.remove("./.TestAudio.wav")

    def test_readFail(self):
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
        is run
        """
        self.TestAudio = AudioFile.gen_default_wav(
            "./.TestAudio.wav",
            mode='w',
            overwrite_existing=True
        )

    def test_switchMode(self):
        """
        Check that the switch_mode() function can write frames, then read then,
        then switch to write mode and append to these frames, before switching
        back to read all written frames
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
        ie. temporary test audio files generated during the tests.
        """
        del self.TestAudio
        os.remove("./.TestAudio.wav")


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

    def test_allGrains(self):
        """Check all samples are read correctly if no arguments are given"""
        # TestAudio file has 100 samples
        self.assertEqual(
            self.TestAudio.read_grain(0, -1).size, 101, "Read Grain - all "
            "grains test failed: Didn't Read all samples")

    def test_sliceStartToMiddle(self):
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

    def test_negativeIndexing(self):
        """Check that slice from end is read from and too the correct sample"""
        grain = self.TestAudio.read_grain(-51, 51)
        self.assertEqual(grain.size, 51, "Read Grain - Negative indexing test "
                         "failed: Didn't return correct number of samples.")
        self.assertEqual(grain[0], 0.0, "Read Grain - Negative indexing test "
                         "failed: Didn't start at correct grain index.")
        self.assertEqual(grain[-1], 0.5, "Read Grain - Negative indexing test "
                         "failed: Didn't read to correct end index.")

    def test_zeroPadding(self):
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
        os.remove("./.TestAudio.wav")


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
        self.TestAudio.write_frames(
            AudioFile.mono_arrays_to_stereo(
                np.linspace(-0.5, 0.5, 101),
                np.linspace(-0.5, 0.5, 101)
            )
        )

    def test_mixToMono(self):
        mono_file = self.TestAudio.convert_to_mono(overwrite_original=False)
        # Check new file exists
        self.assertTrue(os.path.exists("./.TestAudio.mono.wav"))
        # Check new file is mono
        self.assertEqual(mono_file.channels(), 1)
        # Check original exists
        self.assertTrue(os.path.exists("./.TestAudio.wav"))
        # Check original is stereo
        self.assertEqual(self.TestAudio.channels(), 2)
        # Check new file is the same length in frames as the original
        self.assertEqual(self.TestAudio.frames(), mono_file.frames())
        # Check that if mono file already exists, it is used to replace the
        # original if the function is run again with overwrite_original set to
        # True
        self.TestAudio.convert_to_mono(overwrite_original=True)


ReadGrainSuite = unittest.TestLoader().loadTestsFromTestCase(ReadGrainTest)
SwitchModeSuite = unittest.TestLoader().loadTestsFromTestCase(SwitchModeTests)
FileCreationSuite = unittest.TestLoader().loadTestsFromTestCase(
    FileCreationTests
)

AllTestsSuite = unittest.TestSuite(
    [ReadGrainSuite, SwitchModeSuite, FileCreationSuite]
)
