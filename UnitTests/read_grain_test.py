import unittest
from pysound import AudioFile

class ReadGrainTestCase(unittest.TestCase)
    """Base class for all read grain tests"""

class ReadGrainTest(ReadGrainTestCase):
    def setUp(self):
        """Create functions and variables that will be defined before each test is run"""
        # TODO: Have wav file written and deleted from within this test
        self.TestAudio = AudioFile("./ManualWritten.TestAudio.Mono.wav", "r")

    def allGrainsTest(self):
        """Check all samples are read correctly if no arguments are given"""
        # TestAudio file has 100 samples
        self.assertEqual(self.TestAudio.read_grain(0, -1).size, 100, "Read Grain - all grains test failed.\nDidn't Read all samples")

    def sliceStartToMiddleTest(self):
        """
        Check that slice from begining of audio is read from and too the correct
        sample
        """
        grain = self.TestAudio.read_grain(0, 51)
        self.assertEqual(grain.size, 51, "Read Grain - Start to middle test failed.\nDidn't read correct number of samples.")
        self.assertEqual(grain[0], 0.0, "Read Grain - Start to middle test failed.\nDidn't read from correct start index.")
        self.assertEqual(grain[-1], 0.5, "Read Grain - Start to middle test failed.\nDidn't read to correct end index.")

    def negativeIndexingTest(self):
        """Check that slice from end is read from and too the correct sample"""
        grain = self.TestAudio.read_grain(-1, -51)
        if grain.size != 51 or grain[0] != 0.0 or grain[-1] != 0.0:
            print grain
            raise ValueError("Read grain test 2 failed:\nDidn't read correct slice of samples")

    def zeroPaddingTest(self)
        """
        Check that reading samples further than the end sample results in zero
        padding after the last sample
        """

    def tearDown(self):
        """
        Delete anything that is left over once tests are complete.
        ie. temporary test audio files generated during the tests.
