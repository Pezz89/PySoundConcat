import unittest
from pysound import AudioFile

class ReadGrainTest(unittest.TestCase):
    def setUp(self):
        # TODO: Have wav file written and deleted from within this test
        self.TestAudio = AudioFile("./ManualWritten.TestAudio.Mono.wav", "r")

    def allGrainsTest(self):
        """Check all samples are read correctly if no arguments are given"""
        # TestAudio file has 100 samples
        if TestAudio.read_grain(0, -1).size != 100:
            raise ValueError("Read grain test 1 failed:\nDidn't read all samples in file.")

    def sliceStartToMiddleTest(self):
        """
        Check that slice from begining of audio is read from and too the correct
        sample
        """
        grain = TestAudio.read_grain(0, 51)
        if grain.size != 51 or grain[0] != 0.0 or grain[-1] != 0.0:
            print grain
            raise ValueError("Read grain test 2 failed:\nDidn't read correct slice of samples")

    def negativeIndexingTest(self):
        """Check that slice from end is read from and too the correct sample"""
        grain = TestAudio.read_grain(-1, -51)
        if grain.size != 51 or grain[0] != 0.0 or grain[-1] != 0.0:
            print grain
            raise ValueError("Read grain test 2 failed:\nDidn't read correct slice of samples")

    def zeroPaddingTest(self)
        """
        Check that reading samples further than the end sample results in zero
        padding after the last sample
        """

