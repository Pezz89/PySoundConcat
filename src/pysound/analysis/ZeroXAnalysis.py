from __future__ import print_function
import os
import numpy as np

import fileops.pathops as pathops


class ZeroXAnalysis:

    """ """

    def __init__(self, AnalysedAudioFile, zeroxpath):
        self.AnalysedAudioFile = AnalysedAudioFile
        self.zeroxpath = zeroxpath
        self.zerox_window_count = None

        # Check that the class file has a path to write the rms file to
        if not self.zeroxpath:
            # If it doesn't then attampt to generate a path based on the
            # location of the database that the object is a part of.
            if not self.AnalysedAudioFile.db_dir:
                # If it isn't part of a database and doesn't have a path then
                # there is no where to write the rms data to.
                raise IOError('Analysed Audio object must have an zerox file '
                              'path or be part of a database')
            self.zeroxpath = os.path.join(
                self.AnalysedAudioFile.db_dir,
                'zerox',
                self.AnalysedAudioFile.name + '.lab'
            )

        if self.AnalysedAudioFile.force_analysis:
            pathops.delete_if_exists(self.zeroxpath)
            self.zeroxpath = self.create_zerox_analysis()
        else:
            # Check if analysis file already exists.
            try:
                with open(self.zeroxpath, 'r') as zeroxfile:
                    # If an zerox file is provided then count the number of
                    # lines (1 for each window)
                    print("Reading Zero Crossing file:\t\t",
                          os.path.relpath(self.zeroxpath))
                    self.zerox_window_count = sum(1 for line in zeroxfile)
            except IOError:
                # If it doesn't then generate a new file
                self.zeroxpath = self.create_zerox_analysis()

    def create_zerox_analysis(self, window_size=25):
        """Generate zero crossing detections for windows of the signal"""
        self.zeroxpath = os.path.join(
            self.AnalysedAudioFile.db_dir,
            "zerox",
            self.AnalysedAudioFile.name + ".lab"
        )
        with open(self.zeroxpath, 'w') as zeroxfile:
            print("Creating zero-crossing file:\t\t",
                  os.path.relpath(self.zeroxpath))
            i = 0
            while i < self.AnalysedAudioFile.frames():
                # TODO: Find a more elegant way of writing this
                zero_crossings = np.where(
                    np.diff(np.sign(
                        self.AnalysedAudioFile.read_grain(i, window_size))
                    )
                )[0].size
                zeroxfile.write(
                    "{0} {1} {2}\n".format(
                        self.AnalysedAudioFile.samps_to_secs(i),
                        self.AnalysedAudioFile.samps_to_secs(i+window_size),
                        zero_crossings
                    )
                )
                i += window_size

    def get_zerox_from_file(self):
        """
        Retrieve zero crossing analysis from pre-created zero crossing file.
        """
        # TODO:
