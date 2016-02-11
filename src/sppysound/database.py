from __future__ import print_function, division
import os
import shutil
import collections
from scipy import signal
import numpy as np
import pysndfile
import matplotlib.pyplot as plt
import pdb
import sys
import traceback
import logging
import h5py
import multiprocessing as mp

from fileops import pathops
from audiofile import AnalysedAudioFile, AudioFile
import analysis.RMSAnalysis as RMSAnalysis
import analysis.AttackAnalysis as AttackAnalysis
import analysis.ZeroXAnalysis as ZeroXAnalysis
import analysis.FFTAnalysis as FFTAnalysis
import analysis.SpectralCentroidAnalysis as SpectralCentroidAnalysis
import analysis.SpectralSpreadAnalysis as SpectralSpreadAnalysis
import analysis.F0Analysis as F0Analysis

logger = logging.getLogger(__name__).addHandler(logging.NullHandler())

class AudioDatabase:

    """A class for encapsulating a database of AnalysedAudioFile objects."""

    def __init__(
        self,
        db_dir=None,
        audio_dir=None,
        analysis_list=[],
        *args,
        **kwargs
    ):
        """
        Create the folder hierachy for the database of files to be stored in.

        Adds any pre existing audio files and analyses to the object
        automatically.
        audio_dir:
        self.db_dir:
        analysis_list:
        """
        self.db_dir = db_dir
        self.audio_dir = audio_dir
        self.analysis_list = analysis_list
        self.config = kwargs.pop("config", None)
        self.logger = logging.getLogger(__name__ + '.AudioDatabase')

        # Check that all analysis list args are valid
        valid_analyses = {'rms', 'zerox', 'fft', 'spccntr', 'spcsprd', 'f0'}
        for analysis in analysis_list:
            if analysis not in valid_analyses:
                raise ValueError("\'{0}\' is not a valid analysis type".format(analysis))

        self.analysis_list = set(self.analysis_list)

        self.logger.info("Initialising Database...")

        # Create empty list to fill with audio file paths
        self.audio_file_list = []

    def load_database(self, reanalyse=False):
        """Create/Read from a pre-existing database"""

        subdir_paths = self.create_subdirs()

        if self.audio_dir:
            # Check that audio directory exists
            if not os.path.exists(self.audio_dir):
                raise IOError("The audio directory provided ({0}) doesn't "
                            "exist").format(self.audio_dir)
            self.organize_audio(subdir_paths)

        analysed_audio = self.analyse_database(subdir_paths, reanalyse)

        # with self.analysed_audio[48] as AAF:
            # AAF.FFT.plotstft(AAF.read_grain(), AAF.samplerate, binsize=AAF.ms_to_samps(100))

    def analyse_database(self, subdir_paths, reanalyse):
        # Create data file for storing analysis data for the database
        datapath = os.path.join(subdir_paths['data'], 'analysis_data.hdf5')
        self.data = h5py.File(datapath, 'a')
        self.analysed_audio = []

        for item in self.audio_file_list:
            filepath = os.path.join(subdir_paths['audio'], item)
            print("--------------------------------------------------")
            # if there is no wav file then skip
            try:
                with AnalysedAudioFile(
                    filepath,
                    'r',
                    analyses=self.analysis_list,
                    name=os.path.basename(item),
                    db_dir=self.db_dir,
                    data_file=self.data,
                    reanalyse=reanalyse,
                    config=self.config
                ) as AAF:
                    AAF.create_analysis()
                    self.analysed_audio.append(AAF)
            except IOError as err:
                # Skip any audio file objects that can't be analysed
                self.logger.warning("File cannot be analysed: {0}\nReason: {1}\n"
                      "Skipping...".format(item, err))
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          file=sys.stdout)
                continue
        print("--------------------------------------------------")
        self.logger.debug("Analysis Finished.")

    def create_subdirs(self):

        # If the database directory isnt specified then the directory where the
        # audio files are stored will be used
        if not self.db_dir:
            if not self.audio_dir:
                raise IOError("No database location specified. Either a "
                              "database ocation or audio file location must be"
                              " specified.")
            self.db_dir = self.audio_dir


        # Check to see if the database directory already exists
        # Create if not
        pathops.dir_must_exist(self.db_dir)

        def initialise_subdir(dirkey):
            """
            Create a subdirectory in the database with the name of the key
            provided.
            Returns the path to the created subdirectory.
            """
            # Make sure database subdirectory exists
            directory = os.path.join(self.db_dir, dirkey)
            try:
                # If it doesn't, Create it.
                os.mkdir(directory)
                self.logger.info(''.join(("Created directory: ", directory)))
            except OSError as err:
                # If it does exist, add it's content to the database content
                # dictionary.
                if os.path.exists(directory):
                    self.logger.warning("\'{0}\' directory already exists:"
                    " {1}".format(dirkey, os.path.relpath(directory)))
                    if dirkey == 'audio':
                        for item in pathops.listdir_nohidden(directory):
                            self.audio_file_list.append(item)
                else:
                    raise err
            return directory

        # Create a sub directory for every key in the analysis list
        # store reference to this in dictionary
        self.logger.info("Creating sub-directories...")
        directory_set = {'audio', 'data'}
        subdir_paths = {
            key: initialise_subdir(key) for key in directory_set
        }
        return subdir_paths

    def organize_audio(self, subdir_paths, symlink=True):
        self.logger.info("Moving any audio to sub directory...")

        valid_filetypes = {'.wav', '.aif', '.aiff'}
        # Move audio files to database
        # For all files in the audio dirctory...
        for root, directories, filenames in os.walk(self.audio_dir):
            for item in filenames:
                # If the file is a valid file type...
                item = os.path.join(root,item)
                if os.path.splitext(item)[1] in valid_filetypes:
                    self.logger.debug(''.join(("File added to database content: ", item)))
                    # Get the full path for the file
                    filepath = os.path.join(self.audio_dir, item)
                    # If the file isn't already in the database...
                    if not os.path.isfile(
                        '/'.join((subdir_paths["audio"], os.path.basename(filepath)))
                    ):
                        # Copy the file to the database
                        if symlink:
                            filename = os.path.basename(filepath)
                            os.symlink(filepath, os.path.join(subdir_paths["audio"], filename))
                            self.logger.info(''.join(("Linked: ", item, "\tTo directory: ",
                                subdir_paths["audio"], "\n")))
                        else:
                            shutil.copy2(filepath, subdir_paths["audio"])
                            self.logger.info(''.join(("Moved: ", item, "\tTo directory: ",
                                subdir_paths["audio"], "\n")))

                    else:
                        self.logger.info(''.join(("File:  ", item, "\tAlready exists at: ",
                            subdir_paths["audio"])))
                    # Add the file's path to the database content dictionary
                    self.audio_file_list.append(
                        os.path.join(subdir_paths["audio"], item)
                    )

    def close(self):
        self.data.close()
    def __enter__(self):
        return self
    def __exit__(self):
        self.close()

class Matcher:

    """
    Database comparison object.

    Used to compare and match entries in two AnalysedAudioFile databases.
    """

    def __init__(self, database1, database2, analysis_dict,*args, **kwargs):
        self.logger = logging.getLogger(__name__ + '.Matcher')
        self.source_db = database1
        self.target_db = database2
        self.analysis_dict = analysis_dict
        self.common_analyses = []
        """
        self.match_type = {
            "mean": self.mean_formatter,
            "median": self.median_formatter
        }
        """

        self.logger.debug("Initialised Matcher")
    def match(self, match_function, grain_size, overlap):
        """
        Find the closest match to each object in database 1 in database 2 using the matching function specified.
        """
        # Find all analyses shared by both the source and target entry
        common_analyses = self.source_db.analysis_list & self.target_db.analysis_list
        self.matcher_analyses = []
        # Create final list of analyses to perform matching on based on
        # selected match analyses.
        for key in self.analysis_dict.iterkeys():
            if key not in common_analyses:
                self.logger.warning("Analysis: \"{0}\" not avilable in {1} and/or {2}".format(key, source_entry, target_entry))
            else:
                self.matcher_analyses.append(key)

        # Run matching
        match_function(grain_size, overlap)

    def count_grains(self, database, grain_length, overlap):
        '''Calculate the number of grains in the database'''
        entry_count = len(database.analysed_audio)
        grain_indexes = np.empty((entry_count, 2))

        for ind, entry in enumerate(database.analysed_audio):
            length = entry.samps_to_ms(entry.frames)
            hop_size = grain_length / overlap
            grain_indexes[ind][0] = int(length / hop_size) - 1
        grain_indexes[:, 1] = np.cumsum(grain_indexes[:, 0])
        grain_indexes[:, 0] = grain_indexes[:, 1] - grain_indexes[:, 0]
        return grain_indexes

    def brute_force_matcher(self, grain_size, overlap):
        source_grain_indexes = self.count_grains(self.source_db, grain_size, overlap)

        for analysis in self.matcher_analyses:
            if not analysis == 'zerox':
                continue
            self.logger.debug("Current analysis: {0}".format(analysis))
            analysis_formatting = self.analysis_dict[analysis]

            for tind, target_entry in enumerate(self.target_db.analysed_audio):
                # Create an array of grain times for source sample
                target_times = target_entry.generate_grain_times(grain_size, overlap)

                # Get data for all target grains for each analysis
                target_data = target_entry.analysis_data_grains(target_times, analysis)

                # Get the analysis object for the current entry
                analysis_object = target_entry.analyses[analysis]
                # Format the target data ready for matching using the analysis
                # objects match formatting function.
                target_data = analysis_object.formatters[analysis_formatting](target_data)

                data_distance = np.empty((target_data.shape[0], source_grain_indexes[-1][-1]))
                for sind, source_entry in enumerate(self.source_db.analysed_audio):
                    # Create an array of grain times for source sample
                    source_times = source_entry.generate_grain_times(grain_size, overlap)

                    self.logger.debug("Matching \"{0}\" for: {1} to {2}".format(analysis, source_entry.name, target_entry.name))

                    # Get data for all source grains for each analysis
                    source_data = source_entry.analysis_data_grains(source_times, analysis)

                    # Format the source data ready for matching using the analysis
                    # objects match formatting function.
                    source_data = analysis_object.formatters[analysis_formatting](source_data)

                    # Get the start and end array indexes allocated for the
                    # current entry's grains.
                    start_index, end_index = source_grain_indexes[sind]
                    # Calculate the difference between the source and target
                    # values of each grain and add to array
                    data_distance[:, start_index:end_index] = np.abs(np.vstack(target_data) - source_data)


            pdb.set_trace()



    def swap_databases(self):
        """Convenience method to swap databases, changing the source database into the target and vice-versa"""
        self.source_db, self.target_db = self.target_db, self.source_db

