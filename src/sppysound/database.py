from __future__ import print_function, division
import os
import shutil
import collections
from scipy import signal, spatial
import numpy as np
import pysndfile
import pdb
import sys
import traceback
import logging
import h5py
import pitch_shift
from sklearn.preprocessing import Imputer

from fileops import pathops
from audiofile import AnalysedAudioFile, AudioFile
from helper import OrderedSet
import analysis.RMSAnalysis as RMSAnalysis
import analysis.AttackAnalysis as AttackAnalysis
import analysis.ZeroXAnalysis as ZeroXAnalysis
import analysis.FFTAnalysis as FFTAnalysis
import analysis.SpectralCentroidAnalysis as SpectralCentroidAnalysis
import analysis.SpectralSpreadAnalysis as SpectralSpreadAnalysis
import analysis.F0Analysis as F0Analysis
import analysis.CentroidAnalysis as CentroidAnalysis

logger = logging.getLogger(__name__).addHandler(logging.NullHandler())

class AudioDatabase:

    """
    A class for encapsulating a database of AnalysedAudioFile objects.

    Arguments:

    - audio_dir: directory containing audio files to be used as the database

    - db_dir: directory to be used/created for storage of analysis data and
      storage/linking of audio files.

    - analysis_list: the list of analysis strings for analyses to be used in
      the database.
    """

    def __init__(
        self,
        audio_dir=None,
        db_dir=None,
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
        valid_analyses = {
            'rms',
            'zerox',
            'fft',
            'spccntr',
            'spcsprd',
            'spcflux',
            'spccf',
            'spcflatness',
            'f0',
            'peak',
            'centroid',
            'variance',
            'kurtosis',
            'skewness',
        }
        for analysis in analysis_list:
            if analysis not in valid_analyses:
                raise ValueError("\'{0}\' is not a valid analysis type".format(analysis))

        # Filter out repetitions in list if they exist
        self.analysis_list = set(self.analysis_list)

        self.logger.info("Initialising Database...")

        # Create empty list to fill with audio file paths
        self.audio_file_list = OrderedSet()

        self.data = None

    def __getitem__(self, key):
        """
        Allow for entry retreival via indexing.

        Returns and AnalysedAudioFile object at the index provided.
        """
        return self.analysed_audio[key]

    def load_database(self, reanalyse=False):
        """Create/Read from a pre-existing database"""

        subdir_paths = self.create_subdirs()

        if self.audio_dir:
            # Check that audio directory exists
            if not os.path.exists(self.audio_dir):
                raise IOError("The audio directory provided ({0}) doesn't "
                            "exist".format(self.audio_dir))
            self.organize_audio(subdir_paths, symlink=self.config.database["symlink"])

        self.analyse_database(subdir_paths, reanalyse)

    def analyse_database(self, subdir_paths, reanalyse):
        """
        create selected analyses for audio files in the database.

        Parameters:

        - subdir_paths: a dictionary containing paths to the 'audio' directory and 'data' directory of the database.

        - reanalyse: If previous analyses are found this can be set to True to overwrite them.
        """
        # Create data file for storing analysis data for the database
        datapath = os.path.join(subdir_paths['data'], 'analysis_data.hdf5')
        try:
            self.data = h5py.File(datapath, 'a')
        except IOError:
            raise IOError("Unable to create/append to file: {0}\nThis may be "
                          "due to another instance of this program running or a "
                          "corrupted HDF5 file.\n Make sure this is the only "
                          "running instance and regenerate HDF5 if "
                          "neccesary.".format(datapath))
        self.analysed_audio = []

        for item in self.audio_file_list:
            filepath = os.path.join(subdir_paths['audio'], os.path.basename(item))
            # if there is no wav file then skip
            try:
                with AnalysedAudioFile(
                    filepath,
                    'r',
                    data_file=self.data,
                    analyses=self.analysis_list,
                    name=os.path.basename(item),
                    db_dir=self.db_dir,
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
        self.logger.debug("Analysis Finished.")

    def add_file(self, file_object):
        '''Add an AnalysedAudioFile object to the database'''
        if type(file_object) is AnalysedAudioFile:
            self.analysed_audio.add(file_object)
            self.audio_file_list.append(file_object.filepath)
        else:
            raise TypeError("Object {0} of type {1} cannot be added to the database".format(file_object, file_object.__class__.__name__))

    def create_subdirs(self):
        """
        Generate database folder structure at the self.db_dir location

        If the folder structure already exists this will be used, else a new structure will be generated.
        """
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
        # Save sub-directory paths for later access
        self.subdirs = subdir_paths
        return subdir_paths

    def organize_audio(self, subdir_paths, symlink=True):
        """
        Moves/symlinks any audio from the audio directory specified to the database.

        Parameters:

        - subdir_paths: A dictionary of 'audio' and 'data' sub directories of the database.

        - symlink: If true, symbolic links will be used for audio files
          rather than copying. This allows audio files to stay in there
          original locations.
        """
        self.logger.info("Moving any audio to sub directory...")

        valid_filetypes = {'.wav', '.aif', '.aiff'}
        # Move audio files to database
        # For all files in the audio dirctory...
        for root, directories, filenames in os.walk(self.audio_dir):
            for item in filenames:
                # If the file is a valid file type...
                item = os.path.abspath(os.path.join(root,item))
                if os.path.splitext(item)[1] in valid_filetypes:
                    self.logger.debug(''.join(("File added to database content: ", item)))
                    # Get the full path for the file
                    filepath = os.path.abspath(os.path.join(self.audio_dir, os.path.basename(item)))
                    # If the file isn't already in the database...
                    filename = os.path.basename(filepath)
                    destination = os.path.abspath(os.path.join(subdir_paths["audio"], os.path.basename(filepath)))

                    if not os.path.isfile(destination) and not os.path.lexists(destination):
                        # Copy the file to the database
                        if symlink:
                            try:
                                os.symlink(item, os.path.join(os.path.abspath(subdir_paths["audio"]), filename))
                            except:
                                pass
                            self.logger.info(''.join(("Linked: ", item, "\tTo directory: ",
                                subdir_paths["audio"], "\n")))
                        else:
                            try:
                                os.unlink(destination)
                            except OSError:
                                pass
                            shutil.copy2(filepath, subdir_paths["audio"])
                            self.logger.info(''.join(("Copied: ", item, "\tTo directory: ",
                                subdir_paths["audio"], "\n")))

                    else:
                        if not symlink:
                            try:
                                linkpath = os.readlink(destination)
                                os.unlink(destination)
                            except OSError:
                                continue
                            shutil.copy2(linkpath, subdir_paths["audio"])
                            self.logger.info(''.join(("Copied: ", item, "\tTo directory: ",
                                subdir_paths["audio"], "\n")))
                        else:
                            self.logger.info(''.join(("File:  ", item, "\tAlready exists at: ",
                                subdir_paths["audio"])))
                    # Add the file's path to the database content dictionary
                    self.audio_file_list.add(
                        os.path.join(subdir_paths["audio"], os.path.basename(item))
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

    def __init__(self, database1, database2, *args, **kwargs):
        # Get matcher configurations object
        self.config = kwargs.pop('config', None)
        self.logger = logging.getLogger(__name__ + '.Matcher')

        # Set the number of best matches for each grain to store for use in synthesis.
        self.match_quantity = self.config.matcher["match_quantity"]
        self.source_db = database1
        self.target_db = database2
        self.output_db = kwargs.pop("output_db", None)
        self.rematch = kwargs.pop("rematch", self.config.matcher["rematch"])

        # Store a dictionary of analyses to perform matching on.
        self.analysis_dict = self.config.analysis_dict

        self.logger.debug("Initialised Matcher")

    def match(
        self,
        match_function,
        grain_size=None,
        overlap=None
    ):
        """
        Find the closest match to each object in database 1 in database 2 using the matching function specified.
        """
        if not grain_size:
            grain_size = self.config.matcher["grain_size"]
        if not overlap:
            overlap = self.config.matcher["overlap"]

        # Find all analyses shared by both the source and target entry
        common_analyses = self.source_db.analysis_list & self.target_db.analysis_list
        self.matcher_analyses = []
        # Create final list of analyses to perform matching on based on
        # selected match analyses.
        for key in self.analysis_dict.iterkeys():
            if key not in common_analyses:
                self.logger.warning("Analysis: \"{0}\" not avilable in {1} and/or {2}".format(key, self.source_db, self.target_db))
            else:
                self.matcher_analyses.append(key)

        # Run matching
        match_function(grain_size, overlap)

    def count_grains(self, database, grain_length, overlap):
        '''Calculate the number of grains in the database'''
        entry_count = len(database.analysed_audio)
        grain_indexes = np.empty((entry_count, 2))

        for ind, entry in enumerate(database.analysed_audio):
            length = entry.frames
            hop_size = grain_length / overlap
            grain_indexes[ind][0] = int(length / hop_size) - 1
        grain_indexes[:, 1] = np.cumsum(grain_indexes[:, 0]).astype(int)
        grain_indexes[:, 0] = grain_indexes[:, 1] - grain_indexes[:, 0]
        return grain_indexes

    def kdtree_matcher(self, grain_size, overlap):
        invalid_inds = []
        for i, entry in enumerate(self.target_db.analysed_audio):
            entry.generate_grain_times(grain_size, overlap, save_times=True)
            if not entry.times.size:
                invalid_inds.append(i)
        for i in sorted(invalid_inds, reverse=True):
            del self.target_db.analysed_audio[i]
        invalid_inds = []
        for i, entry in enumerate(self.source_db.analysed_audio):
            entry.generate_grain_times(grain_size, overlap, save_times=True)
            if not entry.times.size:
                invalid_inds.append(i)
        for i in sorted(invalid_inds, reverse=True):
            del self.source_db.analysed_audio[i]
        # Count grains of the source database
        source_sample_indexes = self.count_grains(self.source_db, grain_size, overlap)
        try:
            self.output_db.data.create_group("match")
        except ValueError:
            self.logger.info("Match group already exists in the {0} HDF5 file.".format(self.output_db))

        if self.rematch:
            self.output_db.data["match"].clear()

        if self.config:
            weightings = self.config.matcher_weightings
        else:
            weightings = {x: 1. for x in self.matcher_analyses}


        # Create an imputer object for handeling Nan values.
        imp = Imputer(axis=0, strategy='median')

        for tind, target_entry in enumerate(self.target_db.analysed_audio):
            # Check if match data already exists and use it rather than
            # regenerating if it does.
            if target_entry.name in self.output_db.data["match"].keys():
                self.logger.info("Match data already exists for {0}. Using this "
                                 "data. Run with the \'--rematch\' flag to "
                                 "overwrite.".format(self.output_db))
                continue

            # Create an array of grain times for target sample
            target_times = target_entry.times
            x_size = target_times.shape[0]
            match_indexes = np.empty((x_size, self.match_quantity), dtype=int)
            match_vals = np.empty((x_size, self.match_quantity))
            match_vals.fill(np.inf)

            # Allocate memory for target analyses.
            all_target_analyses = np.empty((len(self.matcher_analyses), target_times.shape[0]))

            for i, analysis in enumerate(self.matcher_analyses):
                analysis_formatting = self.analysis_dict[analysis]

                target_data, s = target_entry.analysis_data_grains(target_times, analysis, format=analysis_formatting)
                target_data *= weightings[analysis]
                all_target_analyses[i] = target_data


            nan_columns = np.all(np.isnan(all_target_analyses), axis=0)
            all_target_analyses[:, nan_columns] = 0.
            # Impute values for Nans
            all_target_analyses = imp.fit_transform(all_target_analyses)
            # all_target_analyses[np.isnan(all_target_analyses)] = np.inf

            for sind, source_entry in enumerate(self.source_db.analysed_audio):
                self.logger.info("K-d Tree Matching: {0} to {1}".format(source_entry.name, target_entry.name))
                # Create an array of grain times for source sample
                source_times = source_entry.times
                if not source_times.size:
                    continue

                all_source_analyses = np.empty((len(self.matcher_analyses), source_times.shape[0]))


                for i, analysis in enumerate(self.matcher_analyses):
                    analysis_formatting = self.analysis_dict[analysis]

                    source_data, s = source_entry.analysis_data_grains(source_times, analysis, format=analysis_formatting)
                    source_data *= weightings[analysis]
                    all_source_analyses[i] = source_data

                # Impute values for Nans
                nan_columns = np.all(np.isnan(all_source_analyses), axis=0)
                all_source_analyses[:, nan_columns] = 0.
                all_source_analyses = imp.fit_transform(all_source_analyses)

                # all_source_analyses[np.isnan(all_source_analyses)] = np.inf

                source_tree = spatial.cKDTree(all_source_analyses.T, leafsize=100)
                results_vals, results_inds = source_tree.query(all_target_analyses.T, k=self.match_quantity, p=2)

                if len(results_vals.shape) < 2:
                    results_vals = np.array([results_vals]).T
                    results_inds = np.array([results_inds]).T

                vals_append = np.append(match_vals, results_vals, axis=1)
                vals_sort = np.argsort(vals_append)
                #TODO: Check that this minus 1 should be there...
                inds_append = np.append(match_indexes, results_inds+int(source_sample_indexes[sind][0]), axis=1)

                m = np.arange(len(vals_append))[:, np.newaxis]
                best_match_inds = inds_append[m, vals_sort]
                match_indexes = best_match_inds[:, :self.match_quantity]
                best_match_vals = vals_append[m, vals_sort]
                match_vals = best_match_vals[:, :self.match_quantity]

            match_grain_inds = self.calculate_db_inds(match_indexes, source_sample_indexes)

            ###################################################################
            # Find optimum continuity between selection of best matches per
            # grain
            ###################################################################

            final_indexes = np.array([])
            # For every match found
            for h, match_indexes in enumerate(match_grain_inds[:-1]):
                self.logger.info("Calculating grain distances for grain: {0} of {1}".format(h, match_grain_inds.shape[0]))
                # Get all analysis data for each match
                all_source_analyses = np.zeros((len(self.matcher_analyses), self.match_quantity))
                for i, analysis in enumerate(self.matcher_analyses):
                    analysis_formatting = self.analysis_dict[analysis]

                    for j, (db_ind, grain_ind) in enumerate(match_indexes):
                        grain_time = self.source_db.analysed_audio[db_ind].times[grain_ind-1]
                        analysis_val, s = self.source_db.analysed_audio[db_ind].analysis_data_grains(grain_time, analysis, format=analysis_formatting)
                        analysis_val *= weightings[analysis]
                        all_source_analyses[i][j] = analysis_val

                # get the next grain's match indexes...
                next_grain_indexes = match_grain_inds[h+1]
                # Get all analysis data for each match
                next_grain_analyses = np.zeros((len(self.matcher_analyses), self.match_quantity))
                for i, analysis in enumerate(self.matcher_analyses):
                    analysis_formatting = self.analysis_dict[analysis]

                    for j, (db_ind, grain_ind) in enumerate(next_grain_indexes):
                        grain_time = self.source_db.analysed_audio[db_ind].times[grain_ind-1]
                        analysis_val, s = self.source_db.analysed_audio[db_ind].analysis_data_grains(grain_time, analysis, format=analysis_formatting)
                        analysis_val *= weightings[analysis]
                        next_grain_analyses[i][j] = analysis_val

                # Impute values for Nans
                nan_columns = np.all(np.isnan(all_source_analyses), axis=0)
                all_source_analyses[:, nan_columns] = 0.
                all_source_analyses = imp.fit_transform(all_source_analyses)
                # Impute values for Nans
                nan_columns = np.all(np.isnan(next_grain_analyses), axis=0)
                next_grain_analyses[:, nan_columns] = 0.
                next_grain_analyses = imp.fit_transform(next_grain_analyses)

                source_tree = spatial.cKDTree(all_source_analyses.T, leafsize=100)
                # Return array of distances and indexes for matches in the next
                # grain that are closest to matches in the current grains.
                results_vals, results_inds = source_tree.query(next_grain_analyses.T, k=1, p=2)

                if len(results_vals.shape) < 2:
                    results_vals = np.array([results_vals]).T
                    results_inds = np.array([results_inds]).T

                if not final_indexes.size:
                    a = np.argmax(results_vals)
                    final_indexes = np.vstack((match_indexes[a], match_indexes[results_inds[a]]))
                else:
                    a = np.argmax(results_vals)
                    final_indexes = np.append(final_indexes, match_indexes[results_inds[a]], axis=0)



            match_grain_inds = final_indexes
                # For each analysis in current match and previous match...
                # build kd tree for all grains of current analysis in current match

                # Calculate distance between all previous grains and current
                # grains

                # Accumulate distance array with distance multiplied by
                # weighting

            '''
                #self.match_db.analysed_audio[match_db_ind]
                all_source_analyses = np.empty((len(self.matcher_analyses), source_times.shape[0]))


                for i, analysis in enumerate(self.matcher_analyses):
                    analysis_formatting = self.analysis_dict[analysis]

                    source_data, s = source_entry.analysis_data_grains(source_times, analysis, format=analysis_formatting)
                    source_data *= weightings[analysis]
                    all_source_analyses[i] = source_data
            '''

            ###################################################################

            datafile_path = ''.join(("match/", target_entry.name))
            try:
                self.output_db.data[datafile_path] = match_grain_inds
                self.output_db.data[datafile_path].attrs["grain_size"] = grain_size
                self.output_db.data[datafile_path].attrs["overlap"] = overlap

            except RuntimeError as err:
                raise RuntimeError("Match data couldn't be written to HDF5 "
                                   "file.\n Match data may already exist in the "
                                   "file.\n Try running with the '--rematch' flag "
                                   "to overwrite this data.\n Original error: "
                                   "{0}".format(err))



    def brute_force_matcher(self, grain_size, overlap):
        '''Searches for matches to each grain by brute force comparison'''

        # Count grains of the source database
        source_sample_indexes = self.count_grains(self.source_db, grain_size, overlap)
        try:
            self.output_db.data.create_group("match")
        except ValueError:
            self.logger.debug("Match group already exists in the {0} HDF5 file.".format(self.output_db))

        if self.rematch:
            self.output_db.data["match"].clear()

        if self.config:
            weightings = self.config.matcher_weightings
        else:
            weightings = {x: 1. for x in self.matcher_analyses}

        for tind, target_entry in enumerate(self.target_db.analysed_audio):
            # Check if match data already exists and use it rather than
            # regenerating if it does.
            if target_entry.name in self.output_db.data["match"].keys():
                self.logger.info("Match data already exists for {0}. Using this "
                                 "data. Run with the \'--rematch\' flag to "
                                 "overwrite.".format(self.output_db))
                continue
            # Create an array of grain times for target sample
            target_times = target_entry.generate_grain_times(grain_size, overlap, save_times=True)

            # Allocate memory for storing accumulated distances between
            # source and target grains
            x_size = target_times.shape[0]
            y_size = int(source_sample_indexes[-1][-1])
            chunk_size = 8192



            try:
                del self.output_db.data["data_distance"]
            except KeyError:
                pass

            self.output_db.data.create_dataset("data_distance", (x_size, y_size), dtype=np.float, chunks=True)

            try:
                del self.output_db.data["distance_accum"]
            except KeyError:
                pass

            self.output_db.data.create_dataset("distance_accum", (x_size, y_size), dtype=np.float, chunks=True, fillvalue=0)

            for analysis in self.matcher_analyses:
                self.logger.info("Current analysis: {0}".format(analysis))
                analysis_formatting = self.analysis_dict[analysis]

                # Get data for all target grains for each analysis
                target_data, s = target_entry.analysis_data_grains(target_times, analysis, format=analysis_formatting)

                data_max = 0.
                for sind, source_entry in enumerate(self.source_db.analysed_audio):

                    # Get the start and end array indexes allocated for the
                    # current entry's grains.
                    start_index, end_index = source_sample_indexes[sind]

                    # Create an array of grain times for source sample
                    source_times = source_entry.generate_grain_times(grain_size, overlap, save_times=True)
                    self.logger.info("Matching \"{0}\" for: {1} to {2}".format(analysis, source_entry.name, target_entry.name))

                    # Get data for all source grains for each analysis
                    source_data, s = source_entry.analysis_data_grains(source_times, analysis, format=analysis_formatting)
                    source_entry.close()

                    # Calculate the euclidean distance between the source and
                    # source values of each grain and add to array
                    a = self.distance_calc(target_data, source_data)

                    self.output_db.data["data_distance"][:, int(start_index):int(end_index)] = a
                    self.output_db.data.flush()
                    a_max = np.max(a)
                    if a_max > data_max:
                        data_max = a_max

                # Normalize and weight the distances. A higher weighting gives
                # an analysis presedence over others.
                i = 0
                membuff = np.zeros((chunk_size, chunk_size))
                membuff2 = np.zeros((chunk_size, chunk_size))
                while i < x_size:
                    j = chunk_size
                    if i+j > x_size:
                        j = x_size - i

                    k = 0
                    while k < y_size:
                        l = chunk_size
                        if k+l > y_size:
                            l = y_size - k
                        self.logger.info("Calculating weighted "
                                         "distances:\nSource chunk {0} - {1} of "
                                         "{2}\nTarget chunk {3} - {4} of "
                                         "{5}".format(
                                             i, i+j, x_size, k, k+l, y_size
                                         ))

                        self.output_db.data["data_distance"].read_direct(membuff, np.s_[i:i+j, k:k+l], np.s_[0:j, 0:l])
                        self.output_db.data["distance_accum"].read_direct(membuff2, np.s_[i:i+j, k:k+l], np.s_[0:j, 0:l])

                        weighted_mem = membuff[0:j, 0:l] * (1/data_max) * weightings[analysis]
                        self.output_db.data["data_distance"][i:i+j, k:k+l] = weighted_mem
                        self.output_db.data["distance_accum"][i:i+j, k:k+l] = membuff2[0:j, 0:l] + weighted_mem

                        k += chunk_size

                    i += chunk_size

            self.logger.info("Calculating the closest {0} overall matches...". format(self.match_quantity))
            i = 0
            # Allocate memory for storing chunks.
            chunk_vals = np.zeros((chunk_size, chunk_size))
            chunk_inds = np.tile(np.arange(chunk_size), (chunk_size, 1))
            # Allocate memory for storing the best matches for each target
            # grain
            match_indexes = np.empty((x_size, self.match_quantity))
            match_indexes.fill(np.nan)
            # Allocate memory for storing the match distance of these grains.
            match_vals = np.empty((x_size, self.match_quantity))
            match_vals.fill(np.inf)


            while i < x_size:
                j = chunk_size
                if i+j > x_size:
                    j = x_size - i

                k = 0
                while k < y_size:
                    l = chunk_size
                    if k+l > y_size:
                        l = y_size - k
                    self.logger.info("Calculating best overall "
                                        "matches:\nSource chunk {0} - {1} of "
                                        "{2}\nTarget chunk {3} - {4} of "
                                        "{5}".format(
                                            i, i+j, x_size, k, k+l, y_size
                                        ))

                    # Read the current chunk to memory
                    self.output_db.data["distance_accum"].read_direct(chunk_vals, np.s_[i:i+j, k:k+l], np.s_[0:j, 0:l])

                    # Append the chunk distances to the best matches so far.
                    vals_append = np.append(match_vals[i:i+j], chunk_vals[i:i+j], axis=1)
                    c_inds = chunk_inds[0:j] + i

                    m = np.arange(len(vals_append))[:, np.newaxis]
                    # Sort all values to intergrate new chunk distances with
                    # previous best matches.
                    vals_sort = np.argsort(vals_append, axis=1)
                    inds_append = np.append(match_indexes[i:i+j], c_inds, axis=1)
                    best_match_inds = inds_append[m, vals_sort]
                    match_indexes[i:i+j] = best_match_inds[:, :self.match_quantity]
                    best_match_vals = vals_append[m, vals_sort]
                    match_vals[i:i+j] = best_match_vals[:, :self.match_quantity]
                    k += chunk_size
                i += chunk_size

            match_grain_inds = self.calculate_db_inds(match_indexes, source_sample_indexes)

            # Generate the path to the data group that will store the match
            # data in the HDF5 file.
            datafile_path = ''.join(("match/", target_entry.name))

            try:
                self.output_db.data[datafile_path] = match_grain_inds
                self.output_db.data[datafile_path].attrs["grain_size"] = grain_size
                self.output_db.data[datafile_path].attrs["overlap"] = overlap

            except RuntimeError as err:
                raise RuntimeError("Match data couldn't be written to HDF5 "
                                   "file.\n Match data may already exist in the "
                                   "file.\n Try running with the '--rematch' flag "
                                   "to overwrite this data.\n Original error: "
                                   "{0}".format(err))


    def distance_calc(self, data1, data2):
        """
        Calculates the euclidean distance between two arrays of data.

        Distance is calculated with special handeling of Nan values, if they exist in the data.
        A Nan value matched to another Nan value is classed as a perfect match.
        a Nan matched to any other value is calculated as the furthest match +
        10%
        """
        # Find all numbers that aren't Nan, inf, None etc...
        data1_finite_inds = np.isfinite(data1)
        data2_finite_inds = np.isfinite(data2)
        # Find all special numbers
        data1_other_inds = data1_finite_inds == False
        data2_other_inds = data2_finite_inds == False

        # Calculate euclidean distances between the two data arrays.
        distances = np.abs(np.vstack(data1)-data2)**2
        # Find the largest non-Nan distance
        # If all values ar nan then skip this.
        if distances[np.isfinite(distances)].size:
            largest_distance = np.max(distances[np.isfinite(distances)])
        else:
            largest_distance = 1.

        # Find grains where both the source and target values are Nan.
        nan_intersects = np.vstack(data1_other_inds) & data2_other_inds

        # Set these grain's distances to 0 as they match.
        distances[nan_intersects] = 0.

        distances[np.isnan(distances)] = largest_distance + (largest_distance*0.1)

        return distances

    def calculate_db_inds(self, match_indexes, source_sample_indexes):
        """
        Generate the database sample index and grain index for each match based
        on their indexes generated from the concatenated matching

        Output array will be a 3 dimensional array with an axis for each target
        grain, a dimension for each match of said grain and a dimension
        containing database sample index and the sample's grain index.
        """

        # source_sample_indexes = source_sample_indexes[source_sample_indexes[:, 1] != 0.]

        mi_shape = match_indexes.shape
        x = match_indexes.flatten()
        # Find indexes within the range of each source sample index.
        x = np.logical_and(
            np.vstack(x)>=source_sample_indexes[:,0],
            np.vstack(x)<=source_sample_indexes[:,1]
        )

        if not np.all(np.any(x, axis=1)):
            raise ValueError("Not all match indexes have a corresponding sample index. This shouldn't happen...\n"
                             "Check that all database path arguments are correct then try re-running with the --rematch and --reanalyse flags.\n"
                             "If this doesn't work, delete the audio and data directories in all databases and try again...")

        x = x.reshape(mi_shape[0], mi_shape[1], x.shape[1])
        x = np.argmax(x, axis=2)

        # Calculate sample index in database
        match_start_inds = source_sample_indexes[x.flatten(), 0].reshape(mi_shape)
        # Calculate grain index offset from the start of the sample
        match_grain_inds = match_indexes.reshape(mi_shape) - match_start_inds

        return np.dstack((x, match_grain_inds)).astype(int)

    def swap_databases(self):
        """Convenience method to swap databases, changing the source database into the target and vice-versa"""
        self.source_db, self.target_db = self.target_db, self.source_db


class Synthesizer:

    """An object used for synthesizing output based on grain matching."""

    def __init__(self, database1, database2, *args, **kwargs):
        """Initialize synthesizer instance"""
        self.logger = logging.getLogger(__name__ + '.Matcher')

        self.match_db = database1
        self.output_db = database2
        self.target_db = kwargs.pop("target_db", None)

        self.config = kwargs.pop("config", None)

        self.enforce_intensity_bool = self.config.synthesizer["enforce_intensity"]
        # Key word arguments overwrite config file.
        self.enforce_intensity_bool = kwargs.pop("enforce_intensity", self.enforce_intensity_bool)
        if self.enforce_intensity_bool and ("rms" not in self.target_db.analysis_list or "rms" not in self.match_db.analysis_list):
            raise RuntimeError("BLARGHHH")

        self.enforce_f0_bool = self.config.synthesizer["enforce_f0"]
        # Key word arguments overwrite config file.
        self.enforce_f0_bool = kwargs.pop("enforce_f0", self.enforce_f0_bool)
        if self.enforce_f0_bool and ("f0" not in self.target_db.analysis_list or "f0" not in self.match_db.analysis_list):
            raise RuntimeError("F0 enforcement cannot be enabled if both databases do not have F0 analyses.")

        if self.enforce_intensity:
            if not self.target_db:
                raise ValueError("Target database must be provided if rms or F0 enforcement is enabled.")

    def synthesize(self, grain_size=None, overlap=None):
        """
        Synthesized output from the match data in the output database to create
        audio in the output database.
        """
        if not grain_size:
            grain_size = self.config.synthesizer["grain_size"]
        if not overlap:
            overlap = self.config.synthesizer["overlap"]
        jobs = [(i, self.output_db.data["match"][i]) for i in self.output_db.data["match"]]
        # TODO: insert error here if there are no jobs.
        if not jobs:
            raise RuntimeError("There is no match data to synthesize. The match program may need to be run first.")

        for job_ind, (name, job) in enumerate(jobs):
            # Generate output file name/path
            filename, extension = os.path.splitext(name)
            output_name = ''.join((filename, '_output', extension))
            output_path = os.path.join(self.output_db.subdirs["audio"], output_name)
            # Create audio file to save output to.
            output_config = self.config.output_file
            grain_matches = self.output_db.data["match"][name]
            # Get the grain size and overlap used for analysis.
            match_grain_size = grain_matches.attrs["grain_size"]
            match_overlap = grain_matches.attrs["overlap"]

            _grain_size = grain_size
            with AudioFile(
                output_path,
                "w",
                samplerate=output_config["samplerate"],
                format=output_config["format"],
                channels=output_config["channels"]
            ) as output:
                hop_size = int(np.floor(grain_size / overlap))
                output_frames = np.zeros(_grain_size*2 + (int(hop_size*len(grain_matches))))
                offset = 0
                for target_grain_ind, matches in enumerate(grain_matches):
                    # If there are multiple matches, choose a match at random
                    # from available matches.
                    #match_index = np.random.randint(matches.shape[0])
                    match_db_ind, match_grain_ind = matches
                    with self.match_db.analysed_audio[match_db_ind] as match_sample:
                        self.logger.info("Synthesizing grain:\n"
                            "Source sample: {0}\n"
                            "Source grain index: {1}\n"
                            "Target output: {2}\n"
                            "Target grain index: {3} out of {4}".format(
                                match_sample,
                                match_grain_ind,
                                output_name,
                                target_grain_ind,
                                len(grain_matches)
                            ))
                        match_sample.generate_grain_times(match_grain_size, match_overlap, save_times=True)

                        # TODO: Make proper fix for grain index offset of 1
                        try:
                            match_grain = match_sample[match_grain_ind-1]
                        except:
                            pdb.set_trace()


                        if self.enforce_intensity_bool:
                            # Get the target sample from the database
                            target_sample = self.target_db[job_ind]

                            # Calculate garin times for sample to allow for
                            # indexing.
                            target_sample.generate_grain_times(match_grain_size, match_overlap, save_times=True)

                            match_grain = self.enforce_intensity(match_grain, match_sample, match_grain_ind, target_sample, target_grain_ind)

                        if self.enforce_f0_bool:
                            # Get the target sample from the database
                            target_sample = self.target_db[job_ind]

                            # Calculate grain times for sample to allow for
                            # indexing.
                            target_sample.generate_grain_times(match_grain_size, match_overlap, save_times=True)

                            match_grain = self.enforce_pitch(match_grain, match_sample, match_grain_ind, target_sample, target_grain_ind)

                        # Apply hanning window to grain
                        match_grain *= np.hanning(match_grain.size)
                        try:
                            output_frames[offset:offset+match_grain.size] += match_grain
                        except:
                            pass
                    offset += hop_size
                # If output normalization is active, normalize output.
                if self.config.synthesizer["normalize"]:
                    output_frames = (output_frames / np.max(np.abs(output_frames))) * 0.9
                output.write_frames(output_frames)

    def enforce_pitch(self, grain, source_sample, source_grain_ind, target_sample, target_grain_ind):
        """
        Shifts the pitch of the grain by the difference between it's f0 and the f0 of the grain specified.

        This method will fail if either AnalysedAudioFile object does not have an f0 analysis.
        """

        # Get grain start and finish range to retreive analysis frames from.
        # TODO: Make proper fix for grain index offset of 1
        target_times = target_sample.times[target_grain_ind-1]

        # Get mean of f0 frames in time range specified.
        target_f0 = target_sample.analysis_data_grains(target_times, "f0", format="median")[0][0]

        # Get grain start and finish range to retreive analysis frames from.
        # TODO: Make proper fix for grain index offset of 1
        source_times = source_sample.times[source_grain_ind-1]



        # Get mean of f0 frames in time range specified.
        source_f0 = source_sample.analysis_data_grains(source_times, "f0", format="median")[0][0]

        f0_array = np.array([source_f0, target_f0])
        if np.any(np.isnan(f0_array)):
            return grain*0
        ratio_difference = target_f0 / source_f0

        if not np.isfinite(ratio_difference):
            return grain*0

        # If the ratio difference is within the limits
        ratio_limit = self.config.synthesizer["enf_f0_ratio_limit"]

        if ratio_difference > ratio_limit:
            self.logger.warning("Grain f0 ratio too large({0}), enforcing f0 at limit ({1})".format(
                                    ratio_difference,
                                    ratio_limit,
                                ))
            ratio_difference = ratio_limit
        elif ratio_difference < 1./ratio_limit:
            self.logger.warning("Grain f0 ratio too large ({0}), enforcing f0 at limit ({1})".format(
                                    ratio_difference,
                                    1./ratio_limit,
                                ))
            ratio_difference = 1./ratio_limit

        grain = pitch_shift.shift(grain, ratio_difference)
        '''
        if ratio_difference > ratio_limit or ratio_difference < 1./ratio_limit:
            grain *= 0
        '''

        return grain

    def enforce_intensity(self, grain, source_sample, source_grain_ind, target_sample, target_grain_ind):
        """
        Scales the amplitude of the grain by the difference between it's intensity and the intensity of the grain specified.

        This method will fail if either AnalysedAudioFile object does not have any intensity analyses.
        """

        # Get grain start and finish range to retreive analysis frames from.
        # TODO: Make proper fix for grain index offset of 1
        target_times = target_sample.times[target_grain_ind-1]

        # Get mean of RMS frames in time range specified.
        target_rms = target_sample.analysis_data_grains(target_times, "rms", format="mean")[0][0]

        target_intensity_value = target_rms

        # Get grain start and finish range to retreive analysis frames from.
        # TODO: Make proper fix for grain index offset of 1
        source_times = source_sample.times[source_grain_ind-1]

        # Get mean of RMS frames in time range specified.
        source_rms = source_sample.analysis_data_grains(source_times, "rms", format="mean")[0][0]

        source_intensity_value = np.mean(source_rms)

        ratio_difference = target_intensity_value / source_intensity_value

        if not np.isfinite(ratio_difference):
            return grain
        # If the ratio difference is within the limits
        ratio_limit = self.config.synthesizer["enf_intensity_ratio_limit"]

        if ratio_difference > ratio_limit:
            self.logger.warning(
                "Grain RMS ratio too large({0}), enforcing RMS at limit ({1})\n".format(
                ratio_difference,
                ratio_limit,
            ))
            ratio_difference = ratio_limit

        grain *= ratio_difference


        return grain

    def swap_databases(self):
        """Convenience method to swap databases, changing the source database into the target and vice-versa"""
        self.match_db, self.output_db = self.output_db, self.match_db


