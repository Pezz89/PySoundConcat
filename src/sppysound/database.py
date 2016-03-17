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
import pitch_shift

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

    """A class for encapsulating a database of AnalysedAudioFile objects."""

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
            'harm_ratio'
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
            self.organize_audio(subdir_paths)

        self.analyse_database(subdir_paths, reanalyse)

    def analyse_database(self, subdir_paths, reanalyse):
        """
        create selected analyses for audio files in the database.

        Parameters:
            subdir_paths: a dictionary containing paths to the 'audio' directory and 'data' directory of the database.
            reanalyse: If previous analyses are found this can be set to True to overwrite them.
        """
        # Create data file for storing analysis data for the database
        datapath = os.path.join(subdir_paths['data'], 'analysis_data.hdf5')
        self.data = h5py.File(datapath, 'a')
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
            subdir_paths: A dictionary of 'audio' and 'data' sub directories of the database.
            symlink: If true, symbolic links will be used for audio files
            rather than copying. This allows audio files to stay in there
            origonal locations.
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
                            shutil.copy2(filepath, subdir_paths["audio"])
                            self.logger.info(''.join(("Moved: ", item, "\tTo directory: ",
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
            length = entry.samps_to_ms(entry.frames)
            hop_size = grain_length / overlap
            grain_indexes[ind][0] = int(length / hop_size) - 1
        grain_indexes[:, 1] = np.cumsum(grain_indexes[:, 0])
        grain_indexes[:, 0] = grain_indexes[:, 1] - grain_indexes[:, 0]
        return grain_indexes

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
        #
        final_match_indexes = []

        if self.config:
            weightings = self.config.matcher_weightings
        else:
            weightings = {x: 1. for x in self.matcher_analyses}

        for tind, target_entry in enumerate(self.target_db.analysed_audio):
            # Create an array of grain times for target sample
            target_times = target_entry.generate_grain_times(grain_size, overlap)

            # Stores an accumulated distance between source and target grains,
            # added to by each analysis.
            distance_accum = np.zeros((target_times.shape[0], source_sample_indexes[-1][-1]))
            for analysis in self.matcher_analyses:
                self.logger.info("Current analysis: {0}".format(analysis))
                analysis_formatting = self.analysis_dict[analysis]
                # Get the analysis object for the current entry
                analysis_object = target_entry.analyses[analysis]


                # Get data for all target grains for each analysis
                target_data, s = target_entry.analysis_data_grains(target_times, analysis, format=analysis_formatting)

                # Allocate memory for storing accumulated distances between
                # source and target grains
                self.data_distance = np.zeros((target_data.shape[0], source_sample_indexes[-1][-1]))

                for sind, source_entry in enumerate(self.source_db.analysed_audio):

                    # Get the start and end array indexes allocated for the
                    # current entry's grains.
                    start_index, end_index = source_sample_indexes[sind]

                    # Create an array of grain times for source sample
                    source_times = source_entry.generate_grain_times(grain_size, overlap)
                    self.logger.info("Matching \"{0}\" for: {1} to {2}".format(analysis, source_entry.name, target_entry.name))

                    # Get data for all source grains for each analysis
                    source_data, s = source_entry.analysis_data_grains(source_times, analysis, format=analysis_formatting)

                    # Calculate the euclidean distance between the source and
                    # source values of each grain and add to array
                    a = self.distance_calc(target_data, source_data)

                    self.data_distance[:, start_index:end_index] = a

                # Normalize and weight the distances. A higher weighting gives
                # an analysis presedence over others.
                self.data_distance *= (1/self.data_distance.max()) * weightings[analysis]
                distance_accum += self.data_distance

            # Sort indexes so that best matches are at the start of the array.
            match_indexes = distance_accum.argsort(axis=1)[:, :self.match_quantity]

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
        return match_grain_inds


    def distance_calc(self, data1, data2):
        """
        Calculates the euclidean distance between two arrays of data.

        Distance is calculated with special handeling of Nan values, if they exist in the data.
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
        mi_shape = match_indexes.shape
        x = match_indexes.flatten()
        x = np.logical_and(
            np.vstack(x)>=source_sample_indexes[:,0],
            np.vstack(x)<=source_sample_indexes[:,1]
        )
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

        self.config = kwargs.pop("config", None)

        self.enforce_rms_bool = self.config.synthesizer["enforce_rms"]
        # Key word arguments overwrite config file.
        self.enforce_rms_bool = kwargs.pop("enforce_rms", self.enforce_rms_bool)

        self.enforce_f0_bool = self.config.synthesizer["enforce_f0"]
        # Key word arguments overwrite config file.
        self.enforce_f0_bool = kwargs.pop("enforce_f0", self.enforce_f0_bool)

        self.target_db = kwargs.pop("target_db", None)
        if self.enforce_rms:
            if not self.target_db:
                raise ValueError("Target database must be provided if rms or F0 enforcement is enabled.")

    def synthesize(self, grain_size=None, overlap=None):
        """
        Takes a 3D array containing the sample and grain indexes for each grain to be synthesized.

        If grain size or overlap isn't specified, the values from config are used.
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
                hop_size = (grain_size / overlap) * output.samplerate/1000
                _grain_size *= output.samplerate / 1000
                output_frames = np.zeros(_grain_size + (hop_size*len(grain_matches)-1))
                offset = 0
                for target_grain_ind, matches in enumerate(grain_matches):
                    # If there are multiple matches, choose a match at random
                    # from available matches.
                    match_index = np.random.randint(matches.shape[0])
                    match_db_ind, match_grain_ind = matches[match_index]
                    with self.match_db.analysed_audio[match_db_ind] as match_sample:
                        match_sample.generate_grain_times(match_grain_size, match_overlap)

                        # TODO: Make proper fix for grain index offset of 1
                        match_grain = match_sample[match_grain_ind-1]

                        if self.enforce_rms_bool:
                            # Get the target sample from the database
                            target_sample = self.target_db[job_ind]

                            # Calculate garin times for sample to allow for
                            # indexing.
                            target_sample.generate_grain_times(match_grain_size, match_overlap)

                            match_grain = self.enforce_rms(match_grain, match_sample, match_grain_ind, target_sample, target_grain_ind)

                        if self.enforce_f0_bool:
                            # Get the target sample from the database
                            target_sample = self.target_db[job_ind]

                            # Calculate grain times for sample to allow for
                            # indexing.
                            target_sample.generate_grain_times(match_grain_size, match_overlap)

                            match_grain = self.enforce_pitch(match_grain, match_sample, match_grain_ind, target_sample, target_grain_ind)

                        match_grain *= np.hanning(match_grain.size)
                        output_frames[offset:offset+match_grain.size] += match_grain
                    offset += hop_size
                # If output normalization is active, normalize output.
                if self.config.synthesizer["normalize"]:
                    output_frames = (output_frames / np.max(np.abs(output_frames))) * 0.9
                output.write_frames(output_frames)

    def enforce_pitch(self, grain, source_sample, source_grain_ind, target_sample, target_grain_ind):
        """Shifts the pitch of the grain by the difference between it's f0 and the f0 of the grain specified."""

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

        ratio_difference = target_f0 / source_f0

        if not np.isfinite(ratio_difference):
            return grain

        # If the ratio difference is within the limits
        ratio_limit = self.config.synthesizer["enf_f0_ratio_limit"]

        if ratio_difference > ratio_limit:
            self.logger.warning("Grain f0 ratio too large({0}), enforcing f0 at limit ({1})\n"
                                "Source sample: {2}\n"
                                "Source grain index: {3}\n"
                                "Target sample: {4}\n"
                                "Target grain index: {5}".format(
                                    ratio_difference,
                                    ratio_limit,
                                    source_sample,
                                    source_grain_ind,
                                    target_sample,
                                    target_grain_ind
                                ))
            ratio_difference = ratio_limit
        elif ratio_difference < 1./ratio_limit:
            self.logger.warning("Grain f0 ratio too large ({0}), enforcing f0 at limit ({1})\n"
                                "Source sample: {2}\n"
                                "Source grain index: {3}\n"
                                "Target sample: {4}\n"
                                "Target grain index: {5}".format(
                                    ratio_difference,
                                    1./ratio_limit,
                                    source_sample,
                                    source_grain_ind,
                                    target_sample,
                                    target_grain_ind
                                ))
            ratio_difference = 1./ratio_limit

        grain = pitch_shift.shift(grain, ratio_difference)

        return grain

    def enforce_rms(self, grain, source_sample, source_grain_ind, target_sample, target_grain_ind):
        """Scales the amplitude of the grain by the difference between it's rms and the rms of the grain specified."""

        # Get grain start and finish range to retreive analysis frames from.
        # TODO: Make proper fix for grain index offset of 1
        target_times = target_sample.times[target_grain_ind-1]

        # Get mean of RMS frames in time range specified.
        target_rms = target_sample.analysis_data_grains(target_times, "rms", format="mean")[0][0]
        target_peak = target_sample.analysis_data_grains(target_times, "peak", format="mean")[0][0]
        tval = np.mean([target_rms, target_peak])

        # Get grain start and finish range to retreive analysis frames from.
        # TODO: Make proper fix for grain index offset of 1
        source_times = source_sample.times[source_grain_ind-1]

        # Get mean of RMS frames in time range specified.
        source_rms = source_sample.analysis_data_grains(source_times, "rms", format="mean")[0][0]
        source_peak = source_sample.analysis_data_grains(source_times, "peak", format="mean")[0][0]
        sval = np.mean([source_rms, source_peak])

        ratio_difference = tval / sval
        if not np.isfinite(ratio_difference):
            return grain
        # If the ratio difference is within the limits
        ratio_limit = self.config.synthesizer["enf_rms_ratio_limit"]

        if ratio_difference > ratio_limit:
            self.logger.warning("Grain RMS ratio too large({0}), enforcing RMS at limit ({1})\n"
                                "Source sample: {2}\n"
                                "Source grain index: {3}\n"
                                "Target sample: {4}\n"
                                "Target grain index: {5}".format(
                                    ratio_difference,
                                    ratio_limit,
                                    source_sample,
                                    source_grain_ind,
                                    target_sample,
                                    target_grain_ind
                                ))
            ratio_difference = ratio_limit

        grain *= ratio_difference

        return grain

    def swap_databases(self):
        """Convenience method to swap databases, changing the source database into the target and vice-versa"""
        self.match_db, self.output_db = self.output_db, self.match_db


