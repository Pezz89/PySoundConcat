from __future__ import print_function
import os
import numpy as np
import logging
import pdb

from fileops import pathops

logger = logging.getLogger(__name__)

class Analysis(object):

    """Basic descriptor class to build analyses on."""

    def __init__(self, AnalysedAudioFile, analysis_group, name, config=None):
        # Create object logger
        self.logger = logging.getLogger(__name__ + '.{0}Analysis'.format(name))
        # Store AnalysedAudioFile object to be analysed.
        self.AnalysedAudioFile = AnalysedAudioFile
        self.analysis_group = analysis_group
        self.name = name

    def create_analysis(self, *args, **kwargs):
        """
        Create the analysis and save to the HDF5 file.

        analysis_function: The function used to create the analysis. returned
        data will be stored in the HDF5 file.
        """

        try:
            self.analysis = self.analysis_group.create_group(self.name)
        except ValueError:
            self.logger.info("{0} analysis group already exists".format(self.name))
            self.analysis = self.analysis_group[self.name]

        # If forcing new analysis creation then delete old analysis and create
        # a new one
        if self.AnalysedAudioFile.force_analysis:
            self.logger.info("Force re-analysis is enabled. "
                                "deleting: {0}".format(self.analysis.name))
            # Delete all pre-existing data in database.
            for i in self.analysis.iterkeys():
                del self.analysis[i]
            # Run the analysis function and format it's returned data ready to
            # be saved in the HDF5 file
            data_dict, attrs_dict = self.hdf5_dataset_formatter(*args, **kwargs)
            for key, value in data_dict.iteritems():
                self.analysis.create_dataset(key, data=value)
            for key, value in attrs_dict.iteritems():
                self.analysis.attrs[key] = value
        else:

            if self.analysis.items():
                self.logger.info("Analysis already exists. Reading from: "
                                 "{0}".format(self.analysis.name))
            else:
                # If it doesn't then generate a new file
                # Run the analysis function and format it's returned data ready to
                # be saved in the HDF5 file
                data_dict, attrs_dict = self.hdf5_dataset_formatter(*args, **kwargs)
                for key, value in data_dict.iteritems():
                    self.analysis.create_dataset(key, data=value)
                for key, value in attrs_dict.iteritems():
                    self.analysis.attrs[key] = value

    def get_analysis_grains(self, start, end):
        """
        Retrieve analysis frames for period specified in start and end times.
        arrays of start and end time pairs will produce an array of equivelant
        size containing frames for these times.
        """
        times = self.analysis_group[self.name]["times"][:]
        start = start / 1000
        end = end / 1000
        vtimes = times.reshape(-1, 1)

        selection = np.transpose((vtimes >= start) & (vtimes <= end))

        #start_ind = np.min(selection)
        #end_ind = np.argmax(selection)
        frames = self.analysis_group[self.name]["frames"][:]

        grain_data = (frames, selection)

        return grain_data

    def hdf5_dataset_formatter(analysis_method, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.

        Places data and attributes in 2 dictionaries to be stored in the HDF5
        file.
        Note: This is a generic formatter designed as a template to be
        overwritten by a descriptor sub-class.
        '''
        output, attributes = analysis_method(*args, **kwargs)
        return ({'data': output}, {'attrs': attributes})

    ################################################################################
    # Formatting functions
    ################################################################################

    def log2_median(self, x):
        return np.log2(np.median(x))

    def log2_mean(self, x):
        return np.log2(np.mean(x))

    def formatter_func(self, selection, frames, valid_inds, formatter=None):
        # get all valid frames from current grain
        frames = frames[selection & valid_inds]

        #if less than half the frames are valid then the grain is not valid.
        if frames.size < valid_inds[selection].nonzero()[0].size/2:
            return np.nan
        else:
            return formatter(frames)
        return formatter(frames)

    def analysis_formatter(self, frames, selection, format):
        """Calculate the average analysis value of the grain using the match format specified."""
        valid_inds = np.isfinite(frames)

        format_style_dict = {
            'mean': np.mean,
            'median': np.median,
            'log2_mean': self.log2_mean,
            'log2_median': self.log2_median,
        }
        output = np.empty(len(selection))
        for ind, i in enumerate(selection):
            output[ind] = self.formatter_func(i, frames, valid_inds, formatter=format_style_dict[format])
        # output = np.apply_along_axis(self.formatter_func, 1, selection, frames, valid_inds, formatter=format_style_dict[format])
        return output
