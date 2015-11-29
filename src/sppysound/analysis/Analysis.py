from __future__ import print_function
import os
import numpy as np
import logging
import pdb

from fileops import pathops

logger = logging.getLogger(__name__)

class Analysis(object):

    """Basic descriptor class to build analyses on."""

    def __init__(self, AnalysedAudioFile, analysis_group, name):
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
            self.logger.warning("{0} analysis group already exists".format(self.name))
            self.analysis = self.analysis_group[self.name]

        # If forcing new analysis creation then delete old analysis and create
        # a new one
        if self.AnalysedAudioFile.force_analysis:
            self.logger.warning("Force re-analysis is enabled. "
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
            # Check if analysis file already exists.
            try:
                if self.analysis['data']:
                    self.logger.info("Analysis already exists. "
                                    "Reading from: {0}".format(self.analysis.name))
                    # If an analysis file is provided then count the number of lines
                    # (1 for each window)
                    self.logger.info(''.join(("Reading {0} data: "
                                            "(HDF5 File)".format(self.name),
                                            self.analysis.name)))
                    self.window_count = self.analysis['data'].size
                    self.logger.debug(''.join(("{0} Window Count: ".format(self.name),
                                            str(self.window_count))))
            except KeyError:
                # If it doesn't then generate a new file
                # Run the analysis function and format it's returned data ready to
                # be saved in the HDF5 file
                data_dict, attrs_dict = self.hdf5_dataset_formatter(*args, **kwargs)
                for key, value in data_dict.iteritems():
                    self.analysis.create_dataset(key, data=value)
                for key, value in attrs_dict.iteritems():
                    self.analysis.attrs[key] = value

    def hdf5_dataset_formatter(analysis_method, *args, **kwargs):
        '''
        Formats the output from the analysis method to save to the HDF5 file.

        Places data and attributes in 2 dictionaries to be stored in the HDF5
        file.
        Note: This is a generic formatter designed as a template to be
        overwritten by a descriptor sub-class.
        '''
        output = analysis_method(*args, **kwargs)
        return ({'data' : output}, {'attrs' : None})
