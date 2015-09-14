from __future__ import print_function
import os
import numpy as np
import math
import pdb

from fileops import pathops


class AttackAnalysis:

    """Encapsulation of attack estimation analysis."""

    def __init__(self, AnalysedAudioFile, atkpath):
        self.AnalysedAudioFile = AnalysedAudioFile
        self.attackpath = atkpath
        self.attack_start = None
        self.attack_end = None
        self.attack_size = None
        self.logattacktime = None
        # Check if analysis file already exists.
        # TODO: check if RMS has changed, if it has then new values will need
        # to be generated even if a file already exists.
        if not self.attackpath:
            if not self.AnalysedAudioFile.db_dir:
                raise IOError("Analysed Audio object must have an atk file"
                              " path or be part of a database")
            self.attackpath = os.path.join(
                self.AnalysedAudioFile.db_dir,
                "atk",
                self.AnalysedAudioFile.name +
                ".lab")
        if self.AnalysedAudioFile.force_analysis:
            pathops.delete_if_exists(self.attackpath)
            self.attackpath = self.create_attack_analysis()
        else:
            try:
                # If it does then get values from file
                self.get_attack_from_file()
            except IOError:
                # Otherwise, generate new values
                self.create_attack_analysis()

    def create_attack_analysis(self, multiplier=3):
        """
        Estimate the start and end of the attack of the audio.

        Adaptive threshold method (weakest effort method) described here:
        http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf
        Stores values in a file at the attack path provided with the following
        format:
        attack_start attack_end
        """
        pdb.set_trace()
        # Make sure RMS has been calculated
        if not self.AnalysedAudioFile.RMS:
            raise IOError("RMS analysis is required to estimate attack")
        with open(self.attackpath, 'w') as attackfile:
            print("Creating attack estimation file:\t\t",
                  os.path.relpath(self.attackpath))
            rms_contour = self.AnalysedAudioFile.RMS.get_rms_from_file()
            # Scale RMS contour to range so all calculations are performed in
            # the range 0.0 to 1.0
            # TODO: Should calculations be done in range of rms rather than
            # converting for performance increase?
            rms_contour = self.scale_to_range(rms_contour)
            # Create a grid of thresholds ranging from 0.0 to 1.0
            thresholds = np.arange(1, 11) * 0.1
            thresholds = thresholds.reshape(-1, 1)
            # Find first index of rms that is over the threshold for each
            # thresholds
            threshold_inds = np.argmax(rms_contour >= thresholds, axis=1)

            # TODO: Need to make sure rms does not return to a lower threshold
            # after being > a threshold.

            # Calculate the time difference between each of the indexes
            ind_diffs = np.ediff1d(threshold_inds)
            # Find the average time between thresholds
            mean_ind_diff = np.mean(ind_diffs)
            # Calculate the start threshold by finding the first threshold that
            # goes below the average time * the multiplier
            try:
                # For each threshold value find the times where the signal goes
                # from below the threshold to above the threshold

                # find the smallest positive time between each threshold
                # passing to the next threshold. each sucsessive time cannot be
                # less than that of the previous times?

                a = np.argmax(ind_diffs < (mean_ind_diff * multiplier))
                attack_start_ind = threshold_inds[a]
                # Calculate the end threshold by thr same method except looking
                # above the average time * the multiplier
                best_end_thresh = ind_diffs > (mean_ind_diff * multiplier)
                if not best_end_thresh:
                    attack_end_ind = threshold_inds[-1]
                else:
                    attack_end_ind = threshold_inds[np.argmax(best_end_thresh)]
            except ValueError as err:
                raise ValueError("Attack estimation failed: {0}".format(err))
            print("START: {0}\nEND: {1}".format(attack_start_ind, attack_end_ind))
            # TODO: Refine position by searching for local min and max of these
            # values
            self.attack_start = self.AnalysedAudioFile.samps_to_secs(
                attack_start_ind)
            self.attack_end = self.AnalysedAudioFile.samps_to_secs(
                attack_end_ind)
            # Values are stored in the file with the following format:
            # attack_start attack_end
            attackfile.write("{0} {1}\n".format(self.attack_start,
                                                self.attack_end))

    def calc_log_attack_time(self):
        """
        Calculate the logarithm of the time duration between the time the
        signal starts to the time that the signal reaches it's stable part
        Described here:
        http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf
        """
        if not self.attack_start or not self.attack_end:
            raise ValueError("Attack times must be calculated before calling"
                             "the log attack time method")
        self.logattacktime = math.log10(self.attackend-self.attackstart)

    def get_attack_from_file(self):
        """Read the attack values from a previously generated file."""
        # TODO:
        print("Reading attack estimation file:\t\t",
              os.path.relpath(self.attackpath))
        with open(self.attackpath, 'r') as attackfile:
            for line in attackfile:
                # Split the values and convert to their correct types
                starttime, endtime = line.split()
                self.attack_start = float(starttime)
                self.attack_end = float(endtime)
                self.attack_size = self.attack_end - self.attack_start

    @staticmethod
    def scale_to_range(array, high=1.0, low=0.0):
        mins = np.min(array)
        maxs = np.max(array)
        rng = maxs - mins
        return high - (((high - low) * (maxs - array)) / rng)
