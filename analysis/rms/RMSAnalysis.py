import os
import numpy as np

class RMSAnalysis:
    def __init__(self, AnalysedAudioFile, rmspath):
        # Store the path to the rms file if it already exists
        self.rmspath = rmspath

        # Stores the number of RMS window values when calculating the RMS
        # contour
        self.rms_window_count = None
        # Check if analysis file already exists.
        # If an RMS file is provided then count the number of lines (1 for each
        # window)
        if self.rmspath:
            with open(self.rmspath, 'r') as rmsfile:
                self.rms_window_count = sum(1 for line in rmsfile)
        # If it does then great...
        # If it doesn't then generate a new file
        else:
            self.rmspath = self.create_rms_analysis(AnalysedAudioFile)

    #-------------------------------------------------------------------------
    # RMS ESTIMATION METHODS
    def create_rms_analysis(self, AnalysedAudioFile, window_size=25, window_type='triangle', window_overlap=8):
        print AnalysedAudioFile
        """Generate an energy contour analysis by calculating the RMS values of windowed segments of the audio file"""
        window_size = AnalysedAudioFile.ms_to_samps(window_size)
        #Generate a window function to apply to rms windows before analysis
        window_function = AnalysedAudioFile.gen_window(window_type, window_size)
        # Check that the class file has a path to write the rms file to
        if not self.rmspath:
            # If it doesn't then attampt to generate a path based on the
            # location of the database that the object is a part of.
            if not AnalysedAudioFile.db_dir:
                # If it isn't part of a database and doesn't have a path then
                # there is no where to write the rms data to.
                raise IOError('Analysed Audio object must have an RMS file path or be part of a database')
            self.rmspath = os.path.join(AnalysedAudioFile.db_dir, 'rms', AnalysedAudioFile.name + '.lab')
        i = 0
        try:
            with open(self.rmspath, 'w') as rms_file:
                print 'Creating RMS file:\t\t\t', os.path.relpath(self.rmspath)
                self.rms_window_count = 0
                # For all frames in the file, read overlapping windows and
                # calculate the rms values for each window then write the data
                # to file
                while i < AnalysedAudioFile.frames():
                    frames = AnalysedAudioFile.read_grain(i, window_size)
                    frames = frames * window_function
                    rms = np.sqrt(np.mean(np.square(frames)))
                    rms_file.write('{0} {1:6f}\n'.format(i + int(round(window_size / 2.0)), rms))
                    i += int(round(window_size / window_overlap))
                    self.rms_window_count += 1

            return self.rmspath
        #If the rms file couldn't be opened then raise an error
        except IOError:
            return False

    def get_rms_from_file(self, start=0, end=-1):
        """
        Read values from RMS file between start and end points provided (in
        samples)
        """
        # Convert negative numbers to the end of the file offset by that value
        if end < 0:
            end = self.frames() + end
        # Create empty array with a size equal to the maximum possible RMS
        # values
        rms_array = np.empty((2, self.rms_window_count))
        # Open the RMS file
        with open(self.rmspath, 'r') as rmsfile:
            i = 0
            for line in rmsfile:
                # Split the values and convert to their correct types
                time, value = line.split()
                time = int(time)
                value = float(value)
                # If the values are within the desired range, add them to the
                # array
                if time >= start and time <= end:
                    # The first value will be rounded down to the start
                    if i == 0:
                        time = start
                    rms_array[0][i] = time
                    rms_array[1][i] = value
                    i += 1
            # The last value will be rounded up to the end
            rms_array[0][i] = end
        rms_array = rms_array[:, start:start+i]
        # Interpolate between window values to get per-sample values
        rms_contour = np.interp(np.arange(end), rms_array[0], rms_array[1])
        return rms_contour
