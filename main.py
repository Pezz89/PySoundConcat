#!/usr/bin/python

from pysndfile import *
from wave_gen import gen_wave
import numpy as np
import argparse as ap
from os.path import (
    isfile,
    relpath
)
from zerox import find_zerox_pysndfile
from time_stretch import cheat_granulate_audio
from audio_graph import plot_audio
def gen_window(window_type, window_size):
    """
    Generates a window function of given size and type
    Returns a 1D numpy array
    """
    if window_type is "hanning":
        return np.hanning(window_size)
    elif window_type is "hamming":
        return np.hamming(window_size)
    elif window_type is "bartlett":
        return np.bartlett(window_size)
    elif window_type is "blackman":
        return np.blackman(window_size)
    elif window_type is "kaiser":
        return np.kaiser(window_size)
    else:
        raise ValueError

def audio_file_info(audio_file, file_path):
    """
    Prints audio information
    """
    print "*************************************************"
    print "File:                    ", relpath(file_path)  
    print "No. channels:            ", audio_file.channels()
    print "Samplerate:              ", audio_file.samplerate()
    print "Format:                  ", audio_file.format()
    print "No. Frames:              ", audio_file.frames()
    print "Encoding string:         ", audio_file.encoding_str()
    print "Major format string:     ", audio_file.major_format_str()
    print "Seekable?:               ", bool(audio_file.seekable())
    print "Errors?:                 ", audio_file.error()
    print "*************************************************"


def check_args(args):
    """
    Check arguments provided by user for validity
    """
    if not isfile(args.input_file):
        print "File doesn't exist."
        raise IOError
    if isfile(args.output_file):
        print "Overwriting file: ", args.output_file

def parse_arguments():
    """
    Parses arguments
    Returns a namespace with values for each argument 
    """
    parser = ap.ArgumentParser(description = "Prosess audio files")
    
    parser.add_argument(
        'input_file', 
        metavar='', 
        help = "Sound file to process"
    )
    
    parser.add_argument(
        'output_file', 
        metavar='', 
        help = "Sound file to output"
    )
    
    parser.add_argument(
        '--verbose', 
        '-v', 
        action='count'
    )
    
    main_args = parser.parse_args()
    return main_args


    
def main():
    #Parse arguments
    main_args = parse_arguments()
    check_args(main_args)

    #Create audio file instances
    input_audio = PySndfile(main_args.input_file, mode = 'r')
    output_audio = PySndfile(
        main_args.output_file, 
        mode = 'w', 
        format = input_audio.format(),
        channels = input_audio.channels(),
        samplerate = input_audio.samplerate()
    )

    if main_args.verbose > 1:
        audio_file_info(input_audio, main_args.input_file)
        audio_file_info(output_audio, main_args.output_file)
    
    cheat_granulate_audio(input_audio, output_audio)
    #print find_zerox_pysndfile(input_audio, 10000, search_before = False)
    
    #plot_audio(gen_wave(2, 5, "sine"))
if __name__ == "__main__":
    main()
