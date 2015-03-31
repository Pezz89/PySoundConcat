#!/usr/bin/python

import numpy as np
import argparse as ap
import os
import audiofile as af

def check_args(args):
    """
    Check arguments provided by user for validity
    """
    if not os.path.isfile(args.input_file):
        raise IOError("File doesn't exist.", args.input_file)
    if os.path.isfile(args.output_file):
        print "Overwriting file: ", args.output_file

def parse_arguments():
    """
    Parses arguments
    Returns a namespace with values for each argument 
    """
    parser = ap.ArgumentParser(description = "Prosess audio files")
    
    parser.add_argument(
        'database', 
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
    #check_args(main_args)
    
    #----------
    #Begin processing audio files here:
    #----------
    #Creates a database structure at the location given and returns a list of
    #AnalysedAudioObjects for each item in the database
    database = af.AudioDatabase(main_args.database)
    database.generate_analyses()
    """
    #Create audio file instances
    input_audio = af.AnalysedAudioFile(main_args.input_file, mode = 'r')
    output_audio = af.AnalysedAudioFile(
        main_args.output_file, 
        mode = 'w', 
        format = input_audio.format(),
        channels = input_audio.channels(),
        samplerate = input_audio.samplerate()
    )

    if main_args.verbose > 1:
        input_audio.audio_file_info()
        output_audio.audio_file_info()
    """    
if __name__ == "__main__":
    main()
