#!/usr/bin/python

import argparse
import audiofile
import logging
from fileops import loggerops
import pdb
import os
from database import AudioDatabase, Matcher, Synthesizer
import config

def parse_arguments():
    """
    Parses arguments
    Returns a namespace with values for each argument
    """
    parser = argparse.ArgumentParser(
        description='Generate a database at argument 1 based on files in '
        'argument 2.'
    )

    parser.add_argument(
        'source',
        type=str,
        help='Directory of audio files to be added to the database'
    )

    parser.add_argument(
        'target',
        type=str,
        nargs='?',
        default='',
        help='Directory to generate the database in. If the directory does not'
        ' exist then it will be created if possible'
    )

    parser.add_argument(
        'output',
        type=str,
        help='output database directory'
    )

    parser.add_argument(
        '--analyse',
        '-a',
        nargs='*',
        help='Specify analyses to be created. Valid analyses are: \'rms\''
        '\'f0\' \'atk\' \'fft\'',
        default=[
            "rms",
            "zerox",
            "fft",
            "spccntr",
            "spcsprd",
            "spcflux",
            "spccf",
            "spcflatness",
            "f0",
            "peak",
            "centroid",
            "variance",
            "kurtosis",
            "skewness"
        ]
    )

    parser.add_argument(
        '--rms',
        nargs='+',
        help='Specify arguments for creating RMS analyses'
    )

    parser.add_argument(
        '--zerox',
        nargs='+',
        help='Specify arguments for creating zero-crossing analyses'
    )

    parser.add_argument(
        '--fft',
        nargs='+',
        help='Specify arguments for creating zero-crossing analyses'
    )

    parser.add_argument(
        "--reanalyse", action="store_true",
        help="Force re-analysis of all analyses, overwriting any existing "
        "analyses"
    )

    parser.add_argument(
        "--rematch", action="store_true",
        help="Force re-matching, overwriting any existing match data "
    )

    args = parser.parse_args()

    return args


def main():
    # Process commandline arguments
    main_args = parse_arguments()

    # Create/load a pre-existing source database
    source_db = AudioDatabase(
        args.source,
        analysis_list=args.analyse,
        config=config
    )
    source_db.load_database(reanalyse=False)

    # Create/load a pre-existing target database
    target_db = AudioDatabase(
        args.target,
        analysis_list=args.analyse,
        config=config
    )
    target_db.load_database(reanalyse=False)

    # Create/load a pre-existing output database
    output_db = AudioDatabase(
        args.output,
        config=config
    )
    output_db.load_database(reanalyse=False)

    # Initialise a matching object used for matching the source and target
    # databases.
    matcher = Matcher(
        source_db,
        target_db,
        config.analysis_dict,
        output_db=output_db,
        config=config,
        quantity=,
        rematch=args.rematch
    )

    # Perform matching on databases using the method specified.
    matcher.match(
        matcher.brute_force_matcher,
        grain_size=config.matcher["grain_size"],
        overlap=config.matcher["overlap"]
    )

    # Initialise a synthesizer object, used for synthesis of the matches.
    synthesizer = Synthesizer(
        source_db,
        output_db,
        target_db=target_db,
        config=config
    )

    # Perform synthesis.
    synthesizer.synthesize(
        grain_size=config.synthesizer["grain_size"],
        overlap=config.synthesizer["overlap"]
    )

if __name__ == "__main__":
    main()
