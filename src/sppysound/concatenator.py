#!/usr/bin/env python

import argparse
import audiofile
import logging
from fileops import loggerops
import pdb
import os
import sys
from database import AudioDatabase, Matcher, Synthesizer
import config

modpath = sys.argv[0]
modpath = os.path.splitext(modpath)[0]+'.log'


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
        help='Directory of source files/database to take grains from '
        'when synthesizing output'
    )

    parser.add_argument(
        'target',
        type=str,
        help='Directory of target files/database to match source grains to.'
    )

    parser.add_argument(
        'output',
        type=str,
        help='Directory to use as database for outputing results and match '
        'information.\nOutput audio will be stored in the /audio sub-directory '
        'and match data will be stored in the /data directory.'
    )

    parser.add_argument(
        '--analyse',
        '-a',
        nargs='*',
        help='Specify analyses to be created. Valid analyses are: \'rms\''
        '\'f0\' \'zerox\' \'fft\' etc... (see the documentation for full '
        'details on available analyses)',
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
        "--reanalyse",
        action="store_true",
        help="Force re-analysis of all analyses, overwriting any existing "
        "analyses"
    )

    parser.add_argument(
        "--rematch",
        action="store_true",
        help="Force re-matching, overwriting any existing match data "
    )

    parser.add_argument(
        "--enforcef0",
        action="store_true",
        help="This flag enables pitch shifting of matched grainsto better match the target."
    )

    parser.add_argument(
        "--enforcerms",
        action="store_true",
        help="This flag enables scaling of matched grains to better match the target's volume."
    )

    parser.add_argument('--verbose', '-v', action='count')

    args = parser.parse_args()

    if not args.verbose:
        args.verbose = 50
    else:
        levels = [50, 40, 30, 20, 10]
        if args.verbose > 5:
            args.verbose = 5
        args.verbose -= 1
        args.verbose = levels[args.verbose]

    return args


def main():
    # Process commandline arguments
    args = parse_arguments()

    logger = loggerops.create_logger(
        logger_streamlevel=args.verbose,
        log_filename=modpath,
        logger_filelevel=args.verbose
    )
    if not args.verbose:
        logger.disables = True

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
