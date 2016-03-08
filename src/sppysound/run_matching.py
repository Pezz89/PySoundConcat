#!/usr/bin/env python

"""Command line interface for matching databases"""

import argparse
import audiofile
import logging
from fileops import loggerops
import pdb
import os
import __builtin__
import config
from database import AudioDatabase, Matcher
pdb.pm
filename = os.path.splitext(__file__)[0]
logger = loggerops.create_logger(log_filename='./{0}.log'.format(filename))

###########################################################################
# File open and closing monitoring
openfiles = set()
oldfile = __builtin__.file

class newfile(oldfile):
    def __init__(self, *args):
        self.x = args[0]
        logger.debug("OPENING %s" % str(self.x))
        oldfile.__init__(self, *args)
        openfiles.add(self)

    def close(self):
        logger.debug("CLOSING %s" % str(self.x))
        oldfile.close(self)
        openfiles.remove(self)
oldopen = __builtin__.open
def newopen(*args):
    return newfile(*args)
__builtin__.file = newfile
__builtin__.open = newopen

def printOpenFiles():
    logger.debug("%d OPEN FILES: [%s]" % (len(openfiles), ", ".join(f.x for f in openfiles)))

###########################################################################

def main():
    """Parse arguments then generate database."""
    logger.info('Started')
    parser = argparse.ArgumentParser(
        description='Generate a database at argument 1 based on files in '
        'argument 2.'
    )
    parser.add_argument(
        'source',
        type=str,
        help='Source database directory'
    )
    parser.add_argument(
        'target',
        type=str,
        help='Target database directory'
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
        help='Specify analyses to be used. Valid analyses are: \'rms\''
        '\'f0\' \'fft\'',
        default=["rms", "zerox", "fft", "spccntr", "spcsprd", "spcflux", "spccf", "spcflatness", "f0", "peak", "centroid", "variance"]
    )
    parser.add_argument(
        "--rematch", action="store_true",
        help="Force re-matching, overwriting any existing match data "
    )
    args = parser.parse_args()
    source_db = AudioDatabase(
        args.source,
        analysis_list=args.analyse,
        config=config
    )
    # Create/load a pre-existing database
    source_db.load_database(reanalyse=False)

    target_db = AudioDatabase(
        args.target,
        analysis_list=args.analyse,
        config=config
    )
    # Create/load a pre-existing database
    target_db.load_database(reanalyse=False)

    output_db = AudioDatabase(
        args.output,
        config=config
    )
    # Create/load a pre-existing database
    output_db.load_database(reanalyse=False)

    analysis_dict = {
        "f0": "log2_median",
        "rms": "mean",
        "zerox": "mean",
        "spccntr": "mean",
        "spcsprd": "mean",
        "spcflux": "mean",
        "spccf": "mean",
        "spcflatness": "mean",
        "peak": "mean",
        "centroid": "mean",
        "variance": "mean"

    }

    matcher = Matcher(source_db, target_db, analysis_dict, output_db=output_db, config=config, quantity=1, rematch=args.rematch)
    matcher.match(matcher.brute_force_matcher, grain_size=config.matcher["grain_size"], overlap=config.matcher["overlap"])

if __name__ == "__main__":
    main()
