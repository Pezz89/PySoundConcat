#!/usr/bin/env python

"""Command line interface for generating an analysed audio file database."""

import argparse
import audiofile
import logging
from fileops import loggerops
import pdb
import os
from database import AudioDatabase
import config
import __builtin__

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
        '--analyse',
        '-a',
        nargs='*',
        help='Specify analyses to be created. Valid analyses are: \'rms\''
        '\'f0\' \'atk\' \'fft\'',
        default=["rms", "zerox", "fft", "spccntr", "spcsprd", "spcflux", "spccf", "spcflatness", "f0", "peak", "centroid"]
    )
    parser.add_argument(
        '--rms',
        nargs='+',
        help='Specify arguments for creating RMS analyses'
    )
    parser.add_argument(
        '--atk',
        nargs='+',
        help='Specify arguments for creating attack analyses'
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
    args = parser.parse_args()

    # Create database object
    database = AudioDatabase(
        args.source,
        args.target,
        analysis_list=args.analyse,
        config=config
    )
    # Create/load a pre-existing database
    database.load_database(reanalyse=args.reanalyse)

if __name__ == "__main__":
    main()
