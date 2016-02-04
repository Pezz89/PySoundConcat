#!/usr/bin/env python

"""Command line interface for matching databases"""

import argparse
import audiofile
import logging
from fileops import loggerops
import pdb
import os
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
        help='Source database directory'
    )
    parser.add_argument(
        'target',
        type=str,
        help='Target database directory'
    )
    parser.add_argument(
        '--analyse',
        '-a',
        nargs='*',
        help='Specify analyses to be created. Valid analyses are: \'rms\''
        '\'f0\' \'atk\' \'fft\'',
        default=["rms", "zerox", "fft", "spccntr", "spcsprd", "f0"]
    )
    args = parser.parse_args()
    src_database = audiofile.AudioDatabase(
        args.source,
        analysis_list=args.analyse,
    )
    # Create/load a pre-existing database
    src_database.load_database(reanalyse=False)

    tar_database = audiofile.AudioDatabase(
        args.target,
        analysis_list=args.analyse,
    )
    # Create/load a pre-existing database
    tar_database.load_database(reanalyse=False)

if __name__ == "__main__":
    main()
