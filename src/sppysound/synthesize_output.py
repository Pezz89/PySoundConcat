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
from database import AudioDatabase, Synthesizer

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
        help='source database directory'
    )
    parser.add_argument(
        'output',
        type=str,
        help='output database directory'
    )
    parser.add_argument(
        'target',
        type=str,
        help='target database directory',
        default=None
    )
    args = parser.parse_args()

    # Load database of samples to be used for output synthesis
    source_db = AudioDatabase(
        args.source,
        config=config,
        analysis_list={"f0", "rms"}
    )
    # Create/load a pre-existing database
    source_db.load_database(reanalyse=False)

    # Load database used to generate matches to source database.
    # This is used when enforcing analyses such as RMS and F0. (Original grains
    # are needed to calculate the ratio to alter the synthesized grain by)
    target_db = AudioDatabase(
        args.target,
        config=config,
        analysis_list={"f0", "rms"}
    )
    # Create/load a pre-existing database
    target_db.load_database(reanalyse=False)

    output_db = AudioDatabase(
        args.output,
        config=config
    )
    # Create/load a pre-existing database
    output_db.load_database(reanalyse=False)

    synthesizer = Synthesizer(source_db, output_db, target_db=target_db, config=config)
    synthesizer.synthesize(grain_size=100, overlap=4)

if __name__ == "__main__":
    main()
