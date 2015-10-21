#!/usr/bin/env python

"""Command line interface for generating an analysed audio file database."""

import argparse
import audiofile


def main():
    import __builtin__
    openfiles = set()
    oldfile = __builtin__.file
    class newfile(oldfile):
        def __init__(self, *args):
            self.x = args[0]
            print "### OPENING %s ###" % str(self.x)
            oldfile.__init__(self, *args)
            openfiles.add(self)

        def close(self):
            print "### CLOSING %s ###" % str(self.x)
            oldfile.close(self)
            openfiles.remove(self)
    oldopen = __builtin__.open
    def newopen(*args):
        return newfile(*args)
    __builtin__.file = newfile
    __builtin__.open = newopen

    def printOpenFiles():
        print "### %d OPEN FILES: [%s]" % (len(openfiles), ", ".join(f.x for f in openfiles))

    """Parse arguments then generate database."""
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
        help='Directory to generate the database in. If the directory does not'
        ' exist then it will be created if possible'
    )
    parser.add_argument(
        '--analyse',
        '-a',
        nargs='*',
        help='Specify analyses to be created. Valid analyses are: \'rms\''
        '\'f0\' \'atk\'',
        default=["rms", "zerox", "atk"]
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
    args = parser.parse_args()

    audiofile.AudioDatabase(
        args.source,
        args.target,
        analysis_list=args.analyse
    )

if __name__ == "__main__":
    main()
