#!/usr/bin/env python

"""Command line interface for generating an analysed audio file database."""

import argparse
import audiofile


def main():
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
