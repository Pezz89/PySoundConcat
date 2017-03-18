#!/usr/bin/env python

import subprocess
import os
import pdb
import fnmatch

import sys

def main():
    p = subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    out = out.strip('\n')
    track_filepath = os.path.join(out, ".gittrack")

    p = subprocess.Popen(["git", "ls-files", out, "--exclude-standard", "--others"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    out = out.splitlines()

    try:
        with open(track_filepath) as f:
            content = f.read().splitlines()
    except IOError:
        return 0


    untracked = []
    for filepath in out:
        for name in content:
            if fnmatch.fnmatch(filepath, name):
                untracked.append(filepath)

    if untracked:
        print "The following files are not tracked: "
        for i in untracked:
            print i
        print "Please either stage these files for the commit or add them to the project's .gitignore to disregard them."
        return 1
    else:
        return 0



if __name__ == "__main__":
    exit(main())
