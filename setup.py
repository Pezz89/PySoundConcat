import os
from setuptools import setup, find_packages


def read(fname):
    """
    Utility function to read the README file.

    Used for the long_description.  It's nice, because now 1) we have a top level
    README file and 2) it's easier to type in the README file than to put a raw
    string in below ...
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="sppysound",
    version="1.0",
    author="Sam Perry",
    author_email="u1265119@unimail.hud.ac.uk",
    description=("A library for audio analysis and synthesis."),
    license="GPL",
    keywords="synthesis audio",
    url="https://github.com/Pezz89/pysound",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    setup_requires=["numpy"],  # Just numpy here
    install_requires=read('requirements.txt')
)
