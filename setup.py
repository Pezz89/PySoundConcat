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
    name="pysound",
    version="0.1",
    author="Sam Perry",
    author_email="u1265119@unimail.hud.ac.uk",
    description=("A library for audio analysis and synthesis."),
    license="GPL",
    keywords="synthesis audio",
    url="https://github.com/Pezz89/pysound",
    packages=find_packages(),
    setup_requires=['numpy'],
    install_requires=read('requirements.txt'),
)
