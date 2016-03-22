Installation
============

Prerequesites
-------------

There are a few dependencies required to install concatenator:

1. Python 2.7.11 (It will probably run on earlier versions)
2. libsndfile - used for audio file IO
3. The HDF5 Library - used for large file storage

There are a number of ways to install python. The simplest is through homebrew
using the following command:

.. code:: bash

    brew install python

An alternative that allows greater flexibility is to use pyenv which allows for
easy switching between python versions and guarantees the exact version needed:

.. code:: bash

    brew install pyenv
    pyenv install 2.7.11
    pyenv global 2.7.11

libsndfile and HDF5 libraries can also be installed via homebrew:

.. code:: bash

    brew install libsndfile
    brew tap homebrew/science
    brew install hdf5

Python library and dependancies installation
--------------------------------------------

The python package and it's dependencies can then be easily installed by
running the ./install.sh script from the root director of the project. Note
that this will install the project in it's project folder. To check that the
project is working correctly, simply run run_tests.

.. code:: bash

    ./install.sh
    ./run_tests

Jupyter Notebook Examples
---------------------------------------

the Jupyter notebook application is required in order to run the interactive
examples. It is recommended that this is installed as part of the iPython
library using:

.. code:: bash

    pip install "ipython[all]"

Notebooks can then be viewed from the Examples folder of the project by
running:

.. code:: bash

    jupyter notebook

This will open a notebook session in the browser.
