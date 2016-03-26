Installation
============

Prerequesites
-------------

Tested working on:

- System: MacBook Air (11-inch, Mid 2011)

- Operating System: OSX 10.11.4 or Ubuntu 14.04

- Processor: 1.6 GHz Intel Core i5

- Memory: 4 GB 1333 MHz DDR3

There are a few dependencies required to install concatenator:

1. Python 2.7.11 (tested)
2. libsndfile - used for audio file IO
3. The HDF5 Library - used for large file storage
4. The Sox audio library - used for pitch shifting

Getting this package
++++++++++++++++++++

This package can be downloaded with git using the following command:

.. code:: bash
    
    git clone https://github.com/Pezz89/pysound

This will clone the project folder into the current directory.

Brew Python
+++++++++++
There are a number of ways to install python. The simplest is through homebrew
(OSX) / Linuxbrew (most Linux distributions) using the following command:

.. code:: bash

    brew install python

Homebrew can be installed by following installation instructions from here:

http://brew.sh/

Linuxbrew can be installed by following installation instructions from here:

https://github.com/Linuxbrew/linuxbrew

Pyenv Python
++++++++++++
An alternative that allows greater flexibility is to use pyenv which allows for
easy switching between python versions and guarantees the exact version needed:

.. code:: bash

    brew install pyenv
    pyenv install 2.7.11
    pyenv global 2.7.11

Note that the following may need to be added to your ~/.bashrc file to add
pyenv pythons to your path.

.. code:: bash

    if which pyenv > /dev/null; then eval "$(pyenv init -)"; fi
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"

Other dependencies
++++++++++++++++++

libsndfile, sox and the HDF5 libraries can also be installed via homebrew/linuxbrew:

.. code:: bash

    brew install libsndfile
    brew tap homebrew/science
    brew install hdf5
    brew install sox

Python library and dependencies installation
--------------------------------------------

The python package and it's dependencies can then be easily installed by
running the ./install.sh script from the root director of the project. Note
that this will install the project in it's project folder. To check that the
project is working correctly, simply run run_tests.

.. code:: bash

    ./install.sh
    ./run_tests

There is a small chance that the installation may fail when installing
dependencies such as scipy or numpy. In these cases the packages must be
installed manually. When this has been done, simply re-run the install.sh
script.

Jupyter Notebook Examples
---------------------------------------

The Jupyter notebook application is required in order to run the interactive
examples. It is recommended that this is installed as part of the iPython
library using:

.. code:: bash

    pip install "ipython[all]"

Notebooks can then be viewed from the Examples folder of the project by
running:

.. code:: bash

    jupyter notebook

This will open a notebook session in the browser.
