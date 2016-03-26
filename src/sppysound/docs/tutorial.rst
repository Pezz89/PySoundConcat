Tutorial
========

This section gives a brief introduction to using the 'concatenator.py' script. The
script can be found in the src/sppysound directory of the project folder, or
can be accessed by running the 'concatenator' symbolic link from the project
folder root.

Getting Started
---------------

To view all available options simply run:

.. code:: bash

    ./concatenator -h

A list of all commands available is then presented:

::

    usage: concatenator [-h] [--src-db] [--tar_db]
                    [--analyse [ANALYSE [ANALYSE ...]]] [--analysis_dict]
                    [--fft] [--kurtosis] [--matcher] [--matcher_weightings]
                    [--rms] [--skewness] [--synthesizer] [--variance]
                    [--reanalyse] [--rematch] [--enforcef0] [--enforcerms]
                    [--copy] [--match_method] [--verbose]
                    source target output

    Concatenator is a tool for synthesizing interpretations of a sound, through
    the analysis and synthesis of audio grains from a corpus database. The program
    works by analysing overlapping segments of audio (known as grains) from both

    ...

For this demonstration, the following file structure will be used:

::
    /Users/samuelperry/concatenator_test/
    |-- source_db
    |   |-- Trumpet.novib.ff.A3.stereo.aif
    |   |-- Trumpet.novib.ff.A4.stereo.aif
    |   |-- Trumpet.novib.ff.A5.stereo.aif

    ...

    |   |-- Trumpet.novib.ff.F5.stereo.aif
    |   |-- Trumpet.novib.ff.G3.stereo.aif
    |   `-- Trumpet.novib.ff.G4.stereo.aif
    `-- target_db
        |-- target.01.wav
        |-- target.02.wav
        |-- target.03.wav
        `-- target.04.wav

A source database containing a small selection of trumpet samples (aquired from

http://theremin.music.uiowa.edu/MIS.html) will be used to match grains with 4
target sounds. This will produce 4 output files, one for each target sound.

The following command is used to to generate the output:

.. code:: bash

    concatenator ./source_db ./target_db ./output_db --src_db \
    ./analysed_source_db --tar_db ./analysed_tar_db

The specified directories are searched recursively for audio files that are
used as items in the database. These item are then matched and synthesized as
explained in the :ref:`overview` section. Output is stored in the audio
directory of the output database that has been created.
This produces this directory structure:

::
    
    /Users/samuelperry/concatenator_test/
    |-- analysed_source_db
    |   |-- audio
    |   |   |-- Trumpet.novib.ff.A3.stereo.aif -> (Symlink)
    |   |   |-- Trumpet.novib.ff.A4.stereo.aif -> (Symlink)

    ...

    |   |   |-- Trumpet.novib.ff.G3.stereo.aif -> (Symlink)
    |   |   `-- Trumpet.novib.ff.G4.stereo.aif -> (Symlink)
    |   `-- data
    |       `-- analysis_data.hdf5
    |-- analysed_tar_db
    |   |-- audio
    |   |   |-- target.01.wav -> (Symlink)
    |   |   |-- target.02.wav -> (Symlink)
    |   |   |-- target.03.wav -> (Symlink)
    |   |   `-- target.04.wav -> (Symlink)
    |   `-- data
    |       `-- analysis_data.hdf5
    |-- output_db
    |   |-- audio
    |   |   |-- target.01_output.wav
    |   |   |-- target.02_output.wav
    |   |   |-- target.03_output.wav
    |   |   `-- target.04_output.wav
    |   `-- data
    |       `-- analysis_data.hdf5
    |-- source_db
    |   |-- Trumpet.novib.ff.A3.stereo.aif
    |   |-- Trumpet.novib.ff.A4.stereo.aif

    ...

    |   |-- Trumpet.novib.ff.G3.stereo.aif
    |   `-- Trumpet.novib.ff.G4.stereo.aif
    `-- target_db
        |-- target.01.wav
        |-- target.02.wav
        |-- target.03.wav
        `-- target.04.wav

By using the --src_db and --tar_db flags, alternative locations are specified
for generating the databases and storing analysis data. Symbolic links are
created, referencing the original audio files without moving them.  This allows
large databases to be used in place without copying or moving it's content.

Alternatively, databases can be generated in place by ommiting the --src_db and
--tar_db flags. this will create the database directory structure directly in
the directories provided as source and target.

The --copy flag can be used in conjunction with these flags in order to create
actual copies of the audio files at the destinations. This allows for the
creation of partable databases that can moved to other machines without
breaking links to the original files. (Any pre-existing symbolic links will be
overwritten with hard copies when using this option.)

Configuration Flags
-------------------
For quick modification of analysis parameters, parameter flags can be specified
directly when calling the script. For example:

.. code:: bash

    concatenator ./source_db ./target_db ./output_db --src_db \
    ./analysed_source_db --tar_db ./analysed_tar_db --reanalyse --fft \
    '--window_size 2048'

This overwrites the value specified for window_size in the config file with the
value provided.

When databases have already been created, previous data is used when re-running
the script over them. This allows for different databases to be used without
continuous reanalysis. However, if analysis or matching parameters are changed,
the "--reanalyse" and "--rematch" flags can be used to force the overwriting of
old data, using the new parameters.


config.py
---------
The config.py file is used for specifying all user defined options and can be
edited in the concatenator project directory. Comments explain the function of
each parameter. The default config.py file looks like this:

.. code:: python

    # Specify analysis parameters for root mean square analysis.
    rms = {
        "window_size": 70,
        "overlap": 2,
    }

    # Specify analysis parameters for variance analysis.
    variance = {
        "window_size": 70,
        "overlap": 2
    }

    # Specify analysis parameters for temporal kurtosis analysis.
    kurtosis = {
        "window_size": 70,
        "overlap": 2
    }

    # Specify analysis parameters for temporal skewness analysis.
    skewness = {
        "window_size": 70,
        "overlap": 2
    }

    # Specify analysis parameters for FFT analysis.
    fft = {
        "window_size": 65536
    }

    database = {
        # Enables creation of symbolic links to files not in the database rather
        # than making pysical copies.
        "symlink": True
    }

    # Sets the weighting for each analysis. a higher weighting gives an analysis
    # higher presendence when finding the best matches.
    matcher_weightings = {
        "f0" : 1.,
        "spccntr" : 1.,
        "spcsprd" : 1.,
        "spcflux" : 1.,
        "spccf" : 1.,
        "spcflatness": 1.,
        "zerox" : 1.,
        "rms" : 1.,
        "peak": 1.,
        "centroid": 1.,
        "kurtosis": 1.,
        "skewness": 1.,
        "variance": 3.,
        "harm_ratio": 1.
    }

    # Specifies the method for averaging analysis frames to create a single value
    # for comparing to other grains. Possible formatters are: 'mean', 'median',
    # 'log2_mean', 'log2_median'
    analysis_dict = {
        "f0": "log2_median",
        "rms": "mean",
        "zerox": "mean",
        "spccntr": "mean",
        "spcsprd": "mean",
        "spcflux": "mean",
        "spccf": "mean",
        "spcflatness": "mean",
        "peak": "mean",
        "centroid": "mean",
        "kurtosis": "mean",
        "skewness": "mean",
        "variance": "mean",
        "harm_ratio": "mean"
    }

    analysis = {
        # Force the deletion of any pre-existing analyses to create new ones. This
        # is needed for overwriting old analyses generated with different
        # parameters to the current ones.
        "reanalyse": False
    }

    matcher = {
        # Force the re-matching of analyses
        "rematch": True,
        "grain_size": 70,
        "overlap": 2,
        # Defines the number of matches to keep for synthesis. Note that this must
        # also be specified in the synthesis config
        "match_quantity": 1,
        # Choose the algorithm used to perform matching. kdtree is recommended for
        # larger datasets.
        "method": 'kdtree'
    }

    synthesizer = {
        # Artificially scale the output grain by the difference in RMS values
        # between source and target.
        "enforce_rms": True,
        # Specify the ratio limit that is the grain can be scaled by.
        "enf_rms_ratio_limit": 100.,
        # Artificially modify the pitch by the difference in f0 values between
        # source and target.
        "enforce_f0": True,
        # Specify the ratio limit that is the grain can be modified by.
        "enf_f0_ratio_limit": 10.,
        "grain_size": 70,
        "overlap": 2,
        # Normalize output, avoid clipping of final output by scaling the final
        # frames.
        "normalize" : True,
        # Defines the number of potential grains to choose from matches when
        # synthesizing output.
        "match_quantity": 1
    }

    output_file = {
        "samplerate": 44100,
        "format": 131075,
        "channels": 1
    }

