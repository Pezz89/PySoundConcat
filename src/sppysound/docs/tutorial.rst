Tutorial
========

This section gives a brief introduction to using the 'concatenator.py' script. The
script can be found in the src/sppysound directory of the project folder, or
can be accessed by running the 'concatenator' symbolic link from the project
folder root.

Getting Started
---------------

All operations are performed through use of the ./concatenator script. It is
designed to intuitively search the locations provided as arguments, create any
analyses that are needed (and do not already exist) automatically, and
match/synthesize results all through one interface.

This allows the user to simply supply three arguments:

- A source directory
- A target directory
- An output directory

The script will then search these for any analyses that have been created
previously, create any new analyses that haven't and generate results using the
default settings. 

An example command call might look something like this:

.. code:: bash

    ./concatenator /path/to/source_db /path/to/target_db /path/to/output_db

This will recursively search these folder and organize audio found so that it
can be used as part of the database. 

Note: If further audio need to be added after having run the script, simply add
it anywhere in the folder and it will be added to the database on the next run.

Detailed Usage
--------------

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

(Details on all available commands are available in the :ref:`usage`)

For this demonstration, the following file structure will be used:

::

    /Users/samuelperry/concatenator_test/
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

By using the ``--src_db`` and ``--tar_db`` flags, alternative locations are specified
for generating the databases and storing analysis data. Symbolic links are
created, referencing the original audio files without moving them.  This allows
large databases to be used in place without copying or moving it's content.

Alternatively, databases can be generated in place by omitting the ``--src_db`` and
``--tar_db`` flags. This will create the database directory structure directly in
the directories provided as source and target.

The ``--copy`` flag can be used in conjunction with these flags in order to create
actual copies of the audio files at the destinations. This allows for the
creation of portable databases that can moved to other machines without
breaking links to the original files. (Any pre-existing symbolic links will be
overwritten with hard copies when using this option.)

Parameter Configuration Flags
-----------------------------
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
the ``--reanalyse`` and ``--rematch`` flags can be used to force the overwriting of
old data, using the new parameters.

Analyses can also be selected manually using the ``--analyse`` flag. This
allow matching and synthesis to be made based on a specific subset of analyses.
For example:

.. code:: bash

    concatenator ./source_db ./target_db ./output_db --src_db \
    ./analysed_source_db --tar_db ./analysed_tar_db --analyse f0 rms

This will run the matching using only the RMS and F0 analyses. 


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
        # than making physical copies.
        "symlink": True
    }

    # Sets the weighting for each analysis. A higher weighting gives an analysis
    # higher precedence when finding the best matches.
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

.. _usage:

concatenate.py Script Flags
---------------------------
-h, --help            show help message and exit

--analyse, -a         [ANALYSES, ...] Specify analyses to be created. Valid analyses are:

                      - "rms"

                      - "zerox"

                      - "fft"

                      - "spccntr"

                      - "spcsprd"

                      - "spcflux"

                      - "spccf"

                      - "spcflatness"

                      - "f0"
                      
                      - "peak"

                      - "centroid"

                      - "variance"

                      - "kurtosis"

                      - "skewness"

                      - "harm_ratio"

--analysis_dict       Set the formatting of each analysis for grain matching. 

                      Example: '--f0 median --rms mean'

--fft                 Overwrite default config setting for fft analysis.

                      Example: '--window_size 2048'

--kurtosis            Overwrite default config setting for kurtosis
                      analysis. 
                      
                      Example: '--window_size 100 --overlap 2'

--matcher             Set matcher settings. 

                      Example: 'match_quantity'

--matcher_weightings  Set weighting for analysis to set their presedence when matching. 

                      Example: '--f0 2 --rms 1.5'

--rms                 Overwrite default config setting for rms analysis.

                      Example: '--window_size 100 --overlap 2'

--skewness            Overwrite default config setting for skewness
                      analysis. 
                      
                      Example: '--window_size 100 --overlap 2'

--synthesizer         Set synthesis settings. 

                      Example: '--enf_rms_ratio_limit 2'

--variance            Overwrite default config setting for variance
                      analysis. 
                      
                      Example: '--window_size 100 --overlap 2'

--reanalyse           Force re-analysis of all analyses, overwriting any
                      existing analyses

--rematch             Force re-matching, overwriting any existing match data

--enforcef0           This flag enables pitch shifting of matched grains to
                      better match the target.
                      
--enforcerms          This flag enables scaling of matched grains to better
                      match the target's volume.

--copy                This flag enables the copying of audio files from
                      their location to the database, rather than creating
                      symbolic links. This is useful for creating portable
                      databases.

--match_method        Choose the algorithm to use when matching analyses. Available algorithms are:

                         Brute force: 'bruteforce'

                         K-d Tree Search: 'kdtree'

--verbose, -v         Specifies level of verbosity in output. For example:
                      '-vvvvv' will output all information. '-v' will output
                      minimal information.

