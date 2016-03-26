API
===
---------------------------
concatenate.py Script Usage
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

--enforcef0           This flag enables pitch shifting of matched grainsto
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

-------------------
AudioFile Class
-------------------
.. autoclass:: audiofile.AudioFile
   :members:

---------------------------
AnalysedAudioFile Class
---------------------------
.. autoclass:: audiofile.AnalysedAudioFile
   :members:

-------------------
Database Class
-------------------
.. autoclass:: database.AudioDatabase
   :members:

-------------------
Matcher Class
-------------------
.. autoclass:: database.Matcher
   :members:

---------------------------
Synthesizer Class
---------------------------
.. autoclass:: database.Synthesizer
   :members:

---------------------------
Analysis Classes
---------------------------
.. autoclass:: analysis.Analysis
   :members:
Centroid Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: analysis.CentroidAnalysis
   :members:

F0 Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: analysis.F0Analysis
   :members:

FFT Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: analysis.FFTAnalysis
   :members:

Harmonic Ratio Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: analysis.F0HarmRatioAnalysis
   :members:

Kurtosis Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: analysis.KurtosisAnalysis
   :members:

Peak Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: analysis.PeakAnalysis
   :members:

RMS Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: analysis.RMSAnalysis
   :members:

Spectral Centroid Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: analysis.SpectralCentroidAnalysis
   :members:

Spectral Crest Factor Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: analysis.SpectralCrestFactorAnalysis
   :members:

Spectral Flatness Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: analysis.SpectralFlatnessAnalysis
   :members:

Spectral Flux Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: analysis.SpectralFluxAnalysis
   :members:

Spectral Spread Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: analysis.SpectralSpreadAnalysis
   :members:

Variance Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: analysis.VarianceAnalysis
   :members:

Zero-Crossing Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: analysis.ZeroXAnalysis
   :members:

