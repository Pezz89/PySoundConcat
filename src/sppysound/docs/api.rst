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
                      
--verbose, -v         Specify the verbosity of the script's output. Additional
                      v will produce greater levels of detail ie. -vvvvv will 
                      produce all messages.

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

----------

.. autoclass:: analysis.F0Analysis
   :members:

----------

.. autoclass:: analysis.RMSAnalysis
   :members:

----------

.. autoclass:: analysis.ZeroXAnalysis
   :members:

----------

.. autoclass:: analysis.FFTAnalysis
   :members:

----------

.. autoclass:: analysis.PeakAnalysis
   :members:
