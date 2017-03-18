# Specify analysis parameters for root mean square analysis.
rms = {
    # Analysis window sizes can be changed for each analysis individually.
    # These do not need to match the grain size of the matcher or synthesis.
    "window_size": 1024,
    "overlap": 4,
}

f0 = {
    "window_size": 2048,
    "overlap": 2,
    # Currently all frames below this ratio are digaurded and left as silence.
    # Different databases will require different values for the best results.
    # Noisier databases will need lower values than more tonal databases.
    "ratio_threshold": 0.81
}

# Specify analysis parameters for variance analysis.
variance = {
    "window_size": 1024,
    "overlap": 4
}

# Specify analysis parameters for temporal kurtosis analysis.
kurtosis = {
    "window_size": 1024,
    "overlap": 4
}

# Specify analysis parameters for temporal skewness analysis.
skewness = {
    "window_size": 1024,
    "overlap": 4
}

# Specify analysis parameters for FFT analysis.
fft = {
    # The FFT window size determines the window size for all spectral analyses.
    "window_size": 2048
}

database = {
    # Enables creation of symbolic links to files not in the database rather
    # than making pysical copies.
    "symlink": True
}

# Sets the weighting for each analysis. a higher weighting gives an analysis
# higher presendence when finding the best matches.
matcher_weightings = {
    "f0" : 2,
    "spccntr" : 1.,
    "spcsprd" : 1.,
    "spcflux" : 3.,
    "spccf" : 3.,
    "spcflatness": 1.,
    "zerox" : 1.,
    "rms" : 3,
    "peak": 0.1,
    "centroid": 0.5,
    "kurtosis": 2.,
    "skewness": 2.,
    "variance": 0.,
    "harm_ratio": 2
}

# Specifies the method for averaging analysis frames to create a single value
# for comparing to other grains. Possible formatters are: 'mean', 'median',
# 'log2_mean', 'log2_median'
analysis_dict = {
    # log2_median formats using mel scale. This is useful for analyses such as
    # F0.
    "f0": "median",
    "rms": "mean",
    "zerox": "mean",
    "spccntr": "median",
    "spcsprd": "median",
    "spcflux": "median",
    "spccf": "median",
    "spcflatness": "median",
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
    "rematch": False,
    # This value must be the same as the synthesis grain size to avoid the
    # speeding up or slowing down of the resulting file in relation to the
    # original.
    "grain_size": 1024,
    "overlap": 2,
    # Defines the number of matches to keep for synthesis. Note that this must
    # also be specified in the synthesis config
    "match_quantity": 2,
    # Choose the algorithm used to perform matching. kdtree is recommended for
    # larger datasets.
    "method": 'kdtree'
}

synthesizer = {
    # Artificially scale the output grain by the difference in RMS values
    # between source and target.
    "enforce_intensity": True,
    # Specify the ratio limit that is the grain can be scaled by.
    "enf_intensity_ratio_limit": 30.,
    # Artificially modify the pitch by the difference in f0 values between
    # source and target.
    "enforce_f0": False,
    # Specify the ratio limit that is the grain can be modified by.
    "enf_f0_ratio_limit": 1.,
    "grain_size": 1024,
    "overlap": 2,
    # Normalize output, avoid clipping of final output by scaling the final
    # frames.
    "normalize" : False,
    # Defines the number of potential grains to choose from matches when
    # synthesizing output.
    "match_quantity": 1
}

# Specifies the format for the output file. Changing this has not been tested
# so may produce errors/undesirable results.
output_file = {
    "samplerate": 44100,
    "format": 131075,
    "channels": 1
}
