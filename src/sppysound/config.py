# Specify analysis parameters for root mean square analysis.
rms = {
    "window_size": 100,
    "overlap": 8,
}

f0 = {
    "window_size": 4096,
    "overlap": 8,
    "ratio_threshold": 0.7
}

# Specify analysis parameters for variance analysis.
variance = {
    "window_size": 100,
    "overlap": 8
}

# Specify analysis parameters for temporal kurtosis analysis.
kurtosis = {
    "window_size": 100,
    "overlap": 8
}

# Specify analysis parameters for temporal skewness analysis.
skewness = {
    "window_size": 100,
    "overlap": 8
}

# Specify analysis parameters for FFT analysis.
fft = {
    "window_size": 4096
}

database = {
    # Enables creation of symbolic links to files not in the database rather
    # than making pysical copies.
    "symlink": True
}

# Sets the weighting for each analysis. a higher weighting gives an analysis
# higher presendence when finding the best matches.
matcher_weightings = {
    "f0" : 1,
    "spccntr" : 1.,
    "spcsprd" : 1.,
    "spcflux" : 1.,
    "spccf" : 1.,
    "spcflatness": 1.,
    "zerox" : 1.,
    "rms" : 1,
    "peak": 1.,
    "centroid": 1.,
    "kurtosis": 1.,
    "skewness": 1.,
    "variance": 1.,
    "harm_ratio": 1
}

# Specifies the method for averaging analysis frames to create a single value
# for comparing to other grains. Possible formatters are: 'mean', 'median',
# 'log2_mean', 'log2_median'
analysis_dict = {
    "f0": "log2_median",
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
    "grain_size": 100,
    "overlap": 8,
    # Defines the number of matches to keep for synthesis. Note that this must
    # also be specified in the synthesis config
    "match_quantity": 5,
    # Choose the algorithm used to perform matching. kdtree is recommended for
    # larger datasets.
    "method": 'kdtree'
}

synthesizer = {
    # Artificially scale the output grain by the difference in RMS values
    # between source and target.
    "enforce_intensity": True,
    # Specify the ratio limit that is the grain can be scaled by.
    "enf_intensity_ratio_limit": 1000.,
    # Artificially modify the pitch by the difference in f0 values between
    # source and target.
    "enforce_f0": True,
    # Specify the ratio limit that is the grain can be modified by.
    "enf_f0_ratio_limit": 3.,
    "grain_size": 100,
    "overlap": 8,
    # Normalize output, avoid clipping of final output by scaling the final
    # frames.
    "normalize" : True,
    # Defines the number of potential grains to choose from matches when
    # synthesizing output.
    "match_quantity": 5
}

output_file = {
    "samplerate": 44100,
    "format": 131075,
    "channels": 1
}
