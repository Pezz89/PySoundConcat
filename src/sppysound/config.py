f0 = {
    "threshold": 0.6
}
RMS = {
    "window_size": 60,
    "overlap": 4
}
variance = {
    "window_size": 60,
    "overlap": 4
}
kurtosis = {
    "window_size": 60,
    "overlap": 4
}
skewness = {
    "window_size": 60,
    "overlap": 4
}

analyser = {
    "fft_size": 65536
}

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

matcher = {
    "rematch": True,
    "grain_size": 60,
    "overlap": 4
}

synthesizer = {
    "enforce_rms": True,
    "enf_rms_ratio_limit": 10.,
    "enforce_f0": True,
    "enf_f0_ratio_limit": 10.,
    "grain_size": 60,
    "overlap": 4
}

output_file = {
    "samplerate": 44100,
    "format": 131075,
    "channels": 1
}
