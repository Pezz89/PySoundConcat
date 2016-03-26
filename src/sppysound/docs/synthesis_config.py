rms = {
    "window_size": 100,
    "overlap": 2,
}

analysis_dict = {
    "f0": "log2_median",
    "rms": "mean"
}

analysis = {
    "reanalyse": False
}

output_file = {
    "samplerate": 44100,
    "format": 131075,
    "channels": 1
}

synthesizer = {
    "enforce_rms": True,
    "enf_rms_ratio_limit": 5.,
    "enforce_f0": True,
    "enf_f0_ratio_limit": 10.,
    "grain_size": 100,
    "overlap": 2,
    "normalize" : True,
    # Defines the number of potential grains to choose from matches when
    # synthesizing output.
    "match_quantity": 20
}

database = {
    "symlink": True
}
