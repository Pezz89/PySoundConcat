rms = {
    "window_size": 100,
    "overlap": 2,
}

analysis_dict = {
    "f0": "log2_median",
    "rms": "mean"
}

matcher_weightings = {
    "f0" : 1.,
    "rms": 1.
}

analysis = {
    "reanalyse": False
}

matcher = {
    "rematch": False,
    "grain_size": 100,
    "overlap": 2,
    # Defines the number of matches to keep for synthesis.
    "match_quantity": 20
}

output_file = {
    "samplerate": 44100,
    "format": 131075,
    "channels": 1
}

database = {
    "symlink": True
}
