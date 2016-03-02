f0 = {
    "threshold": 0.3
}


matcher_weightings = {
        "f0" : 1.,
        "spccntr" : 1.,
        "spcsprd" : 1.,
        "zerox" : 1.,
        "rms" : 1.,
    }

matcher = {
    "rematch": True,
}

synthesizer = {
    "enforce_rms": True,
    "enf_rms_ratio_limit": 10.,
    "enforce_f0": True,
}

output_file = {
    "samplerate": 44100,
    "format": 131075,
    "channels": 1
}
