f0 = {
    "threshold": 0.6
}

analyser = {
    "fft_size": 8192
}

matcher_weightings = {
        "f0" : 1.,
        "spccntr" : 1.,
        "spcsprd" : 1.,
        "spcflux" : 1.,
        "zerox" : 1.,
        "rms" : 1.,
    }

matcher = {
    "rematch": True,
}

synthesizer = {
    "enforce_rms": True,
    "enf_rms_ratio_limit": 100.,
    "enforce_f0": True,
    "enf_f0_ratio_limit": 100.,
}

output_file = {
    "samplerate": 44100,
    "format": 131075,
    "channels": 1
}
