f0 = {
    "threshold": 0.6
}

RMS = {
    "window_size": 150,
    "overlap": 8
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
}

matcher = {
    "rematch": True,
    "grain_size": 70,
    "overlap": 2
}

synthesizer = {
    "enforce_rms": True,
    "enf_rms_ratio_limit": 100.,
    "enforce_f0": True,
    "enf_f0_ratio_limit": 10.,
    "grain_size": 70,
    "overlap": 2
}

output_file = {
    "samplerate": 44100,
    "format": 131075,
    "channels": 1
}
