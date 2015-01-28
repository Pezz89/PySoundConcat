import numpy as np
from math import (
    ceil
)
from audio_graph import plot_audio
import pysndfile as psf

def x_corr_time_lag(grain1, grain2):
    """
    Calculate the the time lag between two grains where grain2 is at maximum
    similarity to grain1
    """
    correlation = np.correlate(grain1, grain2, mode = "full")
    time_lag = np.argmax(correlation) - grain1.size
    return time_lag

def read_grain(audio, start_index, grain_size):
    """
    Read a grain of audio from an audio object. if grain ends after the end of
    the file, the grain is padded with zeros
    """
    #Read grain
    index = audio.seek(start_index, 0)
    if index + grain_size > audio.frames():
        grain = audio.read_frames(
            audio.frames() - index
        )
        grain = np.pad(
            grain, 
            (0, index+grain_size-audio.frames()), 
            'constant', 
            constant_values=(0,0)
        )

    else:
        grain = audio.read_frames(grain_size)
    return grain

def cheat_granulate_audio(
    input_audio, 
    output_audio, 
):
    grain_time_diff = 256
    grain_size = 2048
    stretch_factor = 1.9
    grain_reposition = round(grain_time_diff * stretch_factor)
    overlap_size = grain_time_diff * stretch_factor / 2.0
    total_num_segs = ceil(input_audio.frames() / grain_time_diff)
    input_audio.seek(0, 0)

    grain1 = read_grain(input_audio, 0, grain_size)
    
    #starts at 2nd grain and calculates between second and first,
    #then iterates through grains
    i = 1
    print "Segment No.:     ", total_num_segs
    while i < total_num_segs - 1:
        grain2 = read_grain(input_audio, i*grain_time_diff, grain_size)
        #Read overlap with next grain to calculate the X-correlation
        time_lag = x_corr_time_lag(
            grain2[0:overlap_size],
            grain1[grain_reposition:grain_reposition+overlap_size], 
        )
        print time_lag
        fadein = np.linspace(
            0.0, 
            1.0,
            grain1.size - (i*grain_reposition-overlap_size+time_lag) #
        )
        fadeout = np.linspace(
            1.0, 
            0.0,
            grain1.size - (i*grain_reposition-overlap_size+time_lag) #
        )
        tail = grain1[(i*grain_reposition-overlap_size)+time_lag:grain1.size]*fadeout
        begin = grain2[0:fadein.size]*fadein
        add = begin + tail
        print grain1[:i*grain_reposition-overlap_size+time_lag].size
        print add.size
        print grain2[fadein.size:grain_size].size
        grain1 = np.concatenate(
            (grain1[:i*grain_reposition-overlap_size+time_lag],
            add,
            grain2[fadein.size:grain_size])
        )
        if i == 200:
            break
        i += 1
    output_audio.write_frames(grain1)
    exit()
def granulate_audio(
    input_audio, 
    output_audio, 
    stretch=1.5,
    window_size=1000,
    offset = 500,
    overlap = 250
):
    """
    Time-stretches audio using SOLA granulation
    """
    input_grains = np.array([]) 
    offset_count = 0
    #Read audio into grains of set size with set offset
    while True:
        #Read as many full windows of audio as possible
        try:
            read_frames = input_audio.read_frames(window_size)
            offset_count += offset
            input_audio.seek(offset_count, 0)
            if not input_grains.size:
                input_grains = np.array([read_frames])
            else:
                input_grains = np.append(input_grains, [read_frames], axis = 0)
        except RuntimeError:
            break
    i = 0
    while i < input_grains.shape[0] - 1:
        print i
        i += 1
    print input_grains.shape

    
    #find the best overlap point fo the x-fade by calculating the cross
    #correlation
    
    time_shift = int(round(offset * stretch))
