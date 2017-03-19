import numpy as np
import pdb
import subprocess

from audiofile import AudioFile

def shift(sigin, pitch):
    if np.isnan(pitch):
        return sigin
    input_filepath = "./.shift_input.wav"
    output_filepath = "./.shift_output.wav"

    shift_input = AudioFile.gen_default_wav(
        input_filepath,
        overwrite_existing=True,
        mode='w',
        channels=1,
    )
    # Write grain to be shifted to file
    shift_input.write_frames(sigin)
    # Close file
    del shift_input

    cents = 1200. * np.log2(pitch)
    p_shift_args = ["sox", input_filepath, output_filepath, "pitch", str(cents)]

    p = subprocess.Popen(p_shift_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    with AudioFile(output_filepath, mode='r') as shift_output:
        # Read result
        result = shift_output.read_grain()
    return result

def shift2():
    print "To implement..."
