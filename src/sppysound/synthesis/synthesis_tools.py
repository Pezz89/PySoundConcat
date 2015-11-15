from __future__ import print_function, division
import numpy as np


def convolve(input, impulse_response):
    out = np.zeros(len(input) + len(impulse_response) - 1)
    for input_ind, i in enumerate(input):
        for imp_ind, j in enumerate(impulse_response):
            out[input_ind+imp_ind] = out[input_ind+imp_ind] + i*j
    return out

if __name__ == "__main__":
    a = np.array([1, 0.5, 3, 1])
    b = np.array([1, 0, 0, 0])
    c = convolve(a, b)
    print(c)
    print(np.convolve(a, b))
