import numpy as np

todeg = 180 / np.pi

def lin2db(x, amin=1e-30):
    return 20.0 * np.log10(np.maximum(x, amin))

def db2lin(x):
    return 10**(x / 20)