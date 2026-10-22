import numpy as np

todeg = 180 / np.pi

def lin2db(x, amin=1e-30):
    return 20.0 * np.log10(np.maximum(np.abs(x), amin))

def db2lin(x):
    return 10**(x / 20)

def pow2db(x, amin=1e-30):
    return 10.0 * np.log10(np.maximum(np.abs(x), amin))

def db2pow(x):
    return 10**(x / 10)