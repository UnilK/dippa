import numpy as np
import json

class ArrayState:
    positions = np.array([[0, 0, 0]])
    calibration = np.array([1])
    def __init__(self, pos=np.array([[0, 0, 0]])):
        self.positions = pos
        

def load_array(path, calibrationPath=None):

    array = ArrayState()
    with open(path, "r", encoding="utf-8") as f:
        array.__dict__ = json.load(f)

    m = len(array.positions)
    array.positions = np.stack((*np.array(array.positions).T, np.zeros(m)))
    
    if calibrationPath != None:
        with open(calibrationPath, "r", encoding="utf-8") as f:
            array.calibration = np.array(json.load(f))
    else:
        array.calibration = np.ones(m)

    array.positions[0] -= np.mean(array.positions[0])
    array.positions[1] -= np.mean(array.positions[1])
    
    return array

def array_circle8():
    return load_array("circle_array8.json", "circle_array8_calibration.json")

def array_circle16():
    
    m = 16
    a = np.exp(2j*np.pi*np.arange(m)/m) * 0.1
    
    array = ArrayState()
    array.positions = np.stack((a.real, a.imag, np.zeros(m)))
    
    array.positions[0] -= np.mean(array.positions[0])
    array.positions[1] -= np.mean(array.positions[1])

    return array

def cartesian_product(*arrays):
    """
    Return the Cartesian product of 1D NumPy arrays as a 2D array
    of shape (prod(len(ai)), len(arrays)).
    """
    if not arrays:
        return np.empty((0, 0))
    arrays = [np.asarray(a) for a in arrays]
    grids = np.meshgrid(*arrays, indexing='ij')        # one grid per input
    return np.stack(grids, axis=-1).reshape(-1, len(arrays))

def array_cube():
    
    d = 0.1
    array = ArrayState(cartesian_product(np.linspace(-d, d, 2), np.linspace(-d, d, 2), np.linspace(-d, d, 2)).T)
        
    array.positions[0] -= np.mean(array.positions[0])
    array.positions[1] -= np.mean(array.positions[1])

    return array