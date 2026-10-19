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