import numpy as np
from beamforming import create_distance_vectors, polar_to_unitdir, create_steering_vectors
from util import *
from scipy.special import jv

def planar_directions(directions):
    return np.atan2(directions[:,1], directions[:,0])

def zeni_directions(directions):
    return np.atan2(np.sqrt(directions[:,0]**2 + directions[:,1]**2), directions[:,2])

""" create steered circular harmonic weights for circular array"""
def steered_ch_weights(look_direction, order, mic_array, k):

    mic = planar_directions(mic_array.T)
    dir = planar_directions(look_direction[None,:])
    relative_angles = dir[0] - mic

    m = mic_array.shape[1]
    r = np.sqrt(np.sum(mic_array[:,0] * mic_array[:,0]))

    o1 = order
    dj_zeros = [0, 1.8411837813, 3.0542369282, 4.2011889412]

    w1_zeni = np.arcsin(min(1, dj_zeros[o1] / (k * r)))
    look_zeni = zeni_directions(look_direction[None,:])[0]

    zeni_steer = min(1, k * r / dj_zeros[o1])**2
    zeni_steer1 = k * r * (np.sin(w1_zeni) - np.sin(look_zeni)) * zeni_steer
    kr1n = min(k * r + zeni_steer1, dj_zeros[o1])
    w1 = np.cos(o1 * relative_angles) * (1j**o1) / (jv(o1, kr1n)) / m

    dv0 = create_distance_vectors(polar_to_unitdir(dir, np.array([np.pi/2])), mic_array)

    sv1 = create_steering_vectors(dv0, zeni_steer1 / r)

    correction1 = np.conj(sv1)

    w1 = w1 * correction1[0]

    return w1


""" create blended ch / das weights for circular array """
def blended_ch_das_weights(look_direction, order, mic_array, k):

    m = mic_array.shape[1]
    r = np.sqrt(np.sum(mic_array[:,0] * mic_array[:,0]))

    w1 = steered_ch_weights(look_direction, order, mic_array, k)
    das_w1 = create_steering_vectors(create_distance_vectors(look_direction, mic_array), k) / m * 2

    for i in range(8):
        if i&1:
            das_w1[i] = 0

    j0_zero = 2.4048
    mlw = np.arcsin(min(1, j0_zero / (k*r)))

    bstart = 45 / todeg
    bend = 30 / todeg
    mlw = min(bstart, max(mlw, bend))

    blend = (bstart - mlw) / (bstart - bend)

    w1 = w1 * (1 - blend) + das_w1 * blend

    return w1

def create_optimal_secondary_weights(w1, look_vector, constraints):
    
    N = w1.shape[0]
    nsdim = N - len(constraints)
    assert nsdim > 0

    # compute orthogonal complement basis matrix
    constraints = [i / np.linalg.norm(i) for i in constraints]
    for i in range(N-nsdim):
        for j in range(0, i):
            constraints[i] -= (constraints[i] @ np.conj(constraints[j])) * constraints[j]
            constraints[i] /= np.linalg.norm(constraints[i])

    O = []
    for i in range(nsdim):
        basisv = np.random.randn(N).astype(np.complex128)
        basisv /= np.linalg.norm(basisv)
        for v in constraints:
            basisv -= (basisv @ np.conj(v)) * v
            basisv /= np.linalg.norm(basisv)
        constraints.append(basisv)
        O.append(basisv)

    O = np.array(O).T

    # optimize white noise gain.

    # w2 = O @ np.linalg.inv(np.conj(O).T @ O) @ np.conj(O).T @ look_vector
    w2 = O @ np.linalg.pinv(O) @ look_vector
    w2 = w2 / (np.conj(w2) @ look_vector)

    return w2
