import numpy as np
from beamforming import create_distance_vectors, polar_to_unitdir, create_steering_vectors
from util import *
from scipy.special import jv
import types

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

    w1_zeni = np.arcsin(np.minimum(1, dj_zeros[o1] / (k * r)))
    look_zeni = zeni_directions(look_direction[None,:])[0]

    zeni_steer = np.minimum(1, k * r / dj_zeros[o1])**2
    zeni_steer1 = k * r * (np.sin(w1_zeni) - np.sin(look_zeni)) * zeni_steer
    kr1n = np.minimum(k * r + zeni_steer1, dj_zeros[o1])
    w1 = np.cos(o1 * relative_angles[None,:]) * (1j**o1) / (jv(o1, kr1n[:,None])) / m

    dv0 = create_distance_vectors(polar_to_unitdir(dir, np.array([np.pi/2])), mic_array)

    w1 *= np.conj(create_steering_vectors(dv0, zeni_steer1[:,None] / r))

    return w1

""" create blended ch / das weights for circular array """
def blended_ch_das_weights(look_direction, order, mic_array, k):

    m = mic_array.shape[1]
    r = np.sqrt(np.sum(mic_array[:,0] * mic_array[:,0]))

    w1 = steered_ch_weights(look_direction, order, mic_array, k)
    das_w1 = create_steering_vectors(create_distance_vectors(look_direction, mic_array), k[:,None]) / m * 2

    for i in range(8):
        if (i&1)^(order&1)^1:
            das_w1[:,i] = np.zeros_like(das_w1[:,i])

    j0_zero = 2.4048
    mlw = np.arcsin(np.minimum(1, j0_zero / (k*r)))

    bstart = 45 / todeg
    bend = 30 / todeg
    mlw = np.minimum(bstart, np.maximum(mlw, bend))

    blend = (bstart - mlw) / (bstart - bend)

    w1 = w1 * (1 - blend)[:,None] + das_w1 * blend[:,None]

    return w1

def optimal_secondary_weights(w1, steering_vectors, constraints, S=None):
    
    N = w1.shape[1]
    nsdim = N - len(constraints)
    assert nsdim > 0

    constraints = [(i @ w1[:,:,None])[:,:,0] for i in constraints]
    constraints = np.stack(constraints, axis=-2)

    # compute orthogonal complement basis matrix
    Vh = np.linalg.svd(np.conj(constraints))[2]
    OH = Vh[:,-nsdim:,:]
    O = np.conj(np.transpose(OH, axes=(0, 2, 1)))

    if isinstance(S, types.NoneType):
        # Use S = np.eye()
        w2 = (O @ np.linalg.pinv(O) @ steering_vectors[:,:,None])[:,:,0]
    else:
        w2 = (O @ np.linalg.inv(OH @ S @ O) @ OH @ steering_vectors[:,:,None])[:,:,0]
    w2 /= np.sum(np.conj(w2) * steering_vectors, axis=-1)[:,None]

    return w2