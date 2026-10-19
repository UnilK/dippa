import numpy as np

def polar_to_unitdir(aziv, zeniv):
    unit_look_direction = np.stack((
        np.cos(aziv) * np.sin(zeniv),
        np.sin(aziv) * np.sin(zeniv),
        np.cos(zeniv)), axis=-1)
    return unit_look_direction

def to_polar_angles(point):
    x, y, z = point
    azi = np.atan2(y, x)
    zeni = np.atan2(np.sqrt(x*x+y*y), z)
    return (azi, zeni)

def create_distance_vectors(look_directions, microphone_positions):
    return look_directions @ microphone_positions

def create_steering_vectors(distance_vectors, k):
    return np.exp(1j * distance_vectors * k)

def das_weights(steering_vectors):
    return steering_vectors / steering_vectors.shape[-1]

def mvdr_weights(R, steering_vectors):
    w = steering_vectors @ np.linalg.inv(R)
    norm = np.sum(np.conj(w) * steering_vectors, axis=0)
    w /= norm[None,:]
    return w

def beamform_wideband(x, w):
    return np.sum(np.conj(w) * x, axis=0)

def fb_shuffle(mp):
    m = mp.shape[1]
    nix = np.array([np.argmin(np.linalg.norm(mp + mp[:,i][:,None], axis=0)) for i in range(m)])
    ix2, ix1 = np.meshgrid(nix, nix)
    return ix1, ix2

def valid_frequency_bins(fmin, fmax, N, sr):
    f = np.fft.fftfreq(N, 1 / sr)
    valid = np.where((f >= fmin) & (f <= fmax))[0]
    return len(valid), f, valid

"""signal has shape (mics, N). Output has shape (mics, len(valid))"""
def compute_snapshot(signal, window, valid):
    return np.fft.rfft(signal * window)[:,valid]

""" input has shape(m, N) output has shape (N, m, m)"""
def compute_covariance_matrices(snapshot):
    snapshot.T[:,:,None] @ np.conj(snapshot.T[:,None,:])

"""audio has shape (mics, samples)"""
def analyze_array_signal(audio, N, step, valid, sensor_array, fb_average):
    window = np.hanning(N)[None,:]
    tidx = []
    snaps = []
    covars = []
    ix1, ix2 = fb_shuffle(sensor_array)
    for i in range(0, audio.shape[1]-N+1, step):
        tidx.append(i)
        snap = compute_snapshot(audio[:,i,i+N], window, valid)
        snaps.append(snap)
        covar = compute_covariance_matrices(snap)
        if fb_average:
            covar = (covar + np.conj(covar[:,ix1, ix2])) * 0.5
        covars.append(covar)
    
    return len(tidx), np.array(tidx), np.array(snaps), np.array(covars)
