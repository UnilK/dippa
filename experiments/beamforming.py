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

def wideband_das_weights(steering_vectors):
    return steering_vectors / steering_vectors.shape[-1]

def diffuse_covariance_matrices(sensor_array, k):
    m = sensor_array.shape[1]
    dist = np.zeros((m, m), dtype=np.complex128)
    for i in range(m):
        for j in range(m):
            d = sensor_array[:,i] - sensor_array[:,j]
            dist[i,j] = np.sqrt(np.sum(d*d))
    return np.sinc(k[:,None,None] * dist[None,:,:])

def diagonal_load(covariance_matrices, LNR):
    m = covariance_matrices.shape[-1]
    loading = LNR * np.trace(covariance_matrices, axis1=-2, axis2=-1).real / m
    return covariance_matrices + loading[:,None,None] * np.eye(m)[None,:,:]

def wideband_mvdr_weights(R, steering_vectors):
    w = (np.linalg.inv(R) @ steering_vectors[:,:,None])[:,:,0]
    norm = np.sum(np.conj(w) * steering_vectors, axis=-1)
    w /= norm[:,None]
    return w

def beamform_wideband(x, w):
    return np.sum(np.conj(w) * x, axis=1)

def valid_frequency_bins(fmin, fmax, N, sr):
    """ return number of valid bins, frequencies, valid indices"""
    f = np.fft.fftfreq(N, 1 / sr)
    validf = np.where((f >= fmin) & (f <= fmax))[0]
    return len(validf), f[validf], validf

def compute_snapshot(signal, window, validf):
    """signal has shape (mics, N). Output has shape (mics, len(valid))"""
    return np.fft.rfft(signal * window)[:,validf]

def compute_covariance_matrices(snapshot):
    """ input has shape(m, N) output has shape (N, m, m)"""
    return snapshot.T[:,:,None] @ np.conj(snapshot.T[:,None,:])

def fb_average(covar, sensor_array):
    mp = sensor_array
    m = mp.shape[1]
    nix = np.array([np.argmin(np.linalg.norm(mp + mp[:,i][:,None], axis=0)) for i in range(m)])
    ix2, ix1 = np.meshgrid(nix, nix)
    return (covar + np.conj(covar[...,ix1, ix2])) * 0.5

def stft(audio, window, step, validf):
    """
    audio has shape (mics, samples). window should have even length.
    returns number of snapshots, time indices, stft.
    stft has shape (t, mics, f)
    """
    if len(window.shape) == 1:
        window = window[None,:]
    N = window.shape[1]
    tidx = []
    snaps = []
    for i in range(0, audio.shape[1]-N+1, step):
        tidx.append(i)
        snap = compute_snapshot(audio[:,i:i+N], window, validf)
        snaps.append(snap)
    
    return len(tidx), np.array(tidx), np.array(snaps)

def istft(snapshots, window, tidx, validf):
    """snapshots is a single-channel stft with shape (t, f). window should have enven length"""
    if len(window.shape) == 1:
        window = window[None,:]
    N = window.shape[1]
    audio = np.zeros(tidx[-1]+N, dtype=np.float32)
    norm = np.zeros(tidx[-1]+N, dtype=np.float32)
    for i in range(len(tidx)):
        freq = np.zeros(N//2 + 1, dtype=np.complex64)
        freq[validf] = snapshots[i,:]
        t = tidx[i]
        audio[t:t+N] += (np.fft.irfft(freq) * window)[0]
        norm[t:t+N] += (window**2)[0]
    
    return audio / np.maximum(norm, np.max(norm)*0.5)

def psd_to_sound(psd_snapshots, audio_snapshots, window, tidx, validf):
    """snapshots is a single-channel stft with shape (t, f). window should have enven length"""
    if len(window.shape) == 1:
        window = window[None,:]
    N = window.shape[1]
    audio = np.zeros(tidx[-1]+N, dtype=np.float32)
    norm = np.zeros(tidx[-1]+N, dtype=np.float32)
    for i in range(len(tidx)):
        freq = np.zeros(N//2 + 1, dtype=np.complex64)
        freq[validf] = audio_snapshots[i,:] / (np.abs(audio_snapshots[i,:]) + 1e-18)
        freq[validf] *= np.sqrt(psd_snapshots[i,:])
        t = tidx[i]
        audio[t:t+N] += (np.fft.irfft(freq) * window)[0]
        norm[t:t+N] += (window**2)[0]
    
    return audio / np.maximum(norm, np.max(norm)*0.5)

def smoothing_alpha(duration_samples, step, decay):
    return decay**(step / duration_samples)

def time_average(val, alpha):
    """val has shape (t, ...). Values in ... are averaged over time"""
    val_smooth = np.copy(val)
    val_smooth[0,...] *= (1.0 - alpha)
    for i in range(1, len(val_smooth)):
        val_smooth[i,...] = val_smooth[i,...] * (1.0 - alpha) + val_smooth[i-1,...] * alpha
    return val_smooth

def total_time_average(val):
    """val has shape (t, ...). Values in ... are averaged over time"""
    return np.sum(val, axis=0) / val.shape[0]