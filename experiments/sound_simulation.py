import sys
import numpy as np
import scipy
import soundfile as sf
from medium_attenuation import air_attenuation
from util import *

build_dir = "../cpp/build"
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import propagationmod
import resamplemod

def fftwin_filt(x, filt, cut=True):
    N = 2 * len(filt) - 2
    imp = np.fft.irfft(filt)
    imp = np.roll(imp, N//2)
    imp *= np.hanning(N)
    y = None
    if len(x.shape) > 1:
        y = scipy.signal.convolve(x, imp[None,:])
        if cut:
            return y[:,N//2: -N//2+1]
    else:
        y = scipy.signal.convolve(x, imp)
        if cut:
            return y[N//2: -N//2+1]
    return y

def design_bump_filter(sr, mid, wid, N=2**11):
    fmin = mid-wid
    fmax = mid+wid
    mi = N / sr * fmin
    ma = N / sr * fmax
    filt = np.zeros(N//2+1)
    for i in range(0, N//2+1):
        f = sr / N * i
        if mi <= i+1 and i <= ma:
            filt[i] = 0.5 * np.cos(((mid - f) / wid) * np.pi) + 0.5
    return filt

def extend_noise(x, target_samples):
    if len(x) < target_samples: 
        X = np.fft.rfft(x)
        X *= np.exp(1j*np.random.rand(len(X))*2*np.pi)
        x = np.tile(np.fft.irfft(X), (target_samples-1)//len(x)+1)
    return x[:target_samples]

def extend_audio(x, target_samples, fade_len=256):
    """
    Repeat audio to reach target length with crossfaded loop points.

    Parameters
    ----------
    x : np.ndarray
        Audio array with shape (T,) or (C, T). Time is the last axis.
    target_samples : int
        Desired length along the time axis.
    fade_len : int
        Crossfade length in samples at each repeat boundary.
        Use 0 for no crossfade.

    Returns
    -------
    y : np.ndarray
        Repeated audio with shape (target_samples,) or (C, target_samples),
        matching the input's dimensionality.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x_proc = x[None, :]          # (1, T)
        squeeze_back = True
    elif x.ndim == 2:
        x_proc = x                    # (C, T)
        squeeze_back = False
    else:
        raise ValueError("x must have shape (T,) or (C, T)")

    C, T = x_proc.shape
    if target_samples <= 0:
        return np.empty((target_samples,), dtype=x.dtype) if squeeze_back else np.empty((C, target_samples), dtype=x.dtype)

    # If input already long enough, just trim.
    if T >= target_samples:
        y = x_proc[:, :target_samples]
        return y[0] if squeeze_back else y

    # Clamp fade_len to a safe range.
    if fade_len is None:
        fade_len = 0
    fade_len = int(max(0, fade_len))
    if fade_len > 0:
        # Cannot crossfade more than half of the source, and
        # keep it smaller than a fraction of the target.
        fade_len = min(fade_len, max(1, T // 2), max(1, target_samples // 4))

    # Fast path: no crossfade (tile and trim).
    if fade_len == 0 or T < 2:
        reps = int(np.ceil(target_samples / T))
        y = np.tile(x_proc, reps=reps)[:, :target_samples]
        return y[0] if squeeze_back else y

    # Build crossfade ramps
    fade_out = np.linspace(1.0, 0.0, fade_len, endpoint=False, dtype=np.float64)  # length fade_len
    fade_in  = 1.0 - fade_out

    # Accumulate repeats with overlap-add at boundaries
    y = x_proc.copy()
    # Loop until we reach or exceed target_samples
    while y.shape[1] < target_samples:
        # Overlap last fade_len of y with first fade_len of x_proc
        prev_tail = y[:, -fade_len:].astype(np.float64, copy=False)
        next_head = x_proc[:, :fade_len].astype(np.float64, copy=False)

        cross = prev_tail * fade_out + next_head * fade_in  # shape (C, fade_len)

        # Concatenate: y without its last fade_len, then crossfade, then remainder of x_proc
        y = np.concatenate(
            [y[:, :-fade_len], cross.astype(y.dtype, copy=False), x_proc[:, fade_len:]],
            axis=1
        )

    # Trim to target length
    y = y[:, :target_samples]

    return y[0] if squeeze_back else y
    

def get_audio_from_source(source, toMono=True):
    if isinstance(source, str):
        audio, sr = sf.read(source, always_2d=True)
        if audio.size == 0:
            raise RuntimeError("Empty or unsupported audio file.")
        if toMono:
            return (audio[:, 0], sr)
        else:
            return (audio.T, sr)
    elif isinstance(source, tuple):
        if toMono:
            if len(source[0].shape) > 1:
                return (source[0][0,:], source[1])
            else:
                return source
        else:
            return source
    else:
        raise TypeError("bad scene audio type")

def air_attenuate(audio, sr, position_data):
    N = 64
    f = np.fft.rfftfreq(N, 1/sr)
    a = db2lin(air_attenuation(f).freq[0])
    if isinstance(position_data, tuple):
        n, m = audio.shape()
        filtered = np.zeros((m, n+2*N))
        pos, _ = position_data
        for i in range(0, n-N+1, N):
            filtered[:,i:i+2*N-1] = fftwin_filt(audio[:,i:i+N], a**np.linalg.norm(pos[:,i]), False)
        return filtered[:,N:-N]
    else:
        return fftwin_filt(audio, a**np.linalg.norm(position_data))

def propagate_audio(position_data, mic_positions, audio, sr):
    x = None
    if isinstance(position_data, tuple):
        pos, vel = position_data
        x = propagationmod.simulate_audio_propagation(sr, mic_positions, pos, vel, audio)
    else:
        x = propagationmod.static_audio_propagation(sr, mic_positions, position_data, audio)
    return air_attenuate(x, sr, position_data)

def propagate_planewave_audio(position_data, mic_positions, audio, sr):
    x = None
    if isinstance(position_data, tuple):
        pos, vel = position_data
        x = propagationmod.planewave_propagation(sr, np.linalg.norm(pos, axis=1), mic_positions, pos, audio)
    else:
        x = propagationmod.static_planewave_propagation(
            sr, np.linalg.norm(position_data), mic_positions, position_data, audio)
    return air_attenuate(x, sr, position_data)

def simulate_drone(position_data, mic_positions, audio, sr):
    x = None
    if isinstance(position_data, tuple):
        pos, vel = position_data
        x = propagationmod.simulate_drone(sr, mic_positions, pos, vel, audio)
    else:
        x = propagationmod.simulate_static_drone(sr, mic_positions, position_data, audio)
    return air_attenuate(x, sr, position_data)

def rms_normalize(x):
    return x / (np.sqrt(np.mean(x * x)) + 1e-30)

def prepare_audio(audio_source, duration, sr, toMono=True, normalize=True):
    
    audio, sr2 = get_audio_from_source(audio_source, toMono)
    if sr2 != sr:
        if len(audio.shape) > 1:
            audio = np.array([resamplemod.sinc_resample(a, sr/sr2, 16) for a in audio])
        else:
            audio = resamplemod.sinc_resample(audio, sr/sr2, 16)

    if normalize:
        audio = rms_normalize(audio)
    return extend_audio(audio, duration * sr)

def create_uncorrelated_noise(m, n):
    return rms_normalize(np.random.randn(m, n))

def create_diffuse_noise(n, micpos, sr, quality=1):
    m = micpos.shape[1]
    noise = np.zeros((m, n), dtype=np.float32)
    for i in range(quality):
        # simulate plane waves arriving from random directions.
        pos = np.random.randn(3, n).astype(np.float32)
        pos /= np.linalg.norm(pos, axis=0)
        noise += propagationmod.planewave_propagation(sr, 1.0, micpos, pos, np.random.randn(n))
    noise = rms_normalize(noise)
    return noise

def pink_filter(audio, sr):
    N = sr//100
    f = np.fft.rfftfreq(N)
    f[0] = 1e18
    filt = 1 / f
    filt /= np.linalg.norm(filt)
    return fftwin_filt(audio, filt)

def pink_diffuse_noise(n, micpos, sr):
    noise = create_diffuse_noise(n, micpos, sr)
    return pink_filter(noise, sr)

def prepare_spatial_audio(scene, sensor_array, target_audio, duration, sr, gain = 1, normalize=True):
    target_audio = prepare_audio(target_audio, duration, sr, normalize=normalize) * gain
    return propagate_audio(scene.target_position, sensor_array.positions, target_audio, sr)

def prepare_planewave_audio(scene, sensor_array, target_audio, duration, sr, gain = 1, normalize=True):
    target_audio = prepare_audio(target_audio, duration, sr, normalize=normalize) * gain
    return propagate_planewave_audio(scene.target_position, sensor_array.positions, target_audio, sr)

def prepare_drone_audio(scene, sensor_array, target_audio, duration, sr, gain = 1, normalize=True):
    target_audio = prepare_audio(target_audio, duration, sr, normalize=normalize) * gain
    return simulate_drone(scene.target_position, sensor_array.positions, target_audio, sr)

def calibrate_recorded_audio(audio, calibration):
    return audio * calibration[:,None]