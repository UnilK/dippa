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

def extend_noise(x, target_samples):
    if len(x) < target_samples: 
        X = np.fft.rfft(x)
        X *= np.exp(1j*np.random.rand(len(X))*2*np.pi)
        x = np.tile(np.fft.irfft(X), (target_samples-1)//len(x)+1)
    return x[:target_samples]

def extend_audio(x, target_samples):
    if len(x.shape) > 1:
        if x.shape[1] >= target_samples:
            return x[:,:target_samples]
        fade = 2**11
        n = x.shape[1]
        window = np.minimum(np.minimum(np.arange(n)/fade, n/fade-np.arange(n)/fade), 1)
        grain = x * (0.5 - 0.5 * np.cos(np.pi*window))[None,:]
        y = np.zeros(((target_samples-1)//n+2)*n + 4 * fade)
        for i in range(0, len(y)-n-1, n-fade):
            y[i:i+n] += grain
        return y[fade:fade+target_samples]
    else: 
        if len(x) >= target_samples:
            return x[:target_samples]
        fade = 2**11
        n = len(x)
        window = np.minimum(np.minimum(np.arange(n)/fade, n/fade-np.arange(n)/fade), 1)
        grain = x * (0.5 - 0.5 * np.cos(np.pi*window))
        y = np.zeros(((target_samples-1)//n+2)*n + 4 * fade)
        for i in range(0, len(y)-n-1, n-fade):
            y[i:i+n] += grain
        return y[fade:fade+target_samples]

def get_audio_from_source(source, mono=True):
    if isinstance(source, str):
        audio, sr = sf.read(source, always_2d=True)
        if audio.size == 0:
            raise RuntimeError("Empty or unsupported audio file.")
        if mono:
            return (audio[:, 0], sr)
        else:
            return (audio.T, sr)
    elif isinstance(source, tuple):
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

def prepare_audio(audio_source, duration, sr, toMono=True):
    
    audio, sr2 = get_audio_from_source(audio_source, toMono)
    if sr2 != sr:
        if len(audio.shape) > 1:
            audio = np.array([resamplemod.sinc_resample(a, sr/sr2, 16) for a in audio])
        else:
            audio = resamplemod.sinc_resample(audio, sr/sr2, 16)

    audio = rms_normalize(audio)
    return extend_audio(audio, duration * sr)

def create_uncorrelated_noise(m, n):
    return rms_normalize(np.random.randn(m, n))

def create_diffuse_noise(n, micpos, sr):
    m = micpos.shape[1]
    noise = np.zeros((m, n), dtype=np.float32)
    for i in range(4):
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

def prepare_scene_audio(scene, sensor_array, target_audio, duration, sr, gain = 1):
    target_audio = prepare_audio(target_audio, duration, sr) * gain
    return propagate_audio(scene.target_position, sensor_array.positions, target_audio, sr)

def prepare_drone_audio(scene, sensor_array, target_audio, duration, sr, gain = 1):
    target_audio = prepare_audio(target_audio, duration, sr) * gain
    return simulate_drone(scene.target_position, sensor_array.positions, target_audio, sr)
    