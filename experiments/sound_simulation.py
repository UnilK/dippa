import sys
import numpy as np
import scipy
import soundfile as sf

build_dir = "../cpp/build"
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import propagationmod
import resamplemod

def fftwin_filt(x, filt):
    N = 2 * len(filt) - 2
    imp = np.fft.irfft(filt)
    imp = np.roll(imp, N//2)
    imp *= np.hanning(N)
    y = scipy.signal.convolve(x, imp)
    return y[N//2: -N//2+1]

def extend_noise(x, target_samples):
    if len(x) < target_samples: 
        X = np.fft.rfft(x)
        X *= np.exp(1j*np.random.rand(len(X))*2*np.pi)
        x = np.tile(np.fft.irfft(X), (target_samples-1)//len(x)+1)
    return x[:target_samples]

def extend_audio(x, target_samples):
    if len(x) >= target_samples:
        return x[: target_samples]
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
        data, sr = sf.read(source, always_2d=True)
        if data.size == 0:
            raise RuntimeError("Empty or unsupported audio file.")
        if mono:
            audio = data[:, 0]
        return (audio, sr)
    elif isinstance(source, tuple):
        return source
    else:
        raise TypeError("bad scene audio type")

def propagate_audio(position_data, mic_positions, audio, sr):
    if isinstance(position_data, tuple):
        pos, vel = position_data
        return propagationmod.simulate_audio_propagation(sr, mic_positions, pos, vel, audio)
    else:
        return propagationmod.static_audio_propagation(sr, mic_positions, position_data, audio)

def rms_normalize(x):
    return x / (np.sqrt(np.mean(x * x)) + 1e-30)

def prepare_audio(audio_source, mic_array, position, total_samples, sr):
    
    audio, sr2 = get_audio_from_source(audio_source)
    
    if sr2 != sr:
        audio = resamplemod.sinc_resample(audio, sr/sr2, 16)
    
    audio = rms_normalize(audio)
    audio = extend_audio(audio, total_samples)
    audio = propagate_audio(position, mic_array, audio, sr)

    return audio

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