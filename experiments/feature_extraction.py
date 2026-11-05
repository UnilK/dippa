import numpy as np
from beamforming import valid_frequency_bins
from scipy.fft import dct

SR = 44100
WINDOW_LENGTH = 2**10
WINDOW_STEP = WINDOW_LENGTH // 2
WINDOW = np.hanning(WINDOW_LENGTH)**0.5
FEATURE_LENGTH = (SR - WINDOW_LENGTH + WINDOW_STEP - 1) // WINDOW_STEP
NMFCC = 24
FMIN = 750
FMAX = 6000
NSTFT, VALID_FREQUENCIES, VALID_FREQUENCY_INDICES = valid_frequency_bins(FMIN, FMAX, WINDOW_LENGTH, SR)

def hz_to_mel(f):
    # HTK-style mel
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10.0**(m / 2595.0) - 1.0)

def mel_filterbank(sr, n_fft, n_mels=40, fmin=0.0, fmax=None, normalize='area'):
    """
    Build a triangular mel filterbank matrix of shape (n_mels, n_fft//2 + 1)
    mapping power spectrum bins to mel bands.
    normalize:
      - None: no normalization
      - 'area': each filter sums to 1 (common for MFCC)
    """
    if fmax is None:
        fmax = sr / 2.0

    # One-sided rfft bin frequencies
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_bins)

    # Mel-scaled points (n_mels + 2 to include the edges)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)

    # Bin indices for each mel point
    bin_idx = np.floor((n_fft + 1) * hz / sr).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Build filters
    fb = np.zeros((n_mels, n_bins), dtype=float)
    for m in range(1, n_mels + 1):
        left = bin_idx[m - 1]
        center = bin_idx[m]
        right = bin_idx[m + 1]

        if center > left:
            fb[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)

    if normalize == 'area':
        # Normalize each filter to unit sum (avoids scale dependence)
        sums = fb.sum(axis=1, keepdims=True)
        # Avoid divide-by-zero for degenerate bands
        fb = np.divide(fb, sums, out=np.zeros_like(fb), where=(sums > 0))

    return fb

def compute_mfcc_from_psd(
    psd,
    sr,
    n_mels=40,
    n_mfcc=13,
    fmin=0.0,
    fmax=None,
    lifter=0,
    eps=1e-10
):
    """
    psd: power spectrum from rfft
         shape (n_bins,) for a single frame or (n_frames, n_bins) for multiple frames.
         Assumed to be power (|rfft|^2 / ...). If it's magnitude, square it first.
    sr: sample rate
    n_mels: number of mel bands
    n_mfcc: number of cepstral coefficients to return
    lifter: cepstral lifter coefficient (0 disables). Typical 20–22.
    """
    psd = np.asarray(psd)
    single_frame = False
    if psd.ndim == 1:
        psd = psd[None, :]  # make it (1, n_bins)
        single_frame = True
    elif psd.ndim != 2:
        raise ValueError("psd must be 1D or 2D (frames x freq_bins)")

    n_bins = psd.shape[-1]
    # Infer n_fft from rfft length: n_bins = n_fft//2 + 1
    n_fft = (n_bins - 1) * 2

    # Build mel filterbank (maps freq bins -> mel bands)
    fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, normalize='area')

    # Mel energies
    # Shape: (frames, n_mels)
    mel_energies = psd @ fb.T

    # Log compression
    log_mel = np.log(mel_energies + eps)

    # DCT-II to get cepstral coefficients (keep first n_mfcc)
    mfcc = dct(log_mel, type=2, axis=-1, norm='ortho')[..., :n_mfcc]

    # Optional cepstral liftering to emphasize higher-order coefficients slightly
    if lifter and lifter > 0:
        n = np.arange(n_mfcc)
        lift = 1 + (lifter / 2.0) * np.sin(np.pi * n / lifter)
        mfcc = mfcc * lift

    if single_frame:
        return mfcc[0]
    return mfcc

def extract_raw_mfcc(psd, sr, fmin, fmax, N, validf, nmfcc=NMFCC):
    feat = np.zeros((len(psd), N//2+1))
    feat[:,validf] = psd
    return compute_mfcc_from_psd(feat, sr=sr, n_mels=40, n_mfcc=nmfcc, fmin=fmin, fmax=fmax, lifter=22)

def normalize_raw_mfcc(mfcc):
    mfcc = mfcc - np.max(mfcc)
    mfcc = np.maximum(mfcc, -60)
    mfcc /= np.abs(np.min(mfcc))
    mfcc += 1
    return mfcc

def extract_mfcc_feature(psd, sr, fmin, fmax, N, validf, nmfcc=NMFCC):
    return normalize_raw_mfcc(extract_raw_mfcc(psd, sr, fmin, fmax, N, validf, nmfcc))

def extract_raw_stft(psd):
    return np.pow(psd, 0.15)

def normalize_raw_stft(feature):
    return feature / np.max(feature)

def extract_stft_feature(psd):
    return normalize_raw_stft(extract_raw_stft(psd))