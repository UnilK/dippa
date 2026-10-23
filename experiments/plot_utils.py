import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(ax, f, t, S_db, vmin=-80, vmax=None, log_freq=False, y_label=True, x_label=True):
    # Convert frequency to kHz for display
    y = f / 1000.0
    mesh = ax.pcolormesh(t, y, S_db, shading='nearest', cmap="magma", vmin=vmin, vmax=vmax)
    if y_label:
        ax.set_ylabel("Frequency (kHz)")
    if x_label:
        ax.set_xlabel("Time (s)")
    if log_freq:
        ax.set_yscale("log")
    cbar = plt.colorbar(mesh, ax=ax, format="%.0f dB")
    cbar.set_label("Magnitude (dB)")