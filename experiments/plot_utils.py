import matplotlib.pyplot as plt
import numpy as np
from util import *

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

def plot_stft_feature(ax, f, t, feature, vmin=0, vmax=1, log_freq=False, y_label=True, x_label=True):
    # Convert frequency to kHz for display
    y = f / 1000.0
    mesh = ax.pcolormesh(t, y, feature, shading='nearest', cmap="bone", vmin=vmin, vmax=vmax)
    if y_label:
        ax.set_ylabel("Frequency (kHz)")
    if x_label:
        ax.set_xlabel("Time (s)")
    if log_freq:
        ax.set_yscale("log")
    cbar = plt.colorbar(mesh, ax=ax, format="%.1f")
    cbar.set_label("Activation")

def plot_mfcc_feature(ax, t, feature, vmin=0, vmax=1, log_freq=False, y_label=True, x_label=True):
    # Convert frequency to kHz for display
    y = np.arange(feature.shape[0])
    mesh = ax.pcolormesh(t, y, feature, shading='nearest', cmap="bone", vmin=vmin, vmax=vmax)
    if y_label:
        ax.set_ylabel("Coefficient")
    if x_label:
        ax.set_xlabel("Time (s)")
    if log_freq:
        ax.set_yscale("log")
    cbar = plt.colorbar(mesh, ax=ax, format="%.1f")
    cbar.set_label("Activation")

def plot_mesh_polar(ax, plot_points, z):
    aziv, zeniv = plot_points
    c = ax.pcolormesh(aziv, zeniv * todeg, np.abs(z), shading='gouraud', cmap='viridis')
    ax.set_ylim([0, 90 + 1e-5])
    plt.colorbar(c, ax=ax)