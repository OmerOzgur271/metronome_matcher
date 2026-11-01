"""Onset envelopes, peak picking and hit matching."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .config import BANDS, SR, TOL
from .helpers import segment


def _band_flux(x, start, n, hop, win):
    """Raw per-band spectral flux for n frames; frame f is centred on sample start + f*hop."""
    seg = segment(x, start - hop - win // 2, start + n * hop + win // 2)
    frames = np.lib.stride_tricks.sliding_window_view(seg, win)[::hop][:n + 1]
    spec = np.log1p(10 * np.abs(np.fft.rfft(frames * np.hanning(win), axis=1)) * 2 / win)
    flux = np.maximum(0, np.diff(spec, axis=0))
    freqs = np.fft.rfftfreq(win, 1 / SR)
    return np.array([flux[:, (freqs >= lo) & (freqs <= hi)].sum(1) for _, lo, hi in BANDS])


def _transients(flux, hop):
    radius = round(0.1 * SR / hop)
    envs = []
    for e in flux:
        e = np.maximum(0, e - uniform_filter1d(e, 2 * radius + 1))  # keep only transients
        rms = np.sqrt((e ** 2).mean())
        envs.append(e / rms if rms > 0 else e)
    return np.array(envs)


def envelopes(x, start, length, hop, win):
    """Per-band onset envelopes (spectral flux). Frame f is centred on sample start + f*hop."""
    return _transients(_band_flux(x, start, length // hop, hop, win), hop)


def long_envelopes(x, start, length, hop, win, chunk=8192):
    """Same as envelopes(), computed in chunks so a whole song fits in memory."""
    n = length // hop
    flux = [_band_flux(x, start + i * hop, min(chunk, n - i), hop, win) for i in range(0, n, chunk)]
    return _transients(np.concatenate(flux, axis=1), hop)


def onsets(e, frame_sec, t0):
    peaks, _ = find_peaks(e, height=e.mean() + e.std(), distance=max(1, round(0.03 / frame_sec)))
    return t0 + peaks * frame_sec


def local_onsets(e, frame_sec, t0, window=3.0, min_gap=0.03):
    """onsets() with a threshold that follows the local level, for whole songs with quiet and loud parts.

    Returns (times, strengths); strength is the peak height over the band's median peak height.
    """
    k = max(1, round(window / frame_sec))
    m = uniform_filter1d(e, k)
    s = np.sqrt(np.maximum(0, uniform_filter1d(e ** 2, k) - m ** 2))
    floor = 0.5 * (e.mean() + e.std())  # ignore noise in near-silent stretches
    peaks, props = find_peaks(e, height=np.maximum(m + s, floor), distance=max(1, round(min_gap / frame_sec)))
    h = props["peak_heights"]
    return t0 + peaks * frame_sec, h / np.median(h) if len(h) else h


def match(cover_on, ref_on, tol=TOL):
    used, pairs = set(), []
    for t in cover_on:
        d = np.abs(ref_on - t)
        for j in np.argsort(d):
            if d[j] > tol:
                break
            if j not in used:
                used.add(j)
                pairs.append((t, ref_on[j]))
                break
    return pairs
