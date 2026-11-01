"""Small numeric and formatting helpers used across the package."""
import numpy as np


def block_start(bpm, block):
    return (block - 1) * 4 * 60 / bpm  # 4/4: bar = 4 beats


def segment(x, a, b):
    """x[a:b], zero-padded where the range falls outside x."""
    out = np.zeros(b - a, np.float32)
    lo, hi = max(a, 0), min(b, len(x))
    if hi > lo:
        out[lo - a:hi - a] = x[lo:hi]
    return out


def ncc(x, y):
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / d) if d > 1e-12 else 0.0


def ncc_bands(X, Y, xo, yo, m):
    # Mean of per-band NCCs: a one-beat shift (kick landing on the snare) must not score well.
    return np.mean([ncc(X[b, xo:xo + m], Y[b, yo:yo + m]) for b in range(len(X))])


def parabolic_offset(a, b, c):
    """Sub-sample peak offset in [-0.5, 0.5] from three neighbouring values around peak b."""
    if a - 2 * b + c < 0:
        return float(np.clip(0.5 * (a - c) / (a - 2 * b + c), -0.5, 0.5))
    return 0.0


def local_peaks(curve):
    """Indices of local maxima (edges included), strongest first."""
    is_peak = np.r_[True, curve[1:] >= curve[:-1]] & np.r_[curve[:-1] >= curve[1:], True]
    return sorted(np.flatnonzero(is_peak), key=lambda i: -curve[i])


def clock_time(t):
    """Seconds as '0:12.345, 12 s 345 ms'."""
    return f"{int(t // 60)}:{t % 60:06.3f}, {int(t)} s {round((t % 1) * 1000)} ms"
