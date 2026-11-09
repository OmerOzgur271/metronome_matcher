"""Remove the original's drum hits you don't play, so the "cover + original song" mix doesn't clash.

In fills and variations your cover plays other notes than the original. The attack analyser marks
original hits with no partner in your take ("missed"). Here the separated drums are split into
the kick / snare / cymbal bands (low and high pass, the middle is what's left, so the three add
back up exactly) and faded out in two parts:
  - the attack (just before its start to its end): all bands, since a hit's attack is broadband
    (snare wires and kick click reach the cymbal band); whatever you play then comes from your cover;
  - the tail: only the bands that drum rings in (TAIL), up to the drum's next original hit, so a
    crash you don't play is cut instead of left ringing.
That removed part is subtracted from the real song, so everything else stays the original audio.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .audio_io import decode, probe_audio, write_wav
from .config import BANDS

LEAD = 0.008      # s cut before the attack start (the attack's own run-up)
FADE_IN = 0.005   # s fade down before that
FADE_OUT = 0.03   # s fade back up
MIN_ATTACK = 0.03 # s, the all-band cut lasts at least this long
GAP = 0.005       # s left before the drum's next original hit
TAIL = {"Kick": (("Kick",), 0.5), "Snare": (("Snare", "Cymbal"), 0.25), "Cymbal": (("Cymbal",), 1.0)}  # bands, max s


def _split(drums, rate):
    """Kick / snare / cymbal bands of the drums; they sum back to the input exactly."""
    low = sosfiltfilt(butter(4, BANDS[0][2], btype="low", fs=rate, output="sos"), drums, axis=0)
    high = sosfiltfilt(butter(4, BANDS[-1][1], btype="high", fs=rate, output="sos"), drums, axis=0)
    return [low, drums - low - high, high]


def mute_windows(attacks):
    """(band index, from, to) cuts in song s for the original hits you don't play."""
    names = [name for name, _, _ in BANDS]
    starts = {}
    for r in attacks:
        if r["orig"] is not None:
            starts.setdefault(r["band"], []).append(r["orig"]["start"])
    for v in starts.values():
        v.sort()
    out = []
    for r in attacks:
        if r["status"] != "missed":
            continue
        s = r["orig"]["start"]
        attack_end = max(r["orig"]["end"], s + MIN_ATTACK)
        out += [(b, s - LEAD, attack_end) for b in range(len(BANDS))]
        bands, longest = TAIL[r["band"]]
        later = [x for x in starts[r["band"]] if x > s + 0.001]
        tail_end = min(later[0] - GAP if later else s + longest, s + longest)
        out += [(names.index(name), s - LEAD, max(tail_end, attack_end)) for name in bands]
    return out


def _gains(windows, n, rate):
    g = np.ones((len(BANDS), n), np.float32)
    ramp_in, ramp_out = np.linspace(1, 0, round(FADE_IN * rate)), np.linspace(0, 1, round(FADE_OUT * rate))
    for b, lo, hi in windows:
        a, z = max(0, round(lo * rate)), min(n, round(hi * rate))
        if z <= a:
            continue
        curve = np.concatenate([ramp_in, np.zeros(z - a), ramp_out])
        i0 = a - len(ramp_in)
        seg = slice(max(0, i0), min(n, i0 + len(curve)))
        g[b, seg] = np.minimum(g[b, seg], curve[seg.start - i0:seg.stop - i0])
    return g


def discard_unplayed(song, drums, attacks, out_path):
    """Write the song with the original drum hits you don't play taken out. Returns (path, hits removed)."""
    windows = mute_windows(attacks)
    hits = sum(r["status"] == "missed" for r in attacks)
    rate, channels = probe_audio(song)
    channels = min(channels, 2)
    x = decode(song, rate, channels).reshape(-1, channels).astype(np.float64)
    d = decode(drums, rate, channels).reshape(-1, channels).astype(np.float64)
    n = min(len(x), len(d))
    g = _gains(windows, n, rate)
    removed = sum(band * (1 - g[b])[:, None] for b, band in enumerate(_split(d[:n], rate)))
    x[:n] -= removed
    return write_wav(x, rate, out_path), hits
