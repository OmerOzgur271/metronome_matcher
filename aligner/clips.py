"""Cut each problem range out of your take, with a click at the song's BPM, so you can hear what's off.

Each clip runs from one bar before the range to one bar after it (as far as the take goes).
The click sits on the original's beat grid, measured from the original's hits in the clip, so
it is where the original's beats are: when you're late you hear yourself behind the click,
just as in the final mix. Bar downbeats get a higher, louder click.
"""
from pathlib import Path

import numpy as np

from .audio_io import decode, write_mp3

RATE = 44100
CLICK_SEC = 0.03
PAD_BARS = 1


def _click(freq, gain):
    t = np.arange(int(CLICK_SEC * RATE)) / RATE
    return gain * np.sin(2 * np.pi * freq * t) * np.exp(-t * 120)


def _grid_phase(ref_times, bpm):
    """Offset of the original's 16th-note grid from bar 1 at 0 s (circular mean of the hits' phases)."""
    if not len(ref_times):
        return 0.0
    sixteenth = 15 / bpm
    angle = np.angle(np.mean(np.exp(2j * np.pi * np.asarray(ref_times) / sixteenth)))
    return angle / (2 * np.pi) * sixteenth


def click_track(song_from, n, bpm, phase):
    """Clicks for n samples starting at song time song_from; beat k of the song is at k × beat + phase."""
    beat = 60 / bpm
    out = np.zeros(n, np.float32)
    accent, tick = _click(1600, 1.0), _click(1000, 0.6)
    for k in range(int(np.ceil((song_from - phase) / beat)), int((song_from + n / RATE - phase) / beat) + 1):
        i = int(round((k * beat + phase - song_from) * RATE))
        c = accent if k % 4 == 0 else tick
        if 0 <= i < n:
            out[i:i + len(c)] += c[:n - i]
    return out


def problem_clips(cover_path, ref_path, t, ranges, out_dir, click_level=0.6, original_level=0.0):
    """Write one mp3 per range: your take + click (+ the original drums when original_level > 0).

    Levels are relative to the loudest point of your take in the clip. Returns the paths, in the
    order of `ranges`.
    """
    if not ranges:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bpm, start = t["bpm"], t["start"]
    bar_len = 240 / bpm
    cover = decode(cover_path, RATE, 2)
    ref = decode(ref_path, RATE, 2) if original_level > 0 else None
    paths = []
    for i, r in enumerate(ranges, 1):
        song_from = max((r["first"] - 1 - PAD_BARS) * bar_len, start)
        song_to = min((r["last"] + PAD_BARS) * bar_len, start + len(cover) / RATE)
        a = int(round((song_from - start) * RATE))
        clip = cover[a:a + int(round((song_to - song_from) * RATE))].copy()
        peak = float(np.max(np.abs(clip))) or 1.0
        hits = [h["ref_t"] for h in t["hits"] if h["err"] is not None and song_from <= h["ref_t"] <= song_to]
        clip += click_level * peak * click_track(song_from, len(clip), bpm, _grid_phase(hits, bpm))[:, None]
        if ref is not None:
            b = int(round(song_from * RATE))
            orig = ref[max(b, 0):b + len(clip)]
            clip[:len(orig)] += original_level * peak / (float(np.max(np.abs(orig))) or 1.0) * orig
        clip *= min(1.0, 0.98 / (float(np.max(np.abs(clip))) or 1.0))
        paths.append(write_mp3(clip, RATE, out_dir / f"problem{i}_bars{r['first']}-{r['last']}.mp3"))
    return paths
