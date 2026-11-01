"""Checks that check_timing() finds injected timing mistakes and tempo drift."""
import numpy as np

from aligner import block_start, check_timing
from aligner.config import SR

BPM = 100.0
BEAT = 60 / BPM


def _groove(seconds):
    """(time, drum) for kick on 1 & 3, snare on 2 & 4, 8th-note hats."""
    ev = []
    for i in range(int(seconds / (BEAT / 2))):
        t = i * BEAT / 2
        ev.append((t, "hat"))
        if i % 4 == 0:
            ev.append((t, "kick"))
        if i % 4 == 2:
            ev.append((t, "snare"))
    return ev


def _render(events, seconds, rng, kick_hz=55):
    x = np.zeros(int(seconds * SR), np.float32)
    t = np.arange(int(0.25 * SR)) / SR
    for when, drum in events:
        if drum == "kick":
            v = np.sin(2 * np.pi * kick_hz * t) * np.exp(-t * 18)
        elif drum == "snare":
            v = 0.6 * rng.normal(0, 1, len(t)) * np.exp(-t * 25)
        else:
            v = 0.4 * np.diff(rng.normal(0, 1, len(t) + 1)) * np.exp(-t * 90)
        s = int(round(when * SR))
        if 0 <= s < len(x):
            n = min(len(v), len(x) - s)
            x[s:s + n] += v[:n]
    return x


def _cover(start, seconds, shifts, drift=1.0, seed=0):
    rng = np.random.default_rng(seed)
    song = _groove(start + seconds + 5)
    ref = _render(song, start + seconds + 5, rng)
    cov = []
    for when, drum in song:
        rel = when - start
        if 0 <= rel <= seconds:
            cov.append((rel * drift + shifts.get((round(when, 3), drum), 0.0) + rng.normal(0, 0.003), drum))
    return ref, _render(cov, seconds + 1, rng, kick_hz=48)


def test_finds_injected_mistakes():
    start = block_start(BPM, 21)
    shifts = {(round(start + 10 * BEAT, 3), "kick"): 0.035,
              (round(start + 30.5 * BEAT, 3), "hat"): -0.035,
              (round(start + 45 * BEAT, 3), "snare"): 0.035}
    ref, cover = _cover(start, 40, shifts)
    # The true start is passed in: this groove repeats every bar, so any bar would align equally well.
    t = check_timing(ref, cover, start, BPM)
    found = [m["ref_t"] for m in t["mistakes"]]
    for when, _ in shifts:
        assert any(abs(f - when) < 0.02 for f in found), (when, found)
    assert len(found) <= len(shifts) + 2, found


def test_measures_tempo_drift():
    start = block_start(BPM, 21)
    ref, cover = _cover(start, 40, {}, drift=1.003)
    t = check_timing(ref, cover, start, BPM)
    assert abs(t["tempo"]["cover_bpm"] - BPM / 1.003) < 0.05, t["tempo"]
    assert not t["mistakes"], t["mistakes"]


if __name__ == "__main__":
    test_finds_injected_mistakes()
    test_measures_tempo_drift()
    print("ok")
