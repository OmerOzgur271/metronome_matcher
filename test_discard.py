"""Checks that discard_unplayed() removes only the original hits you don't play, in their own band."""
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from aligner.audio_io import decode
from aligner.discard import discard_unplayed, mute_windows

RATE = 44100


def _write(path, x):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(RATE), "-ac", "1", "-i", "-", str(path)],
                   input=np.asarray(x, np.float32).tobytes(), check=True)


def _hit(x, at, freq, decay):
    t = np.arange(int(0.3 * RATE)) / RATE
    i = int(at * RATE)
    x[i:i + len(t)] += 0.5 * np.sin(2 * np.pi * freq * t) * np.exp(-t * decay)


def _row(band, status, start, end):
    return dict(band=band, status=status, orig=dict(start=start, end=end))


def test_removes_only_the_unplayed_snare():
    n = int(4 * RATE)
    t = np.arange(n) / RATE
    other = 0.1 * np.sin(2 * np.pi * 440 * t)          # the rest of the band, never touched
    drums = np.zeros(n)
    _hit(drums, 1.0, 60, 15)                             # kick you play
    _hit(drums, 2.0, 1000, 30)                           # snare only the original plays
    _hit(drums, 3.0, 60, 15)                             # kick you play
    attacks = [_row("Kick", "matched", 1.0, 1.05), _row("Snare", "missed", 2.0, 2.05), _row("Kick", "matched", 3.0, 3.05)]
    with tempfile.TemporaryDirectory() as tmp:
        song, stem, out = Path(tmp) / "song.wav", Path(tmp) / "drums.wav", Path(tmp) / "out.wav"
        _write(song, other + drums)
        _write(stem, drums)
        _, removed = discard_unplayed(song, stem, attacks, out)
        y = decode(out, RATE)
    assert removed == 1
    snare = slice(int(2.0 * RATE), int(2.25 * RATE))
    assert np.max(np.abs(y[snare] - other[snare])) < 0.02          # snare gone, the 440 Hz part left
    before = slice(0, int(1.98 * RATE))
    assert np.max(np.abs(y[before] - (other + drums)[before])) < 1e-4  # untouched before the cut
    kick2 = slice(int(3.0 * RATE), int(3.2 * RATE))
    assert np.max(np.abs(y[kick2] - (other + drums)[kick2])) < 1e-4    # the kick you play stays


def test_cut_stops_at_the_drums_next_hit():
    attacks = [_row("Cymbal", "missed", 1.0, 1.1), _row("Cymbal", "matched", 1.4, 1.5), _row("Kick", "missed", 5.0, 5.1)]
    w = mute_windows(attacks)
    attack = [x for x in w if x[1] < 1.0 < x[2] and abs(x[2] - 1.1) < 1e-9]
    assert sorted(x[0] for x in attack) == [0, 1, 2]                 # the attack: every band
    tail = [x for x in w if x[0] == 2 and x[1] < 1.0 and x[2] > 1.1]
    assert len(tail) == 1 and abs(tail[0][2] - 1.395) < 1e-9          # crash ring cut until the next cymbal hit
    kick_tail = [x for x in w if x[0] == 0 and x[1] > 4]
    assert max(x[2] for x in kick_tail) == 5.5                        # no next kick: the kick's longest tail

if __name__ == "__main__":
    test_removes_only_the_unplayed_snare()
    test_cut_stops_at_the_drums_next_hit()
    print("ok")
