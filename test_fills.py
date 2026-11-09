"""Checks that the fill detector finds each drummer's fills and nothing in a steady groove."""
import numpy as np

from aligner.fills import detect_fills, fill_report
from test_timing import BEAT, BPM, _render

BAR = 4 * BEAT


def _events(bars, fill_bars, offset=0.0):
    """Kick 1 & 3, snare 2 & 4, 8th hats; in fill bars, 16th snares on beats 3-4 instead."""
    ev = []
    for b in range(bars):
        for i in range(8):
            t = b * BAR + i * BEAT / 2
            if b in fill_bars and i >= 4:
                ev += [(b * BAR + 2 * BEAT + j * BEAT / 4, "snare") for j in range(8)] if i == 4 else []
                continue
            ev.append((t, "hat"))
            if i % 4 == 0:
                ev.append((t, "kick"))
            if i % 4 == 2:
                ev.append((t, "snare"))
    return [(t + offset, d) for t, d in ev]


def test_finds_each_drummers_fills():
    rng = np.random.default_rng(0)
    start = 4 * BAR
    song = _events(24, {8})
    ref = _render(song, 24 * BAR + 1, rng)
    take = [(t - start, d) for t, d in _events(24, {12}) if start <= t < start + 16 * BAR]  # your take: bars 5-20
    cover = _render(take, 16 * BAR + 0.5, rng, kick_hz=48)
    f = detect_fills(ref, cover, dict(start=start, bpm=BPM))

    def bars(segs):
        return [(round(a / BAR, 2), round(b / BAR, 2)) for a, b, _ in segs]

    orig, yours = bars(f["original"]["fills"]), bars(f["you"]["fills"])
    # fill: 16th snares from beat 3 (8.5 bars) to the last 16th (8.9375); the start may read a 16th late,
    # since the fill's first note lands where the groove has a kick
    assert len(orig) == 1 and 8.49 <= orig[0][0] <= 8.57 and abs(orig[0][1] - 8.9375) < 0.01, orig
    assert len(yours) == 1 and 12.49 <= yours[0][0] <= 12.57 and abs(yours[0][1] - 12.9375) < 0.01, yours
    report = fill_report(f)
    assert "you keep the groove" in report and "only you fill" in report


if __name__ == "__main__":
    test_finds_each_drummers_fills()
    print("ok")
