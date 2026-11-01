"""Checks that problem clips put the click on the original's beat grid and your take at the right time."""
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from aligner.clips import RATE, _grid_phase, click_track, problem_clips

BPM = 120.0
BEAT = 60 / BPM


def _onsets(x, thresh):
    above = np.flatnonzero(np.abs(x) > thresh)
    return above[np.r_[True, np.diff(above) > RATE // 50]] / RATE


def test_grid_phase_follows_the_original():
    hits = np.arange(40) * 15 / BPM + 0.012
    assert abs(_grid_phase(hits, BPM) - 0.012) < 1e-6
    assert abs(_grid_phase(hits - 0.024, BPM) + 0.012) < 1e-6


def test_click_track_on_beats_with_accent_on_one():
    x = click_track(10.0, int(4 * RATE), BPM, 0.01)  # song 10..14 s; beat k at k * 0.5 + 0.01
    on = _onsets(x, 0.05)
    assert np.allclose(on, np.arange(8) * BEAT + 0.01, atol=1e-3), on
    peaks = [np.max(np.abs(x[int(t * RATE):int(t * RATE) + 500])) for t in on]
    assert peaks[0] > 0.9 and peaks[4] > 0.9 and max(peaks[1:4]) < 0.7  # 10 s and 12 s are bar downbeats


def test_clip_lines_up_take_and_click():
    start, phase, hit_song = 3.0, 0.008, 9.27   # take starts at song 3 s, one loud hit at song 9.27 s
    take = np.zeros(int(12 * RATE), np.float32)
    i = int((hit_song - start) * RATE)
    take[i:i + 200] = 0.9 * np.sin(2 * np.pi * 5000 * np.arange(200) / RATE)
    t = dict(bpm=BPM, start=start, hits=[dict(err=0.0, ref_t=k * BEAT / 4 + phase) for k in range(200)])
    with tempfile.TemporaryDirectory() as tmp:
        cover = Path(tmp) / "cover.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(RATE), "-ac", "1", "-i", "-",
                        str(cover)], input=take.tobytes(), check=True)
        [path] = problem_clips(cover, cover, t, [dict(first=6, last=6)], tmp, click_level=0.5)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
    y = np.frombuffer(raw, np.float32)
    song_from = 4 * 2.0  # bar 6 starts at 10 s; one bar (2 s) before it
    assert abs(len(y) / RATE - 6.0) < 0.06, len(y) / RATE  # bars 5-7
    on = _onsets(y, 0.2)
    beats = np.arange(12) * BEAT + phase
    hit = hit_song - song_from
    assert any(abs(on - hit) < 2e-3), (on, hit)
    assert all(np.min(np.abs(on - b)) < 2e-3 for b in beats), on


if __name__ == "__main__":
    test_grid_phase_follows_the_original()
    test_click_track_on_beats_with_accent_on_one()
    test_clip_lines_up_take_and_click()
    print("ok")
