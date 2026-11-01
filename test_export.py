"""Checks that export_aligned() writes two mp3s that start together and line up click for click."""
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from aligner.export import export_aligned, shared_span

RATE = 44100


def _write(path, x, rate):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "-",
                    str(path)], input=x.astype(np.float32).tobytes(), check=True)


def _pcm(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def _clicks(times, seconds, rate):
    x = np.zeros(int(seconds * rate), np.float32)
    for t in times:
        x[int(round(t * rate)):int(round(t * rate)) + 40] = 0.8
    return x


def _onsets(x):
    above = np.flatnonzero(np.abs(x) > 0.4)
    return above[np.r_[True, np.diff(above) > RATE // 100]] / RATE


def _check(start, cover_len=6.0, song_len=10.0):
    rng = np.random.default_rng(1)
    song_clicks = np.sort(rng.uniform(0.2, song_len - 0.2, 25))
    cover_clicks = [t - start for t in song_clicks if 0.1 < t - start < cover_len - 0.1]
    with tempfile.TemporaryDirectory() as tmp:
        ref, cover = Path(tmp) / "extracted.wav", Path(tmp) / "cover.mp3"
        _write(ref, _clicks(song_clicks, song_len, RATE), RATE)
        _write(cover, _clicks(cover_clicks, cover_len, 48000), 48000)
        ref_out, cover_out, begin, length = export_aligned(ref, cover, start, Path(tmp) / "r.mp3", Path(tmp) / "c.mp3")
        assert (begin, length) == shared_span(start, cover_len, song_len)
        r, c = _pcm(ref_out), _pcm(cover_out)
        assert abs(len(r) / RATE - length) < 0.03 and abs(len(c) / RATE - length) < 0.03, (len(r), len(c), length)
        a, b = _onsets(r), _onsets(c)
        assert len(a) == len(b) > 3, (a, b)
        assert np.max(np.abs(a - b)) < 0.001, np.abs(a - b)
        expected = [t - begin for t in song_clicks if begin + 0.1 < t < begin + length - 0.1]
        assert all(np.min(np.abs(a - e)) < 0.002 for e in expected)


def test_cover_inside_song():
    _check(start=2.35)


def test_cover_starts_before_song():
    _check(start=-1.2)


def test_cover_runs_past_song_end():
    _check(start=6.5)


if __name__ == "__main__":
    test_cover_inside_song()
    test_cover_starts_before_song()
    test_cover_runs_past_song_end()
    print("ok")
