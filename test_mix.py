"""Checks that mix() places the drum cover on the drumless timeline."""
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from aligner.mix import mix


def _tone(path, spec, seconds):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", spec, "-t", str(seconds), str(path)],
        check=True)


def _pcm(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "44100", "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def test_cover_starts_at_offset():
    with tempfile.TemporaryDirectory() as tmp:
        bed, cover, out = Path(tmp) / "drumless.wav", Path(tmp) / "cover.wav", Path(tmp) / "final.wav"
        _tone(bed, "anullsrc=r=44100:cl=stereo", 2)
        _tone(cover, "sine=frequency=880:sample_rate=48000", 0.3)
        mix(bed, cover, 0.75, out, 1.0)
        y = _pcm(out)
        before = int(0.70 * 44100)
        after = int(0.80 * 44100)
        assert np.max(np.abs(y[:before])) < 0.01
        assert np.max(np.abs(y[after:after + 8000])) > 0.05


def test_negative_start_trims_cover():
    with tempfile.TemporaryDirectory() as tmp:
        bed, cover, out = Path(tmp) / "drumless.wav", Path(tmp) / "cover.wav", Path(tmp) / "final.wav"
        _tone(bed, "anullsrc=r=44100:cl=stereo", 1)
        _tone(cover, "sine=frequency=880:sample_rate=44100", 0.5)
        mix(bed, cover, -0.2, out, 1.0)
        y = _pcm(out)
        assert np.max(np.abs(y[:2000])) > 0.05
        assert np.max(np.abs(y[int(0.4 * 44100):])) < 0.01


def test_zero_gain_leaves_the_bed():
    with tempfile.TemporaryDirectory() as tmp:
        bed, cover, out = Path(tmp) / "drumless.wav", Path(tmp) / "cover.wav", Path(tmp) / "final.wav"
        _tone(bed, "sine=frequency=220:sample_rate=44100", 0.4)
        _tone(cover, "sine=frequency=880:sample_rate=44100", 0.4)
        mix(bed, cover, 0.0, out, 0.0)
        y, bed_y = _pcm(out), _pcm(bed)
        n = min(len(y), len(bed_y))
        assert np.max(np.abs(y[:n] - bed_y[:n])) < 0.02


def test_pad_keeps_only_around_cover():
    with tempfile.TemporaryDirectory() as tmp:
        bed, cover, out = Path(tmp) / "drumless.wav", Path(tmp) / "cover.wav", Path(tmp) / "final.wav"
        _tone(bed, "anullsrc=r=44100:cl=stereo", 40)
        _tone(cover, "sine=frequency=880:sample_rate=44100", 3)
        mix(bed, cover, 15.0, out, 1.0, pad=10)  # keeps song 5..28 s, cover at 10 s
        y = _pcm(out)
        assert abs(len(y) / 44100 - 23) < 0.01, len(y) / 44100
        assert np.max(np.abs(y[:int(9.9 * 44100)])) < 0.01
        assert np.max(np.abs(y[int(10.1 * 44100):int(12.9 * 44100)])) > 0.05
        assert np.max(np.abs(y[int(13.1 * 44100):])) < 0.01


def test_pad_stops_at_song_edges():
    with tempfile.TemporaryDirectory() as tmp:
        bed, cover, out = Path(tmp) / "drumless.wav", Path(tmp) / "cover.wav", Path(tmp) / "final.wav"
        _tone(bed, "anullsrc=r=44100:cl=stereo", 10)
        _tone(cover, "sine=frequency=880:sample_rate=44100", 3)
        mix(bed, cover, 4.0, out, 1.0, pad=10)  # can't go before 0 s or past the song's 10 s
        y = _pcm(out)
        assert abs(len(y) / 44100 - 10) < 0.01, len(y) / 44100
        assert np.max(np.abs(y[int(4.1 * 44100):int(6.9 * 44100)])) > 0.05


if __name__ == "__main__":
    test_cover_starts_at_offset()
    test_negative_start_trims_cover()
    test_zero_gain_leaves_the_bed()
    test_pad_keeps_only_around_cover()
    test_pad_stops_at_song_edges()
    print("ok")
