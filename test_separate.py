"""Checks that separate() reuses stems that fit, separates only the stretch asked for, and puts it
back at the right song time."""
import importlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from aligner.audio_io import decode

sep = importlib.import_module("aligner.separate")  # the package also exports the function `separate`
RATE = 44100


def _write(path, x):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(RATE), "-ac", "1", "-i", "-", str(path)],
                   input=np.asarray(x, np.float32).tobytes(), check=True)


def _silence(path, seconds):
    _write(path, np.zeros(int(seconds * RATE)))


def _separate_with_fake_demucs(song, out_dir, span=None):
    """separate() with a stand-in for demucs that returns its input as both stems."""
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        src, out, model = Path(cmd[-1]), Path(cmd[cmd.index("-o") + 1]), cmd[cmd.index("--name") + 1]
        folder = out / model / src.stem
        folder.mkdir(parents=True)
        for name in ("drums.wav", "no_drums.wav"):
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), str(folder / name)], check=True)
        return 0

    real_run = sep._run_demucs
    sep._run_demucs = fake_run
    try:
        sep.separate(song, out_dir, span=span, log=lambda *_: None)
    finally:
        sep._run_demucs = real_run
    return calls


def test_reuses_stems_even_for_a_fresh_copy_of_the_song():
    with tempfile.TemporaryDirectory() as tmp:
        song = Path(tmp) / "upload" / "Çilekeş - Y.O.K.mp3"
        song.parent.mkdir()
        drums, drumless = sep.stem_paths(song, Path(tmp) / "separated")
        drums.parent.mkdir(parents=True)
        _silence(drums, 3)
        _silence(drumless, 3)
        _silence(song, 3)  # made after the stems, like a UI upload
        later = time.time() + 60
        os.utime(song, (later, later))
        assert not _separate_with_fake_demucs(song, Path(tmp) / "separated", span=(1, 2))


def test_remakes_stems_that_do_not_fit_the_song():
    with tempfile.TemporaryDirectory() as tmp:
        song = Path(tmp) / "gg.wav"
        _silence(song, 3)
        drums, drumless = sep.stem_paths(song, Path(tmp) / "separated")
        drums.parent.mkdir(parents=True)
        _silence(drums, 1.2)  # cut short, e.g. an interrupted run
        _silence(drumless, 3)
        calls = _separate_with_fake_demucs(song, Path(tmp) / "separated")
        assert len(calls) == 1 and "--two-stems=drums" in calls[0] and "htdemucs_ft" in calls[0]
        assert not _separate_with_fake_demucs(song, Path(tmp) / "separated")  # now they fit


def test_separates_only_the_span_and_keeps_song_time():
    with tempfile.TemporaryDirectory() as tmp:
        song = Path(tmp) / "gg.wav"
        x = 0.01 * np.ones(int(20 * RATE), np.float32)  # quiet everywhere ...
        x[int(12 * RATE):int(12 * RATE) + 100] = 0.9    # ... with a loud hit at 12 s
        _write(song, x)
        out = Path(tmp) / "separated"
        assert len(_separate_with_fake_demucs(song, out, span=(8, 15))) == 1
        drums, _ = sep.stem_paths(song, out)
        y = decode(drums, RATE)
        assert abs(len(y) - len(x)) <= 2, (len(y), len(x))            # full song length
        assert abs(np.argmax(np.abs(y)) - 12 * RATE) <= 1                # the hit stays at 12 s
        assert np.max(np.abs(y[:int(7.99 * RATE)])) == 0                 # silence outside the span
        assert np.max(np.abs(y[int(15.01 * RATE):])) == 0
        assert np.max(np.abs(y[int(8.01 * RATE):int(14.99 * RATE)])) > 0.005
        assert not _separate_with_fake_demucs(song, out, span=(9, 14))   # inside: reused
        assert len(_separate_with_fake_demucs(song, out, span=(2, 9))) == 1  # outside: separated again


def test_stems_demucs_made_from_an_mp3_are_moved_onto_the_song_time():
    from aligner.audio_io import probe_start
    with tempfile.TemporaryDirectory() as tmp:
        x = np.zeros(int(6 * RATE), np.float32)
        x[int(2 * RATE):int(2 * RATE) + 200] = 0.9            # a hit at 2 s
        song = Path(tmp) / "gg.mp3"
        _write(song, x)
        delay = probe_start(song)
        assert delay > 0.01, delay                             # mp3s start with encoder delay
        drums, drumless = sep.stem_paths(song, Path(tmp) / "separated")
        drums.parent.mkdir(parents=True)
        shifted = np.r_[np.zeros(round(delay * RATE), np.float32), x]  # how demucs leaves it
        _write(drums, shifted)
        _write(drumless, shifted)
        assert not _separate_with_fake_demucs(song, Path(tmp) / "separated", span=(1, 3))
        y = decode(drums, RATE)
        song_y = decode(song, RATE)
        assert abs(np.argmax(np.abs(y)) - np.argmax(np.abs(song_y))) <= 2  # same sample as the song


if __name__ == "__main__":
    test_reuses_stems_even_for_a_fresh_copy_of_the_song()
    test_remakes_stems_that_do_not_fit_the_song()
    test_separates_only_the_span_and_keeps_song_time()
    test_stems_demucs_made_from_an_mp3_are_moved_onto_the_song_time()
    print("ok")
