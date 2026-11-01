"""Checks that separate() reuses stems demucs already made and reruns it when the song is newer."""
import importlib
import os
import tempfile
import time
from pathlib import Path

sep = importlib.import_module("aligner.separate")  # the package also exports the function `separate`


def test_reuses_existing_stems_and_reruns_for_a_newer_song():
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        drums, drumless = sep.stem_paths(cmd[-1], cmd[cmd.index("-o") + 1])
        for p in (drums, drumless):
            p.write_bytes(b"new")
        return type("Done", (), {"returncode": 0})()

    real_run = sep.subprocess.run
    sep.subprocess.run = fake_run
    try:
        with tempfile.TemporaryDirectory() as tmp:
            song = Path(tmp) / "gg.mp3"
            song.write_bytes(b"song")
            drums, drumless = sep.stem_paths(song, Path(tmp) / "separated")
            drums.parent.mkdir(parents=True)
            drums.write_bytes(b"old")
            drumless.write_bytes(b"old")
            assert sep.separate(song, Path(tmp) / "separated", log=lambda *_: None) == (str(drums), str(drumless))
            assert not calls

            later = time.time() + 10
            os.utime(song, (later, later))
            sep.separate(song, Path(tmp) / "separated", log=lambda *_: None)
            assert len(calls) == 1 and "--two-stems=drums" in calls[0] and "htdemucs_ft" in calls[0]
            assert drums.read_bytes() == b"new"
    finally:
        sep.subprocess.run = real_run


if __name__ == "__main__":
    test_reuses_existing_stems_and_reruns_for_a_newer_song()
    print("ok")
