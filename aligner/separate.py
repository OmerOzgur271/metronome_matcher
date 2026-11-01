"""Split the original song into drums and drumless with demucs.

Runs `demucs --two-stems=drums --name htdemucs_ft --shifts 7 --overlap 0.75 --float32 song`
and keeps the result in out_dir/<model>/<song name>/{drums,no_drums}.wav. Separation takes
minutes, so a song whose stems are already there (and newer than the song) is not split again.
"""
import subprocess
import sys
from pathlib import Path

from .config import DEMUCS_MODEL, DEMUCS_OVERLAP, DEMUCS_SHIFTS


def stem_paths(song, out_dir="separated", model=DEMUCS_MODEL):
    folder = Path(out_dir) / model / Path(song).stem
    return folder / "drums.wav", folder / "no_drums.wav"


def separate(song, out_dir="separated", model=DEMUCS_MODEL, shifts=DEMUCS_SHIFTS, overlap=DEMUCS_OVERLAP,
             log=print):
    """Return (drums.wav, no_drums.wav) for `song`, running demucs only when needed."""
    drums, drumless = stem_paths(song, out_dir, model)
    if all(p.exists() and p.stat().st_mtime >= Path(song).stat().st_mtime for p in (drums, drumless)):
        log(f"0. Using the drums already separated from {Path(song).name} ({drums.parent})")
        return str(drums), str(drumless)
    log(f"0. Separating drums from {Path(song).name} with demucs {model} (shifts {shifts}); this takes a few minutes")
    # demucs prints its progress to the terminal; run it with this interpreter so the venv's install is used.
    proc = subprocess.run([sys.executable, "-m", "demucs", "--two-stems=drums", "--name", model,
                           "--shifts", str(shifts), "--overlap", str(overlap), "--float32",
                           "-o", str(out_dir), str(song)])
    if proc.returncode != 0 or not (drums.exists() and drumless.exists()):
        raise RuntimeError(f"demucs failed to separate {song} (exit code {proc.returncode}); "
                           f"see the terminal for its output")
    return str(drums), str(drumless)
