"""Split the original song into drums and drumless with demucs.

Runs `demucs --two-stems=drums --name htdemucs_ft --shifts 7 --overlap 0.75 --float32` and keeps
the result in out_dir/<model>/<song name>/{drums,no_drums}.wav.

A cover only plays part of the song, so only that part is separated: the stretch where the cover
can be (its search range plus its length) with SNIPPET_MARGIN s more on each side. The stems are
then padded with silence back to the song's full length, so song times stay the same everywhere
else in the app. span.json next to them records which stretch was separated.

Separation takes minutes, so stems that are already there are reused when they are as long as the
song and cover the stretch asked for (stems without span.json, e.g. from running demucs on the
whole song yourself, cover everything). They are matched by song name and length, not file date:
the UI hands over a fresh copy of every upload.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .audio_io import probe_audio, probe_duration
from .config import DEMUCS_MODEL, DEMUCS_OVERLAP, DEMUCS_SHIFTS, SNIPPET_MARGIN

SAME_LENGTH = 0.5  # s, stems this close to the song's length belong to it
STEMS_DIR = Path(__file__).resolve().parent.parent / "separated"  # the project folder, wherever the app runs from


def stem_paths(song, out_dir=STEMS_DIR, model=DEMUCS_MODEL):
    folder = Path(out_dir) / model / Path(song).stem
    return folder / "drums.wav", folder / "no_drums.wav"


def snippet_span(estimate, radius, cover_len, margin=SNIPPET_MARGIN):
    """Song stretch to separate for a cover expected at `estimate` ± `radius` s."""
    return estimate - radius - margin, estimate + radius + cover_len + margin


def _stems_fit(song, *stems):
    if not all(p.exists() for p in stems):
        return False
    try:
        length = probe_duration(song)
        return all(abs(probe_duration(p) - length) <= SAME_LENGTH for p in stems)
    except RuntimeError:  # unreadable, e.g. half written
        return False


def _covers(folder, span):
    info = folder / "span.json"
    if not info.exists():
        return True  # the whole song was separated
    if span is None:
        return False
    done = json.loads(info.read_text())
    return done["from"] <= span[0] + 0.01 and done["to"] >= span[1] - 0.01


def _run_demucs(cmd):
    """Run demucs; its progress goes to the terminal. Returns the exit code."""
    return subprocess.run(cmd).returncode


def _ffmpeg(*args):
    proc = subprocess.run(["ffmpeg", "-y", "-v", "error", *map(str, args)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg failed")


def separate(song, out_dir=STEMS_DIR, span=None, model=DEMUCS_MODEL, shifts=DEMUCS_SHIFTS,
             overlap=DEMUCS_OVERLAP, log=print):
    """Return (drums.wav, no_drums.wav) for `song`, running demucs only when needed.

    `span` is the (from, to) song stretch to separate (see snippet_span); None separates it all.
    """
    drums, drumless = stem_paths(song, out_dir, model)
    length = probe_duration(song)
    if span is not None:
        span = (max(0.0, span[0]), min(length, span[1]))
        if span[1] - span[0] >= length - 1:
            span = None
    if _stems_fit(song, drums, drumless) and _covers(drums.parent, span):
        log(f"0. Using the drums already separated from {Path(song).name} ({drums.parent})")
        return str(drums), str(drumless)

    part = "the whole song" if span is None else \
        f"song {span[0]:.1f}–{span[1]:.1f} s ({span[1] - span[0]:.0f} of {length:.0f} s)"
    log(f"0. Separating drums from {Path(song).name}, {part}, with demucs {model} (shifts {shifts})")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(song)
        if span is not None:  # cut on the exact sample; a plain name keeps demucs' output folder simple
            src = Path(tmp) / "snippet.wav"
            _ffmpeg("-i", song, "-af", f"atrim=start={span[0]:.6f}:end={span[1]:.6f},asetpts=PTS-STARTPTS",
                    "-c:a", "pcm_f32le", src)
        out = Path(tmp) / "out"
        # Run it with this interpreter so the venv's install is used.
        code = _run_demucs([sys.executable, "-m", "demucs", "--two-stems=drums", "--name", model,
                            "--shifts", str(shifts), "--overlap", str(overlap), "--float32",
                            "-o", str(out), str(src)])
        made = [out / model / src.stem / p.name for p in (drums, drumless)]
        if code != 0 or not all(p.exists() for p in made):
            raise RuntimeError(f"demucs failed to separate {song} (exit code {code}); see the terminal for its output")
        drums.parent.mkdir(parents=True, exist_ok=True)
        for stem, dest in zip(made, (drums, drumless)):
            if span is None:
                shutil.move(str(stem), dest)
                continue
            rate, _ = probe_audio(stem)  # pad with silence back onto the song's timeline
            _ffmpeg("-i", stem, "-af", f"adelay=delays={round(span[0] * rate)}S:all=1,"
                                       f"apad=whole_len={round(length * rate)}",
                    "-c:a", "pcm_f32le", dest)
    info = drums.parent / "span.json"
    if span is None:
        info.unlink(missing_ok=True)
    else:
        info.write_text(json.dumps({"from": span[0], "to": span[1]}))
    return str(drums), str(drumless)
