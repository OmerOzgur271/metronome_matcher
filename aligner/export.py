"""Crop the cover and the extracted drums to the stretch of song they share and write both as mp3.

With the cover placed at `start` s of the song, the shared stretch is song time
max(start, 0) .. min(start + cover length, song length). Both files are cut to exactly
that stretch, so they begin on the same beat and have the same length: drop them on two
tracks at 0:00 and they play in sync.
"""
import subprocess

from .audio_io import probe_audio, probe_duration


def shared_span(start, cover_len, ref_len):
    """(song time where both begin, length in s) of the overlap; length 0 when they don't overlap."""
    begin = max(start, 0.0)
    return begin, max(0.0, min(start + cover_len, ref_len) - begin)


def _crop_mp3(src, begin, length, rate, out_path):
    # atrim cuts on the exact sample after decoding (input -ss on an mp3 can land a frame off).
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-af", f"aresample={rate},atrim=start={begin:.6f}:duration={length:.6f},asetpts=PTS-STARTPTS",
         "-c:a", "libmp3lame", "-b:a", "320k", str(out_path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ffmpeg failed to crop {src}")
    return str(out_path)


def export_aligned(reference, cover, start, ref_out, cover_out):
    """Write the shared stretch of both tracks as mp3s at the reference's sample rate.

    Returns (ref_out, cover_out, song time where they begin, length in s).
    """
    begin, length = shared_span(start, probe_duration(cover), probe_duration(reference))
    if length <= 0:
        raise RuntimeError(f"The cover placed at {start:.3f} s does not overlap the extracted drums.")
    rate, _ = probe_audio(reference)
    _crop_mp3(reference, begin, length, rate, ref_out)
    _crop_mp3(cover, begin - start, length, rate, cover_out)
    return str(ref_out), str(cover_out), begin, length
