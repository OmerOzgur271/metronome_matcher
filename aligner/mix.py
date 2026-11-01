"""Mix the drum cover onto the drumless track with ffmpeg."""
import subprocess

from .audio_io import probe_audio, probe_duration


def _placement_filter(start: float, rate: int) -> str:
    """Filter prefix that shifts the cover to `start` seconds (trims when negative)."""
    if start < 0:
        return f"atrim=start={-float(start):.6f},asetpts=PTS-STARTPTS,"
    samples = int(round(float(start) * rate))
    return f"adelay=delays={samples}S:all=1," if samples else ""


def keep_span(start, cover_len, song_len, pad):
    """Song time (from, to) the mix keeps: `pad` s before and after the cover, inside the song."""
    return max(0.0, max(start, 0.0) - pad), min(song_len, start + cover_len + pad)


def mix(drumless: str, cover: str, start: float, out_path: str, cover_gain: float = 1.0, pad=None) -> str:
    """Place drumcover on the drumless timeline at `start` seconds and write a wav.

    `start` is the song time where the cover's first sample belongs. A negative
    start trims the cover instead of delaying it. The drumless track keeps its
    sample rate and is mixed at unity; `cover_gain` scales only the drums.
    With `pad` set, only `pad` s before and after the cover are kept, not the whole song.
    """
    rate, channels = probe_audio(drumless)
    layout = "mono" if channels == 1 else "stereo"
    place = _placement_filter(start, rate)
    keep = ""
    if pad is not None:
        lo, hi = keep_span(start, probe_duration(cover), probe_duration(drumless), pad)
        keep = f",atrim=start={lo:.6f}:end={hi:.6f},asetpts=PTS-STARTPTS"
    filt = (
        f"[0:a]aformat=sample_fmts=fltp:sample_rates={rate}:channel_layouts={layout}[bed];"
        f"[1:a]aformat=sample_fmts=fltp:sample_rates={rate}:channel_layouts={layout},"
        f"{place}volume={float(cover_gain):.6f}[cover];"
        f"[bed][cover]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0{keep}[out]"
    )
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(drumless), "-i", str(cover),
         "-filter_complex", filt, "-map", "[out]", "-c:a", "pcm_s16le", str(out_path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg failed to mix the tracks")
    return str(out_path)
