"""ffmpeg/ffprobe wrappers for decoding and inspecting audio files."""
import subprocess

import numpy as np

from .config import SR


def decode(path, rate=SR, channels=1):
    """Decode any audio file to float32 via ffmpeg: shape (n,) for mono, (n, channels) otherwise."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", str(channels), "-ar", str(rate), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    return x if channels == 1 else x.reshape(-1, channels)


def load(path):
    """Decode any audio file to mono float32 at SR via ffmpeg."""
    return decode(path)


def write_mp3(x, rate, path, bitrate="192k"):
    """Encode float32 samples, shape (n,) or (n, channels), to an mp3."""
    channels = 1 if x.ndim == 1 else x.shape[1]
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(rate), "-ac", str(channels), "-i", "-",
         "-c:a", "libmp3lame", "-b:a", bitrate, str(path)],
        input=np.ascontiguousarray(x, np.float32).tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode().strip() or f"ffmpeg failed to write {path}")
    return str(path)


def probe_audio(path: str) -> tuple[int, int]:
    """Return (sample_rate, channels) for the first audio stream."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        detail = proc.stderr.strip() or "no audio stream"
        raise RuntimeError(f"Could not read {path}: {detail}")
    rate, channels = proc.stdout.strip().split(",")[:2]
    return int(rate), int(channels)


def probe_duration(path: str) -> float:
    """Length of the file in seconds."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    try:
        return float(proc.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Could not read the length of {path}: {proc.stderr.strip() or 'no duration'}") from None


def probe_start(path: str) -> float:
    """Start time of the first audio stream: an mp3's encoder delay, which ffmpeg skips when decoding."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=start_time",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return max(0.0, float(proc.stdout.strip()))
    except ValueError:
        return 0.0


def write_wav(x, rate, path):
    """Write float32 samples, shape (n,) or (n, channels), as a 32-bit float wav."""
    channels = 1 if x.ndim == 1 else x.shape[1]
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(rate), "-ac", str(channels), "-i", "-",
         "-c:a", "pcm_f32le", str(path)],
        input=np.ascontiguousarray(x, np.float32).tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode().strip() or f"ffmpeg failed to write {path}")
    return str(path)
