"""Command-line entry point; launches the UI when no files are given."""
import argparse
from pathlib import Path

from .align import align
from .attacks import analyse_attacks, attack_plot, attack_report
from .helpers import block_start
from .clips import problem_clips
from .discard import discard_unplayed
from .fills import detect_fills, fill_plot, fill_report
from .audio_io import load, probe_duration
from .config import MIX_PAD, SEARCH_RADIUS, SNIPPET_MARGIN
from .export import export_aligned
from .mix import mix
from .problems import problem_ranges
from .report import report, timing_report
from .separate import STEMS_DIR, separate, snippet_span
from .timing import check_timing


def mix_paths(out_dir, song):
    """Where the two final mixes go: <out_dir>/<song name>_cover_on_{original,drumless}.wav."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(song).stem
    return str(out / f"{name}_cover_on_original.wav"), str(out / f"{name}_cover_on_drumless.wav")


def build_parser(description=None):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cover", nargs="?", help="your drum cover, e.g. drumcover.mp3")
    p.add_argument("song", nargs="?", help="the original song, e.g. gg.mp3; demucs splits it into drums + drumless")
    p.add_argument("--bpm", type=float, default=100)
    p.add_argument("--block", type=int, default=21)
    p.add_argument("--buffer", type=float, default=1.0)
    p.add_argument("--radius", type=float, default=SEARCH_RADIUS)
    p.add_argument("--match", type=float, default=30.0)
    p.add_argument("--stems-dir", default=str(STEMS_DIR),
                   help="where demucs keeps the separated drums and drumless (default: separated/ in the project)")
    p.add_argument("--out-dir", default="output", help="where the two final mixes go")
    p.add_argument("--gain", type=float, default=1.0, help="drum cover level in the mixes")
    p.add_argument("--pad", type=float, default=MIX_PAD, help="seconds of song the mixes keep before and after the cover")
    p.add_argument("--keep-unplayed", action="store_true",
                   help="keep the original drum hits you don't play in the 'cover + original song' mix "
                        "(by default they are taken out)")
    p.add_argument("--crop", nargs=2, metavar=("EXTRACTED_MP3", "COVER_MP3"),
                   help="write the separated drums and the cover cropped to the stretch they share, aligned, as mp3s")
    p.add_argument("--attack-chart", help="save the attack analysis chart to this PNG")
    p.add_argument("--fill-chart", help="save the fill timeline (original and yours) to this PNG")
    p.add_argument("--clips", metavar="DIR", help="write each problem range as an mp3 with a click at --bpm here")
    p.add_argument("--with-original", action="store_true", help="also play the original drums quietly in the clips")
    return p


def main(argv=None, description=None):
    a = build_parser(description).parse_args(argv)
    if not (a.cover and a.song):
        from .ui import ui
        ui()
        return
    span = snippet_span(block_start(a.bpm, a.block), a.radius, probe_duration(a.cover), margin=max(SNIPPET_MARGIN, a.pad))
    drums, drumless = separate(a.song, a.stems_dir, span=span)
    ref, cover = load(drums), load(a.cover)
    res = align(ref, cover, a.bpm, a.block, a.buffer, a.radius, a.match)
    print()
    print(report(res))
    t = check_timing(ref, cover, res["start"], a.bpm)
    print()
    print(timing_report(t))
    attacks = analyse_attacks(ref, cover, t)
    print()
    print(attack_report(attacks, t))
    fills = detect_fills(ref, cover, t)
    print()
    print(fill_report(fills))
    if a.fill_chart:
        fill_plot(fills).savefig(a.fill_chart, dpi=120)
        print(f"\nFill timeline written to {a.fill_chart}")
    if a.attack_chart:
        attack_plot(attacks, t).savefig(a.attack_chart, dpi=120)
        print(f"\nAttack chart written to {a.attack_chart}")
    if a.crop:
        _, _, begin, length = export_aligned(drums, a.cover, res["start"], *a.crop)
        print(f"\nCropped to song {begin:.3f}-{begin + length:.3f} s ({length:.1f} s): {a.crop[0]}, {a.crop[1]}")
    if a.clips:
        paths = problem_clips(a.cover, drums, t, problem_ranges(t), a.clips,
                              original_level=0.5 if a.with_original else 0.0)
        print(f"\n{len(paths)} problem clips with a click written to {a.clips}" + "".join(f"\n  {p}" for p in paths))
    on_original, on_drumless = mix_paths(a.out_dir, a.song)
    bed = a.song
    if not a.keep_unplayed:
        bed, removed = discard_unplayed(a.song, drums, attacks, str(Path(a.out_dir) / f"{Path(a.song).stem}_without_unplayed.wav"))
        print(f"\nTook {removed} original drum hits you don't play out of the original song: {bed}")
    mix(bed, a.cover, res["start"], on_original, a.gain, a.pad)
    mix(drumless, a.cover, res["start"], on_drumless, a.gain, a.pad)
    print(f"\nMixes written:\n  cover + original song: {on_original}\n  cover + drumless:      {on_drumless}")
