"""Command-line entry point; launches the UI when no files are given."""
import argparse

from .align import align
from .clips import problem_clips
from .audio_io import load
from .config import MIX_PAD, SEARCH_RADIUS
from .export import export_aligned
from .mix import mix
from .problems import problem_ranges, problems_plot
from .report import report, timing_report
from .timing import check_timing


def build_parser(description=None):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cover", nargs="?", help="drumcover.mp3")
    p.add_argument("reference", nargs="?", help="extracted_drums.wav")
    p.add_argument("--bpm", type=float, default=100)
    p.add_argument("--block", type=int, default=21)
    p.add_argument("--buffer", type=float, default=1.0)
    p.add_argument("--radius", type=float, default=SEARCH_RADIUS)
    p.add_argument("--match", type=float, default=30.0)
    p.add_argument("--drumless", help="drumless.wav; mix the cover onto it at the found start")
    p.add_argument("-o", "--out", default="final.wav", help="output wav when --drumless is set")
    p.add_argument("--gain", type=float, default=1.0, help="drum cover level in the mix")
    p.add_argument("--pad", type=float, default=MIX_PAD, help="seconds of song the mix keeps before and after the cover")
    p.add_argument("--crop", nargs=2, metavar=("EXTRACTED_MP3", "COVER_MP3"),
                   help="write both tracks cropped to the stretch they share, aligned, as these two mp3s")
    p.add_argument("--chart", help="save the error-rate chart (worst time ranges) to this PNG")
    p.add_argument("--clips", metavar="DIR", help="write each problem range as an mp3 with a click at --bpm here")
    p.add_argument("--with-original", action="store_true", help="also play the original drums quietly in the clips")
    return p


def main(argv=None, description=None):
    a = build_parser(description).parse_args(argv)
    if not (a.cover and a.reference):
        from .ui import ui
        ui()
        return
    ref, cover = load(a.reference), load(a.cover)
    res = align(ref, cover, a.bpm, a.block, a.buffer, a.radius, a.match)
    print()
    print(report(res))
    t = check_timing(ref, cover, res["start"], a.bpm)
    print()
    print(timing_report(t))
    if a.crop:
        _, _, begin, length = export_aligned(a.reference, a.cover, res["start"], *a.crop)
        print(f"\nCropped to song {begin:.3f}-{begin + length:.3f} s ({length:.1f} s): {a.crop[0]}, {a.crop[1]}")
    if a.chart:
        problems_plot(t).savefig(a.chart, dpi=120)
        print(f"\nError-rate chart written to {a.chart}")
    if a.clips:
        paths = problem_clips(a.cover, a.reference, t, problem_ranges(t), a.clips,
                              original_level=0.5 if a.with_original else 0.0)
        print(f"\n{len(paths)} problem clips with a click written to {a.clips}" + "".join(f"\n  {p}" for p in paths))
    if a.drumless:
        mix(a.drumless, a.cover, res["start"], a.out, a.gain, a.pad)
        print(f"\nFinal mix written to {a.out}")
