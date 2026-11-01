"""Human-readable outputs: markdown report, alignment plot, stereo preview."""
import numpy as np

from .config import BANDS, FINE, MISTAKE_MS, SR
from .helpers import clock_time, segment
from .problems import problems_report


def report(r):
    T = r["start"]
    lines = [
        f"## Start time: **{T:.3f} s**  ({clock_time(T)})",
        f"**Offset from {r['estimate']:.3f} s:** {r['offset_ms']:+.1f} ms  ",
        f"**Alignment confidence:** {r['confidence']}/100  "
        f"(correlation {r['parts']['correlation']:.2f} · hits {r['parts']['hits']:.2f} · "
        f"consistency {r['parts']['consistency']:.2f} · uniqueness {r['parts']['uniqueness']:.2f})",
        "", "| Band | NCC | Cover hits | Ref hits | Aligned (±25 ms) | Median error |", "|---|---|---|---|---|---|",
    ]
    for b in r["best"]["bands"]:
        lines.append(f"| {b['name']} | {b['ncc']:.2f} | {b['cover']} | {b['ref']} | {b['matched']} | {b['err_ms']:+.1f} ms |")
    segs = ", ".join("silent" if s is None else f"{s:+.1f}" for s in r["best"]["segments"])
    lines += ["", f"**Per-segment drift (ms):** {segs}"]
    if len(r["candidates"]) > 1:
        lines += ["", "**Candidates checked:**"]
        for c in r["candidates"]:
            beats = (c["T"] - r["start"]) / (60 / r["bpm"])
            tag = "**chosen**" if c is r["best"] else f"{beats:+.2f} beats"
            lines.append(f"- {c['T']:.3f} s: NCC {c['fine_ncc']:.3f}, hit agreement {c['hit_score']:.2f} ({tag})")
    return "\n".join(lines)


def _plot_band(ax, t, ref_env, cover_env, bpm, label):
    """Reference envelope up, cover envelope down, with a gray beat grid."""
    ax.fill_between(t[:len(ref_env)], 0, ref_env, color="C0", lw=0, label="extracted_drums")
    ax.fill_between(t[:len(cover_env)], 0, -cover_env, color="C1", lw=0, label="drumcover")
    beat = 60 / bpm
    for g in np.arange(np.ceil(t[0] / beat), t[-1] / beat) * beat:
        ax.axvline(g, color="gray", lw=0.5, alpha=0.5)
    ax.set_ylabel(label)
    ax.set_yticks([])


def plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best, (times, curve) = r["best"], r["curve"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), gridspec_kw=dict(height_ratios=[1.3, 1, 1, 1]))
    ax = axes[0]
    ax.plot(times, curve, lw=1.2)
    ax.axvline(r["estimate"], color="gray", ls="--", label=f"estimate {r['estimate']:.3f}")
    ax.axvline(r["start"], color="C3", label=f"found {r['start']:.3f}")
    ax.set_title("Correlation vs start time")
    ax.legend(loc="lower right")

    # 4 s of aligned envelopes from the first busy spot: reference up, cover down.
    hop = FINE[0] / SR
    cover_f, ref_f = r["cover_f"][:, :best["mf"]], best["ref_f"]
    i0 = int(np.argmax(cover_f.sum(0) > 1.0))
    n = int(4 / hop)
    t = r["start"] + (i0 + np.arange(n)) * hop
    j0 = int(round((t[0] - best["ref_start"]) / hop))
    for b, ax in enumerate(axes[1:]):
        c = cover_f[b, i0:i0 + n]
        _plot_band(ax, t, ref_f[b, j0:j0 + len(c)], c, r["bpm"], BANDS[b][0])
    axes[1].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Song time (s), beat grid in gray")
    fig.tight_layout()
    return fig


def overlay(ref, cover, start, seconds=20):
    """Stereo preview: reference left, cover right, starting at the found offset."""
    s, n = round(start * SR), min(len(cover), seconds * SR)
    st = np.stack([segment(ref, s, s + n), cover[:n]], 1)
    st /= np.abs(st).max(0).clip(1e-9)  # normalise each channel
    return SR, (st * 0.9 * 32767).astype(np.int16)


def _ms(s):
    return f"{1000 * s:+.0f} ms"


def timing_report(t, max_rows=15):
    matched = [h for h in t["hits"] if h["err"] is not None]
    if not matched:
        return "## Timing check\nNo hits could be paired with the original."
    errs = np.abs([h["err"] for h in matched]) * 1000
    per_band = " · ".join(f"{name} {sum(h['band'] == b for h in matched)}" for b, (name, _, _) in enumerate(BANDS))
    lines = [
        "## Timing check (whole cover vs original)",
        f"**{len(matched)} hits compared** ({per_band}) · median error {np.median(errs):.1f} ms · "
        f"{np.mean(errs <= 10) * 100:.0f}% within ±10 ms · {np.mean(errs <= MISTAKE_MS) * 100:.0f}% within ±{MISTAKE_MS} ms  ",
        "Positive = the cover hits **late**, negative = **early**.",
    ]
    tempo = t["tempo"]
    if tempo:
        pct = -100 * tempo["slope"]
        lines.append(
            f"**Tempo:** drift goes from {tempo['start_ms']:+.0f} ms at the start to {tempo['end_ms']:+.0f} ms at the end "
            f"→ cover played at ≈ **{tempo['cover_bpm']:.2f} BPM** vs {t['bpm']:g} ({abs(pct):.2f}% {'slower' if pct < 0 else 'faster'})")

    lines += ["", problems_report(t)]

    mistakes = sorted(t["mistakes"], key=lambda m: -abs(m["local"]))
    lines += ["", f"### Single wrong hits: {len(mistakes)} strokes more than {MISTAKE_MS} ms off the hits around them"]
    if mistakes:
        lines += ["", "| Bar / beat | Song time | Heard in | Off from original | Off from surrounding hits |",
                  "|---|---|---|---|---|"]
        for m in sorted(mistakes[:max_rows], key=lambda m: m["song_t"]):
            lines.append(f"| {m['bar']} / {m['beat']:g} | {m['ref_t']:.3f} s | {' + '.join(m['bands'])} | "
                         f"{_ms(m['err'])} | **{_ms(m['local'])} {'late' if m['local'] > 0 else 'early'}** |")
        if len(mistakes) > max_rows:
            lines.append(f"\n…and {len(mistakes) - max_rows} more (the worst {max_rows} are shown).")

    differs = [b["bar"] for b in t["bars"] if b["pattern_differs"]]
    lines += ["", "**Bars where the cover plays a different pattern (usually fills; not counted as mistakes):** " +
              (", ".join(map(str, differs)) if differs else "none")]
    return "\n".join(lines)
