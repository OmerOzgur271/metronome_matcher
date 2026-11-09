"""Fill detector: where the original and your take leave their groove, shown on a timeline.

A groove repeats every bar, a fill doesn't. Both tracks' onsets are put on the original's 16th
grid (in song time, your take placed at the found start) as a strength per band and 16th. For each
16th, the beat around it is compared with the same beat 1 and 2 bars earlier and later; the
closest of those is its fill score (0 = an exact repeat, 1 = nothing alike). A groove always has
some neighbour that matches, a fill has none. Runs of high score become fill segments. This needs
no bar lines: only the tempo and the 16th grid, which the original's own hits give. Each fill's
edges are then set by its hits: a fill hit has no hit of the same drum 1 or 2 bars before or after
it, and the fill runs from the first such hit to the last (searched a beat beyond the score's run,
which crosses the threshold a little late), so its start and end are exact times.
Your fills and the original's are paired by overlap, to compare where each starts and ends.
"""
import numpy as np

from .clips import _grid_phase
from .config import BANDS, FINE, SR, WEAK_HIT
from .features import local_onsets, long_envelopes

THRESHOLD = 0.5   # fill score above this is a fill
SMOOTH = 4        # 16ths of smoothing on the score (one beat)
MIN_LEN = 3       # 16ths; shorter runs are dropped
MERGE_GAP = 2     # 16ths; runs closer than this are one fill
MIN_ACTIVITY = 0.6  # a beat with less summed hit strength than this (a gap, a stop) is not judged
REPEAT_TOL = 0.04   # s; a hit this close to one a bar or two away is groove, not fill


def _onsets(x, start, n, t0):
    """[(times, strengths)] per band for x[start:start + n]; times shifted by t0."""
    hop, win = FINE
    out = []
    for e in long_envelopes(x, start, n, hop, win):
        times, strength = local_onsets(e, hop / SR, start / SR, min_gap=0.06)
        keep = strength >= WEAK_HIT
        out.append((times[keep] + t0, np.minimum(strength[keep], 3.0)))
    return out


def _grid(onsets, phase, six, k0, n):
    """Strength per band and 16th: E[band, k] for 16ths k0 .. k0 + n."""
    e = np.zeros((len(BANDS), n))
    for b, (times, strength) in enumerate(onsets):
        k = np.round((times - phase) / six).astype(int) - k0
        ok = (k >= 0) & (k < n)
        np.maximum.at(e[b], k[ok], strength[ok])
    return e


def fill_score(e):
    """Per 16th: how unlike the same beat 1 and 2 bars away the beat around it is (closest match)."""
    n = e.shape[1]
    score = np.full(n, np.nan)
    for k in range(2, n - 2):
        a = e[:, k - 2:k + 2]
        if a.sum() < MIN_ACTIVITY:
            continue
        best = None
        for lag in (-32, -16, 16, 32):
            j = k + lag
            if 2 <= j < n - 2:
                b = e[:, j - 2:j + 2]
                d = np.abs(a - b).sum() / (a.sum() + b.sum())
                best = d if best is None else min(best, d)
        if best is not None:
            score[k] = best
    filled = np.where(np.isnan(score), 0.0, score)
    weight = np.convolve(~np.isnan(score), np.ones(SMOOTH), "same")
    return np.where(weight > 0, np.convolve(filled, np.ones(SMOOTH), "same") / np.maximum(weight, 1), np.nan)


def _segments(score, phase, six, k0):
    """(from, to) song s of runs where the score is over THRESHOLD."""
    on = np.nan_to_num(score) > THRESHOLD
    runs, k = [], 0
    while k < len(on):
        if on[k]:
            j = k
            while j < len(on) and on[j]:
                j += 1
            if runs and k - runs[-1][1] <= MERGE_GAP:
                runs[-1][1] = j
            else:
                runs.append([k, j])
            k = j
        else:
            k += 1
    to_s = lambda k: phase + (k0 + k) * six
    return [(to_s(a) - six / 2, to_s(b) - six / 2, float(np.nanmax(score[a:b]))) for a, b in runs if b - a >= MIN_LEN]


def _snap(segments, onsets, bar_len, beat):
    """Set each fill's edges to its first and last hit that repeats nowhere 1 or 2 bars away."""
    fill_hits = []
    for times, _ in onsets:
        for x in times:
            if not any(np.any(np.abs(times - (x + k * bar_len)) <= REPEAT_TOL) for k in (-2, -1, 1, 2)):
                fill_hits.append(x)
    fill_hits = np.sort(fill_hits)
    out = []
    for a, b, peak in segments:
        inside = fill_hits[(fill_hits >= a - beat) & (fill_hits <= b + beat)]
        seg = (float(inside[0]), float(inside[-1]), peak) if len(inside) >= 2 else (a, b, peak)
        if out and seg[0] <= out[-1][1]:  # snapping can make neighbours overlap: one fill
            seg = (out[-1][0], max(out[-1][1], seg[1]), max(out[-1][2], seg[2]))
            out.pop()
        out.append(seg)
    return out


def detect_fills(ref, cover, t):
    """Fill segments and scores for the original and your take, over the stretch your take covers."""
    start, bpm = t["start"], t["bpm"]
    six = 15 / bpm
    cover_len = len(cover) / SR
    ref_on = _onsets(ref, round(start * SR), len(cover), 0.0)
    cov_on = _onsets(cover, 0, len(cover), start)  # onto song time
    phase = _grid_phase(np.concatenate([t_ for t_, _ in ref_on]), bpm)
    k0 = int(np.floor((start - phase) / six))
    n = int(np.ceil(cover_len / six)) + 2
    out = dict(start=start, bpm=bpm, six=six, phase=phase, k0=k0, onsets=dict(original=ref_on, you=cov_on))
    for who, on in (("original", ref_on), ("you", cov_on)):
        score = fill_score(_grid(on, phase, six, k0, n))
        out[who] = dict(score=score, fills=_snap(_segments(score, phase, six, k0), on, 16 * six, 4 * six))
    return out


def _overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def pair_fills(f):
    """[(original fill or None, your fill or None)] in time order; overlapping fills are a pair."""
    yours = list(f["you"]["fills"])
    pairs = []
    for o in f["original"]["fills"]:
        best = max(yours, key=lambda y: _overlap(o, y), default=None)
        if best is not None and _overlap(o, best) > 0:
            yours.remove(best)
            pairs.append((o, best))
        else:
            pairs.append((o, None))
    pairs += [(None, y) for y in yours]
    return sorted(pairs, key=lambda p: (p[0] or p[1])[0])


def _position(song_s, f):
    bar_len = 240 / f["bpm"]
    c = max(0.0, song_s - f["start"])
    return f"{int(c // bar_len) + 1}.{(c % bar_len) / (bar_len / 4) + 1:.2f}"


def fill_report(f):
    lines = ["## Fills",
             "Where each drummer leaves the groove (the beat matches nothing 1–2 bars around it). "
             "Positions are bar.beat in your take; times are song time."]
    pairs = pair_fills(f)
    if not pairs:
        return "\n".join(lines + ["", "No fills found: both tracks repeat their groove all the way."])

    def cell(seg):
        if seg is None:
            return "—"
        return f"{seg[0]:.3f}–{seg[1]:.3f} s<br>bar {_position(seg[0], f)}–{_position(seg[1], f)}"

    def diff(ms):
        return "same" if abs(ms) < 5 else f"{abs(ms):.0f} ms {'late' if ms > 0 else 'early'}"

    lines += ["", "| # | Original fill | Your fill | You start | You end | Length (original → yours) |",
              "|---|---|---|---|---|---|"]
    for i, (o, y) in enumerate(pairs, 1):
        if o and y:
            lines.append(f"| {i} | {cell(o)} | {cell(y)} | **{diff(1000 * (y[0] - o[0]))}** | **{diff(1000 * (y[1] - o[1]))}** | "
                         f"{o[1] - o[0]:.2f} → {y[1] - y[0]:.2f} s |")
        else:
            seg = o or y
            lines.append(f"| {i} | {cell(o)} | {cell(y)} | {'you keep the groove' if o else 'only you fill'} | | "
                         f"{seg[1] - seg[0]:.2f} s |")
    both = sum(bool(o and y) for o, y in pairs)
    lines += ["", f"**{both}** fills you both play · **{sum(bool(o and not y) for o, y in pairs)}** only in the original · "
                  f"**{sum(bool(y and not o) for o, y in pairs)}** only yours · start/end = your first/last fill hit "
                  f"against the original's"]
    return "\n".join(lines)


def fill_plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    bar_len = 240 / f["bpm"]
    bar = lambda s: (np.asarray(s) - f["start"]) / bar_len + 1  # song s -> bar in your take
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    colours = {"Kick": "#1f77b4", "Snare": "#ff7f0e", "Cymbal": "#2ca02c"}
    rows = {"original": 1.0, "you": 0.0}
    for who, y in rows.items():
        for seg in f[who]["fills"]:
            ax.add_patch(plt.Rectangle((bar(seg[0]), y - 0.45), bar(seg[1]) - bar(seg[0]), 0.9, color="#8e7cc3",
                                       alpha=0.28, lw=0))
            ax.text(bar(seg[0]), y + (0.47 if who == "original" else -0.47), f"{seg[0]:.2f}–{seg[1]:.2f} s",
                    fontsize=7, color="#5b4a9e", va="bottom" if who == "original" else "top")
        for b, (name, _, _) in enumerate(BANDS):
            times, strength = f["onsets"][who][b]
            lane = y - 0.3 + 0.3 * b  # kick at the bottom of each row, cymbal on top
            half = 0.05 * np.minimum(strength, 2) + 0.02  # taller tick = stronger hit
            ax.vlines(bar(times), lane - half, lane + half, color=colours[name], lw=1.1,
                      label=name if who == "you" else None)
    ax.set_yticks([1, 0], ["Original", "You"])
    ax.set_ylim(-0.7, 1.7)
    ax.grid(axis="x", color="#cccccc", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, fontsize=8, frameon=False,
              handles=ax.get_legend_handles_labels()[0] + [plt.Rectangle((0, 0), 1, 1, color="#8e7cc3", alpha=0.28)],
              labels=ax.get_legend_handles_labels()[1] + ["fill"])

    k = np.arange(len(f["original"]["score"]))
    song = f["phase"] + (f["k0"] + k) * f["six"]
    for who, colour in (("original", "#4a4a4a"), ("you", "#ff7f0e")):
        bx.plot(bar(song), f[who]["score"], color=colour, lw=1.2, label=who)
    bx.axhline(THRESHOLD, color="gray", ls="--", lw=0.8)
    bx.set_ylim(0, 1)
    bx.set_ylabel("fill score")
    bx.set_xlabel("bar in your take")
    bx.legend(loc="upper right", fontsize=8, ncol=2)
    top = ax.secondary_xaxis("top", functions=(lambda v: (v - 1) * bar_len + f["start"], lambda s: (s - f["start"]) / bar_len + 1))
    top.xaxis.set_major_formatter(FuncFormatter(lambda s, _: f"{int(s // 60)}:{s % 60:04.1f}"))
    top.tick_params(labelsize=8)
    ax.set_xlim(bar(f["start"]), bar(f["start"]) + len(k) * f["six"] / bar_len)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    fig.tight_layout()
    return fig
