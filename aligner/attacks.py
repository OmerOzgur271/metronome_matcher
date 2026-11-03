"""Attack analyser: where every hit starts, peaks and ends, in your take and in the original.

The timing check finds hits with 23 ms analysis windows, too coarse to say exactly where an attack
begins. Here each band gets a fine amplitude envelope (zero-phase band-pass, ~2 ms smoothing; the
same filter on both tracks, so its smearing cancels out in the comparison). Around each hit:
  - peak:  the envelope's maximum just after the detected onset;
  - start: going back from the peak, where the rise passes ATTACK_LEVEL of the way up from the
           level before the hit (other drums still ringing set that floor);
  - end:   where it has decayed to END_LEVEL of the way back down, or the band's next hit.
Every paired hit gets both versions; hits only you play ("extra") and original hits you don't
("missed") get one side. Times are song times (your take placed at the found start), positions
are bar.beat in your take counting from the start of your file, i.e. GarageBand's bar.beat when
the export starts at bar 1.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from .config import BANDS, MISTAKE_MS, SR, WEAK_HIT

SMOOTH = 0.002       # s of envelope smoothing
ATTACK_LEVEL = 0.2   # the attack starts where the rise passes this share of floor -> peak
END_LEVEL = 0.1      # and ends where the decay is back down to this share
PEAK_SEARCH = (-0.015, 0.04)  # s around the detected onset to look for the peak
MAX_LENGTH = 1.5     # s
FLAM_TOGETHER = 0.008  # s, original hits this close in different bands are one stroke...
FLAM_SPLIT = 0.015     # ...and you split it when yours are further apart than this


def _envelopes(x, start, n):
    """Per-band amplitude envelopes of x[start:start + n]."""
    seg = x[max(0, start):max(0, start) + n].astype(np.float64)
    out = []
    for _, lo, hi in BANDS:
        sos = butter(4, [lo, min(hi, 0.999 * SR / 2)], btype="band", fs=SR, output="sos")
        y = sosfiltfilt(sos, seg)
        out.append(np.sqrt(np.maximum(0, uniform_filter1d(y * y, max(1, round(SMOOTH * SR))))))
    return out, max(0, start) / SR


def _shape(env, t0, onset, next_onset):
    """(start, peak, end) in s for the hit detected at `onset`, or None if there is nothing there."""
    i = lambda t: int(round((t - t0) * SR))
    lo, hi = max(0, i(onset + PEAK_SEARCH[0])), min(len(env), i(onset + PEAK_SEARCH[1]))
    if hi - lo < 2:
        return None
    p = lo + int(np.argmax(env[lo:hi]))
    before = env[max(0, p - round(0.06 * SR)):p + 1]
    floor = float(before.min())
    peak = float(env[p])
    if peak <= floor * 1.2:
        return None
    s = p
    rise = floor + ATTACK_LEVEL * (peak - floor)
    while s > 0 and p - s < 0.06 * SR and env[s - 1] > rise:
        s -= 1
    stop = min(len(env), i(min(onset + MAX_LENGTH, next_onset - 0.002)) if next_onset else i(onset + MAX_LENGTH))
    tail = env[p:max(p + 1, stop)]
    after = float(tail.min()) if len(tail) else floor
    below = np.flatnonzero(tail <= after + END_LEVEL * (peak - after))
    e = p + (int(below[0]) if len(below) else len(tail) - 1)
    return t0 + s / SR, t0 + p / SR, t0 + e / SR


def _position(cover_t, bpm):
    bar_len = 240 / bpm
    return f"{int(cover_t // bar_len) + 1}.{(cover_t % bar_len) / (bar_len / 4) + 1:.2f}"


def analyse_attacks(ref, cover, t):
    """One row per hit: your attack and the original's, side by side (see the module docstring)."""
    start, bpm = t["start"], t["bpm"]
    cov_env, cov_t0 = _envelopes(cover, 0, len(cover))
    ref_env, ref_t0 = _envelopes(ref, round((start - 1) * SR), len(cover) + 2 * SR)

    hits = [h for h in t["hits"] if h["strength"] >= WEAK_HIT and h["status"] != "unclear"]
    missed = [m for m in t["missed"] if m["strength"] >= WEAK_HIT]
    nxt_cov, nxt_ref = {}, {}
    for b in range(len(BANDS)):  # each hit's attack ends by the band's next hit at the latest
        ct = sorted(h["cover_t"] for h in t["hits"] if h["band"] == b)
        rt = sorted([h["ref_t"] for h in t["hits"] if h["band"] == b and h["ref_t"] is not None] +
                    [m["ref_t"] for m in t["missed"] if m["band"] == b])
        nxt_cov.update({(b, a): c for a, c in zip(ct, ct[1:])})
        nxt_ref.update({(b, a): c for a, c in zip(rt, rt[1:])})

    rows = []
    for h in hits:
        b = h["band"]
        yours = _shape(cov_env[b], cov_t0, h["cover_t"], nxt_cov.get((b, h["cover_t"])))
        orig = _shape(ref_env[b], ref_t0, h["ref_t"], nxt_ref.get((b, h["ref_t"]))) if h["ref_t"] is not None else None
        if yours is None or (h["ref_t"] is not None and orig is None):
            continue
        yours = tuple(v + start for v in yours)  # onto song time
        rows.append(_row(b, h["cover_t"], bpm, h["in_fill"], "extra" if orig is None else "matched", yours, orig))
    if hits:  # original hits before your first / after your last aren't missed, you weren't playing yet
        first, last = min(h["song_t"] for h in hits) - 0.05, max(h["song_t"] for h in hits) + 0.05
        missed = [m for m in missed if first <= m["ref_t"] <= last]
    for m in missed:
        orig = _shape(ref_env[m["band"]], ref_t0, m["ref_t"], nxt_ref.get((m["band"], m["ref_t"])))
        if orig is not None:
            rows.append(_row(m["band"], m["ref_t"] - start, bpm, m["in_fill"], "missed", None, orig))
    return sorted(rows, key=lambda r: r["song_t"])


def _row(band, cover_t, bpm, in_fill, status, yours, orig):
    def part(s):
        return dict(start=s[0], peak=s[1], end=s[2], rise_ms=1000 * (s[1] - s[0]), length_ms=1000 * (s[2] - s[0])) \
            if s else None
    y, o = part(yours), part(orig)
    return dict(band=BANDS[band][0], position=_position(max(cover_t, 0.0), bpm), cover_t=cover_t,
                song_t=(y or o)["start"], part="fill" if in_fill else "groove", status=status, you=y, orig=o,
                start_ms=1000 * (y["start"] - o["start"]) if y and o else None,
                length_ms=y["length_ms"] - o["length_ms"] if y and o else None)


def attack_table(rows):
    """Rows for the UI table."""
    def f(v, nd=3):
        return "" if v is None else round(float(v), nd)
    head = ["Position (your take)", "Song time (s)", "Drum", "Part", "Who", "Your start (s)", "Original start (s)",
            "Start diff (ms)", "Your end (s)", "Original end (s)", "Your length (ms)", "Original length (ms)",
            "Your rise (ms)", "Original rise (ms)"]
    who = {"matched": "both", "extra": "only you", "missed": "only original"}
    data = []
    for r in rows:
        y, o = r["you"] or {}, r["orig"] or {}
        data.append([r["position"], f(r["song_t"]), r["band"], r["part"], who[r["status"]],
                     f(y.get("start")), f(o.get("start")), f(r["start_ms"], 1), f(y.get("end")), f(o.get("end")),
                     f(y.get("length_ms"), 0), f(o.get("length_ms"), 0), f(y.get("rise_ms"), 1), f(o.get("rise_ms"), 1)])
    return head, data


def _strokes(rows):
    """Original strokes: matched attacks whose original starts are within FLAM_TOGETHER across bands."""
    m = sorted((r for r in rows if r["status"] == "matched"), key=lambda r: r["orig"]["start"])
    groups = []
    for r in m:
        if groups and r["orig"]["start"] - groups[-1][0]["orig"]["start"] <= FLAM_TOGETHER:
            groups[-1].append(r)
        else:
            groups.append([r])
    return [g for g in groups if len({r["band"] for r in g}) > 1]


def attack_report(rows, t):
    m = [r for r in rows if r["status"] == "matched"]
    lines = ["## Attack analysis",
             "Every attack's start, peak and end in your take vs the original. **+ = you start late, − = early.** "
             "Positions are bar.beat in your take (GarageBand's, if your export starts at bar 1)."]
    if not m:
        return "\n".join(lines + ["", "No attacks could be paired with the original."])
    d = np.array([r["start_ms"] for r in m])
    lines += ["", f"**{len(rows)} attacks:** {len(m)} in both, {sum(r['status'] == 'extra' for r in rows)} only you, "
                  f"{sum(r['status'] == 'missed' for r in rows)} only the original · start diff median {np.median(d):+.1f} ms, "
                  f"middle half {np.percentile(d, 25):+.0f}…{np.percentile(d, 75):+.0f} ms, "
                  f"{np.mean(np.abs(d) <= 10):.0%} within ±10 ms, {np.mean(np.abs(d) > MISTAKE_MS):.0%} over ±{MISTAKE_MS} ms"]

    first = sorted(m, key=lambda r: r["cover_t"])
    opening = [r for r in first if r["cover_t"] <= first[0]["cover_t"] + 240 / t["bpm"]]
    o = np.array([r["start_ms"] for r in opening])
    lines += ["", f"**Opening** (first bar from your first hit at {first[0]['position']}): median {np.median(o):+.0f} ms; "
                  "first strokes " + ", ".join(f"{r['position']} {r['band'].lower()} {r['start_ms']:+.0f}"
                                               for r in opening[:6])]

    bar_len = 240 / t["bpm"]
    lines += ["", "**Drift through your take** (median start diff per 4 bars)", "",
              "| Bars | Median | Spread (middle half) | Over ±20 ms | Attacks |", "|---|---|---|---|---|"]
    last = int(max(r["cover_t"] for r in m) // bar_len) + 1
    for a in range(1, last + 1, 4):
        x = np.array([r["start_ms"] for r in m if a <= r["cover_t"] / bar_len + 1 < a + 4])
        if len(x):
            lines.append(f"| {a}–{a + 3} | {np.median(x):+.1f} ms | {np.percentile(x, 25):+.0f}…{np.percentile(x, 75):+.0f} ms | "
                         f"{np.mean(np.abs(x) > MISTAKE_MS):.0%} | {len(x)} |")

    lines += ["", "**Per drum**", "", "| Drum | Start diff | Over ±20 ms | Your length | Original length | Your rise | Original rise |",
              "|---|---|---|---|---|---|---|"]
    for name, _, _ in BANDS:
        rs = [r for r in m if r["band"] == name]
        if rs:
            x = np.array([r["start_ms"] for r in rs])
            med = lambda who, key: np.median([r[who][key] for r in rs])
            lines.append(f"| {name} | {np.median(x):+.1f} ms | {np.mean(np.abs(x) > MISTAKE_MS):.0%} | "
                         f"{med('you', 'length_ms'):.0f} ms | {med('orig', 'length_ms'):.0f} ms | "
                         f"{med('you', 'rise_ms'):.1f} ms | {med('orig', 'rise_ms'):.1f} ms |")

    groove = np.array([r["start_ms"] for r in m if r["part"] == "groove"])
    fill = np.array([r["start_ms"] for r in m if r["part"] == "fill"])
    lines += ["", "**Groove vs fills:** " + " · ".join(
        f"{name} median {np.median(x):+.1f} ms, {np.mean(np.abs(x) > MISTAKE_MS):.0%} over ±{MISTAKE_MS} ms ({len(x)} attacks)"
        for name, x in (("groove", groove), ("fills", fill)) if len(x))]
    if len(groove) and len(fill) and abs(np.median(fill) - np.median(groove)) > 10:
        word = "rush" if np.median(fill) < np.median(groove) else "drag"
        lines.append(f"→ you **{word} your fills** by about {abs(np.median(fill) - np.median(groove)):.0f} ms "
                     f"compared with your groove (in fills the notes can also differ from the original's).")

    flams = []
    for g in _strokes(rows):
        starts = [r["you"]["start"] for r in g]
        if max(starts) - min(starts) > FLAM_SPLIT:
            first_r, last_r = min(g, key=lambda r: r["you"]["start"]), max(g, key=lambda r: r["you"]["start"])
            flams.append(f"{first_r['position']} {first_r['band'].lower()} then {last_r['band'].lower()} "
                         f"{1000 * (max(starts) - min(starts)):.0f} ms later")
    lines += ["", f"**Flams** (strokes the original plays together that you split by more than {FLAM_SPLIT * 1000:.0f} ms): "
                  + (f"{len(flams)} — " + "; ".join(flams[:8]) if flams else "none")]

    worst = sorted(m, key=lambda r: -abs(r["start_ms"]))[:10]
    lines += ["", "**Worst attacks**", "", "| Position | Drum | Part | Start diff |", "|---|---|---|---|"]
    lines += [f"| {r['position']} | {r['band']} | {r['part']} | {r['start_ms']:+.0f} ms |" for r in worst]
    return "\n".join(lines)


def attack_plot(rows, t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bar_len = 240 / t["bpm"]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw=dict(height_ratios=[2.3, 1]))
    m = [r for r in rows if r["status"] == "matched"]
    if not m:
        ax.text(0.5, 0.5, "No attacks could be paired with the original", ha="center", transform=ax.transAxes)
        return fig
    for b in sorted({int(r["cover_t"] // bar_len) for r in rows if r["part"] == "fill"}):
        ax.axvspan(b + 1, b + 2, color="#8e7cc3", alpha=0.13, lw=0)
    colours = {"Kick": "#1f77b4", "Snare": "#ff7f0e", "Cymbal": "#2ca02c"}
    for name, _, _ in BANDS:
        rs = [r for r in m if r["band"] == name]
        ax.scatter([r["cover_t"] / bar_len + 1 for r in rs], [r["start_ms"] for r in rs], s=14, alpha=0.75,
                   color=colours.get(name), label=name)
    xs = np.array([r["cover_t"] / bar_len + 1 for r in m])
    ds = np.array([r["start_ms"] for r in m])
    order = np.argsort(xs)
    xs, ds = xs[order], ds[order]
    trend = [np.median(ds[(xs > x - 1) & (xs < x + 1)]) for x in xs]
    ax.plot(xs, trend, color="black", lw=1.5, label="your trend (±1 bar)")
    lim = max(40, min(120, np.percentile(np.abs(ds), 98) * 1.2))
    for kind, y, mark in (("extra", lim * 0.93, "v"), ("missed", -lim * 0.93, "^")):
        rs = [r for r in rows if r["status"] == kind]
        ax.scatter([r["cover_t"] / bar_len + 1 for r in rs], [y] * len(rs), marker=mark, s=18, color="#999999",
                   label={"extra": "only you", "missed": "only the original"}[kind])
    for y in (-MISTAKE_MS, MISTAKE_MS):
        ax.axhline(y, color="gray", ls="--", lw=0.8)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("attack start vs original (ms)\n+ late / − early")
    ax.set_xlabel("bar in your take (purple = fill / different pattern)", labelpad=2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8, ncol=6, frameon=False)
    ax.set_title("Attack starts: every hit of yours against the original's", fontsize=11)

    names = [n for n, _, _ in BANDS if any(r["band"] == n for r in m)]
    xi = np.arange(len(names))
    for k, (who, label, colour) in enumerate((("you", "your length", "#ff7f0e"), ("orig", "original length", "#4a4a4a"))):
        bx.bar(xi + (k - 0.5) * 0.38, [np.median([r[who]["length_ms"] for r in m if r["band"] == n]) for n in names],
               width=0.38, color=colour, label=label)
    bx.set_xticks(xi, names)
    bx.set_ylabel("median attack length (ms)\nstart → decayed")
    bx.legend(fontsize=8)
    fig.tight_layout()
    return fig
