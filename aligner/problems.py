"""Find the time ranges where the cover's timing is worst, say why, and how to fix it.

A hit counts as off when it is more than MISTAKE_MS from the original: that is what you hear in
the final mix. When the cover's tempo drifts past that over the song, the drift is reported once
as its own problem and hits are measured against your own (drifting) tempo instead; otherwise
every bar after the drift builds up would count as off and hide the real local problems.

Only clear hits outside fills count (in a fill the cover plays other notes, so there is nothing
reliable to compare). Windows of RANGE_BARS bars are ranked by the share of off hits, overall or
of one drum alone (a late kick is only a few of a bar's hits, but you hear it). The worst
non-overlapping windows are merged where they touch and trimmed to the bars with off hits. Each
range is then diagnosed from how its hits are off:
  - the whole section late/early by the same amount -> dragging / rushing;
  - hits scattered around your own pulse -> loose timing;
  - one drum much worse than the others, one subdivision, or the bar after a fill.
"""
import numpy as np

from .config import BANDS, MISTAKE_MS, WEAK_HIT

RANGE_BARS = 4     # look for problems in windows of this many bars
MIN_RATE = 0.2     # a window is a problem when this share of its hits (or of one drum's hits) is off
MIN_HITS = 8       # windows with fewer hits than this are not judged
MIN_BAND_HITS = 4  # a drum needs this many hits in a window for its own share to count
LOOSE_MS = 12      # hits spread more than this around your own pulse = loose timing
SHIFT_MS = 12      # a section's typical error past this = it sits early/late as a block
TOP = 5


def tempo_drift(t):
    """The tempo fit when the cover drifts more than MISTAKE_MS from the original over the song."""
    tempo = t["tempo"]
    return tempo if tempo and abs(tempo["end_ms"] - tempo["start_ms"]) > MISTAKE_MS else None


def _counted(t):
    """Clear paired hits outside fills, each with "e": its error against the original, or against
    your own tempo when it drifts."""
    drift = tempo_drift(t)
    out = []
    for h in t["hits"]:
        if h["err"] is not None and h["strength"] >= WEAK_HIT and h["status"] in ("ok", "mistake"):
            base = drift["icpt"] + drift["slope"] * h["cover_t"] if drift else 0.0
            out.append(dict(h, e=h["err"] - base))
    return out


def _off(h):
    return abs(h["e"]) * 1000 > MISTAKE_MS


def mmss(sec):
    sec = max(sec, 0.0)
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def bar_rates(hits, band=None):
    """{bar: (hits, off hits)}, optionally for one band."""
    out = {}
    for h in hits:
        if band is None or h["band"] == band:
            n, k = out.get(h["bar"], (0, 0))
            out[h["bar"]] = (n + 1, k + _off(h))
    return out


def _score(rates, band_rates, a, b):
    """Worst share of off hits in bars a..b: over all hits, or over one drum's hits."""
    def total(rs, i):
        return sum(rs.get(n, (0, 0))[i] for n in range(a, b + 1))

    n, k = total(rates, 0), total(rates, 1)
    if n < MIN_HITS:
        return None
    shares = [k / n] + [total(rs, 1) / total(rs, 0) for rs in band_rates if total(rs, 0) >= MIN_BAND_HITS]
    return max(shares), k


def _spans(hits):
    """(first bar, last bar, score) of the stretches with the highest share of off hits."""
    rates = bar_rates(hits)
    if not rates:
        return []
    band_rates = [bar_rates(hits, band) for band in range(len(BANDS))]
    first, last = min(rates), max(rates)
    size = min(RANGE_BARS, last - first + 1)

    windows = []
    for a in range(first, last - size + 2):
        score = _score(rates, band_rates, a, a + size - 1)
        if score and score[0] >= MIN_RATE:
            windows.append((*score, a))
    picked = []
    for _, _, a in sorted(windows, reverse=True):
        if all(a + size <= p or p + size <= a for p in picked):
            picked.append(a)
    spans = []
    for a in sorted(picked):
        if spans and a <= spans[-1][1] + 1:
            spans[-1][1] = a + size - 1
        else:
            spans.append([a, a + size - 1])
    def bad_bar(n):  # bad on its own: over all its hits, or one drum with at least two hits
        shares = [k / m for m, k in [rates.get(n, (0, 0))] + [rs.get(n, (0, 0)) for rs in band_rates] if m >= 2]
        return max(shares, default=0) >= MIN_RATE

    trimmed = []
    for a, b in spans:
        bad = [n for n in range(a, b + 1) if bad_bar(n)] or [a, b]
        trimmed.append((bad[0], bad[-1], _score(rates, band_rates, bad[0], bad[-1])[0]))
    return trimmed


def _subdivision(beat):
    frac = beat % 1
    return "on the beat" if frac == 0 else "on the &s" if frac == 0.5 else "on the 16ths (e / a)"


def _diagnose(t, hits, a, b, score, fills):
    bpm, start = t["bpm"], t["start"]
    bar_len = 240 / bpm
    drifting = tempo_drift(t) is not None
    against = "your own tempo" if drifting else "the original"
    hs = [h for h in hits if a <= h["bar"] <= b]
    off = [h for h in hs if _off(h)]
    err = 1000 * float(np.median([h["e"] for h in hs]))
    local = np.array([h["local"] for h in hs])
    spread = 1000 * 1.4826 * float(np.median(np.abs(local - np.median(local))))  # robust: a few bad hits don't count
    r = dict(first=a, last=b, hits=len(hs), off=len(off), rate=len(off) / len(hs), score=score, err_ms=err,
             spread_ms=spread, song_from=(a - 1) * bar_len, song_to=b * bar_len)
    r["take_from"], r["take_to"] = r["song_from"] - start, r["song_to"] - start
    late = err > 0
    slow, practise = round(0.8 * bpm), round(0.9 * bpm)
    where = f"song {mmss(r['song_from'])}–{mmss(r['song_to'])} ({mmss(r['take_from'])}–{mmss(r['take_to'])} in your take)"
    tags, causes, fixes = [], [], []

    shifted = abs(err) > SHIFT_MS
    if shifted and late:
        tags.append("dragging")
        causes.append(f"**Dragging:** the whole section sits about {err:.0f} ms behind {against}.")
        fixes.append(f"Loop {where} with the original in your headphones at full tempo and play on top of the "
                     f"beat: lock your hi-hat to the original's hi-hat. Dragging often comes from playing too hard "
                     f"or tensing up, so loosen the grip and let the sticks bounce.")
    elif shifted:
        tags.append("rushing")
        causes.append(f"**Rushing:** the whole section runs about {-err:.0f} ms ahead of {against}.")
        fixes.append(f"Loop {where} with a click at {slow} BPM (80%), count the quarter notes out loud, then "
                     f"{practise} and {bpm:g} BPM. Rushing usually comes from excitement: let the backbeat "
                     f"land, don't chase it.")
    if spread > LOOSE_MS:
        tags.append("loose")
        causes.append(f"**Loose:** your hits scatter about ±{spread:.0f} ms around your own pulse, "
                      f"so it's not one steady push or pull.")
        fixes.append(f"Slow this groove down to {round(0.7 * bpm)} BPM with a click on 8th notes; play it until four "
                     f"times in a row feel locked, then add 5 BPM at a time. Record and listen back.")

    band_rate = {}
    for band, (name, _, _) in enumerate(BANDS):
        bh = [h for h in hs if h["band"] == band]
        if len(bh) >= 3:
            band_rate[name] = (np.mean([_off(h) for h in bh]), 1000 * float(np.median([h["e"] for h in bh])))
    for name, (rate, band_err) in band_rate.items():
        others = [o for n, (o, _) in band_rate.items() if n != name]
        if rate >= 0.4 and others and all(o <= rate / 2 for o in others):
            tags.append(f"mostly {name.lower()} ({rate * 100:.0f}% of its hits off)")
            causes.append(f"Mostly the **{name.lower()}**: {rate * 100:.0f}% of its hits are off, "
                          f"typically {abs(band_err):.0f} ms {'late' if band_err > 0 else 'early'}.")
            fixes.append({
                "Kick": f"Your foot is {'behind' if band_err > 0 else 'ahead of'} your hands. Play hi-hat + kick "
                        f"together on quarter notes at {slow} BPM until they sound like one note, and check the "
                        f"pedal spring tension.",
                "Snare": "Practise the groove with the click on 2 and 4 only, so the click is your backbeat: "
                         "the snare should hide the click.",
                "Cymbal": "Play the hi-hat/cymbal pattern alone with a click, all strokes even, then add the kick "
                          "and the snare back one at a time.",
            }[name])

    if len(off) >= 4 and not any(tag.startswith("mostly") for tag in tags):  # one drum already explains it
        subs = [_subdivision(h["beat"]) for h in off]
        top = max(set(subs), key=subs.count)
        base = np.mean([_subdivision(h["beat"]) == top for h in hs])
        share = subs.count(top) / len(subs)
        if share >= 0.6 and share >= 1.5 * base:
            tags.append(top)
            causes.append(f"{share * 100:.0f}% of the off hits fall {top}.")
            fixes.append("Count the subdivisions out loud while you play (\"1 e & a\") so those notes get their "
                         "own place in time.")

    after_fill = [h for h in off if h["bar"] - 1 in fills]
    if len(after_fill) >= 2 and len(after_fill) >= 0.4 * len(off):
        side = "late" if np.median([h["e"] for h in after_fill]) > 0 else "early"
        tags.append("after a fill")
        causes.append(f"It goes wrong right after a fill: you come out of the fill {side}.")
        fixes.append("Loop the bar before the fill, the fill and the bar after as one piece; land the crash and "
                     "kick exactly on 1. Play the fill simpler or slower rather than cramming it.")

    if not causes:
        tags.append("scattered hits")
        causes.append("Single hits off in different directions, with nothing in common.")
        fixes.append(f"Loop {where} at {slow} BPM with a click until it's clean, then speed it back up.")
    if shifted and spread <= LOOSE_MS and not drifting:
        fixes.append(f"**Or fix the recording:** split your region at {mmss(r['take_from'])} and "
                     f"{mmss(r['take_to'])} and move that piece {abs(err):.0f} ms {'earlier' if late else 'later'} "
                     f"(turn off Snap to Grid and zoom all the way in). This works here because the whole section "
                     f"is off by the same amount.")
    r["tags"], r["causes"], r["fixes"] = tags, causes, fixes
    return r


def problem_ranges(t, top=TOP):
    """The worst time ranges, worst first, each with what's wrong and how to fix it."""
    fills = {b["bar"] for b in t["bars"] if b["pattern_differs"]}
    hits = _counted(t)
    ranges = [_diagnose(t, hits, a, b, score, fills) for a, b, score in _spans(hits)]
    return sorted(ranges, key=lambda r: (-r["score"], -r["off"]))[:top]


def problems_report(t, ranges=None):
    ranges = problem_ranges(t) if ranges is None else ranges
    lines = []
    drift = tempo_drift(t)
    if drift:
        slower = drift["cover_bpm"] < t["bpm"]
        lines += [
            "### Biggest problem: your tempo drifts",
            f"You play at about **{drift['cover_bpm']:.1f} BPM** instead of {t['bpm']:g}, so you go from "
            f"{drift['start_ms']:+.0f} ms to **{drift['end_ms']:+.0f} ms** against the original "
            f"({'falling behind' if slower else 'running ahead'} over the song).",
            f"- **How to fix it:** record with a click at {t['bpm']:g} BPM in your headphones, or with the original "
            f"playing in them. A slow drift like this can't be fixed by moving regions; it has to be played again.",
            "",
            "The ranges below are measured against **your own tempo**, so they show the problems on top of the drift.",
            "",
        ]
    against = "your own tempo" if drift else "the original"
    lines.append(f"### Worst time ranges (share of hits more than {MISTAKE_MS} ms off {against})")
    if not ranges:
        lines.append(f"No {RANGE_BARS}-bar stretch has more than {MIN_RATE * 100:.0f}% of its hits off. " +
                     ("Apart from the tempo drift, your timing holds up." if drift
                      else "Your timing holds up across the whole cover."))
        return "\n".join(lines)
    lines += ["", "| # | Song time | In your take | Bars | Hits off | Typical error | What's wrong |",
              "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ranges, 1):
        bars = f"{r['first']}" + (f"–{r['last']}" if r["last"] > r["first"] else "")
        lines.append(f"| **{i}** | {mmss(r['song_from'])}–{mmss(r['song_to'])} | "
                     f"{mmss(r['take_from'])}–{mmss(r['take_to'])} | {bars} | {r['off']}/{r['hits']} "
                     f"({r['rate'] * 100:.0f}%) | {r['err_ms']:+.0f} ms {'late' if r['err_ms'] > 0 else 'early'} | "
                     f"{', '.join(r['tags'])} |")
    for i, r in enumerate(ranges, 1):
        bars = f"bar {r['first']}" + (f"–{r['last']}" if r["last"] > r["first"] else "")
        lines += ["", f"#### #{i} · {mmss(r['song_from'])}–{mmss(r['song_to'])} ({bars})", "**What's wrong:**"]
        lines += [f"- {c}" for c in r["causes"]]
        lines += ["", "**How to fix it:**"]
        lines += [f"- {f}" for f in r["fixes"]]
    return "\n".join(lines)
