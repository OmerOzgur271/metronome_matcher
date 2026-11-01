"""After the start is found: compare every hit of the cover with the original and find timing mistakes.

Each cover hit is paired with the nearest original hit in the same band, searching around
where the recent hits say it should be, so slow tempo drift is followed instead of breaking
the pairing. Errors are split into drift (the local trend) and mistakes (a hit off its trend).
Hits that are off are then sorted out:
  - weak onsets are mostly another drum leaking into the band -> "unclear", not reported;
  - several problems within a beat means the cover plays something else there -> "fill";
  - anything else is a real "mistake". Hits at the same moment in different bands are one mistake.
"""
import numpy as np

from .config import BANDS, DRIFT_WINDOW, FINE, MISTAKE_MS, SR, STABLE_BANDS, WEAK_HIT
from .features import local_onsets, long_envelopes

PAD = 0.5       # s of reference read around the cover
MOMENT = 0.015  # s, hits in different bands closer than this are the same stroke
FILL_STROKES = 4  # this many problem strokes within ± a beat = the cover plays a different pattern
STABLE = {b for b, (name, _, _) in enumerate(BANDS) if name in STABLE_BANDS}


def _band_onsets(x, start, length, min_gap):
    """[(times, strengths)] per band. min_gap drops the second peak a ringing drum leaves in its own band."""
    hop, win = FINE
    env = long_envelopes(x, start, length, hop, win)
    return [local_onsets(e, hop / SR, start / SR, min_gap=min_gap) for e in env]


def _pair_hits(cover_on, ref_on, T, window):
    """Pair cover hits (cover time) with original hits (song time), following drift."""
    events = sorted((t, s, b) for b, (times, strength) in enumerate(cover_on) for t, s in zip(times, strength))
    used = [np.zeros(len(r), bool) for r, _ in ref_on]
    recent, hits = [], []
    for tau, strength, b in events:
        song = T + tau
        expected = song - (np.median(recent[-16:]) if recent else 0.0)
        r, r_strength = ref_on[b]
        lo, hi = np.searchsorted(r, [expected - window, expected + window])
        free = [j for j in range(lo, hi) if not used[b][j]]
        if not free:
            hits.append(dict(band=b, cover_t=tau, song_t=song, ref_t=None, err=None, strength=strength))
            continue
        j = min(free, key=lambda j: abs(r[j] - expected))
        used[b][j] = True
        hits.append(dict(band=b, cover_t=tau, song_t=song, ref_t=r[j], err=song - r[j],
                         strength=min(strength, r_strength[j])))
        if b in STABLE:
            recent.append(song - r[j])
    missed = [dict(band=b, ref_t=r[j], strength=s[j])
              for b, (r, s) in enumerate(ref_on) for j in np.flatnonzero(~used[b])]
    return hits, missed


def _trend(hits):
    """Local drift at every paired hit: median error of stable-band hits within ± DRIFT_WINDOW."""
    paired = [h for h in hits if h["err"] is not None]
    pts = sorted((h["song_t"], h["err"]) for h in paired if h["band"] in STABLE and h["strength"] >= WEAK_HIT)
    if not pts:
        pts = sorted((h["song_t"], h["err"]) for h in paired)
    t, e = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    for h in paired:
        lo, hi = np.searchsorted(t, [h["song_t"] - DRIFT_WINDOW, h["song_t"] + DRIFT_WINDOW])
        h["drift"] = float(np.median(e[lo:hi])) if hi > lo else 0.0
        h["local"] = h["err"] - h["drift"]


def _strokes(points):
    """Cluster (time, clean) points across bands into strokes: (time, any band clean)."""
    out = []
    for t, clean in sorted(points):
        if out and t - out[-1][2] <= MOMENT:
            out[-1][1] |= clean
            out[-1][2] = t
        else:
            out.append([t, clean, t])
    return [(t, clean) for t, clean, _ in out]


def _classify(hits, missed, beat):
    paired = [h for h in hits if h["err"] is not None]
    off = [h for h in paired if abs(h["local"]) * 1000 > MISTAKE_MS]
    # A stroke is a problem only if no band pairs it cleanly: kits differ in which bands a drum
    # shows up in, so one band's extra/missed hit means little on its own.
    for h in hits:
        h["clean"] = h["err"] is not None and abs(h["local"]) * 1000 <= MISTAKE_MS
    cover_strokes = _strokes([(h["song_t"], h["clean"]) for h in hits if h["strength"] >= WEAK_HIT])
    ref_strokes = _strokes([(h["ref_t"], h["clean"]) for h in paired if h["strength"] >= WEAK_HIT]
                           + [(m["ref_t"], False) for m in missed if m["strength"] >= WEAK_HIT])
    trouble = np.array(sorted(t for t, clean in cover_strokes + ref_strokes if not clean))

    def busy(t):
        lo, hi = np.searchsorted(trouble, [t - beat, t + beat])
        return hi - lo >= FILL_STROKES

    off_t = np.array(sorted(h["song_t"] for h in off))
    off_strength = [h["strength"] for h in sorted(off, key=lambda h: h["song_t"])]
    for h in hits:
        h["in_fill"] = busy(h["song_t"])
        if h["err"] is None:
            h["status"] = "extra"
        elif abs(h["local"]) * 1000 <= MISTAKE_MS:
            h["status"] = "ok"
        else:
            lo, hi = np.searchsorted(off_t, [h["song_t"] - MOMENT, h["song_t"] + MOMENT])
            if max(off_strength[lo:hi]) < WEAK_HIT:
                h["status"] = "unclear"
            else:
                h["status"] = "fill" if h["in_fill"] else "mistake"
    for m in missed:
        m["status"] = "missed"
        m["in_fill"] = busy(m["ref_t"])


def _mistakes(hits):
    """One entry per stroke: the strongest off hit, plus every band that heard it."""
    bad = sorted((h for h in hits if h["status"] == "mistake"), key=lambda h: h["song_t"])
    groups = []
    for h in bad:
        if groups and h["song_t"] - groups[-1][-1]["song_t"] <= MOMENT:
            groups[-1].append(h)
        else:
            groups.append([h])
    out = []
    for g in groups:
        top = max(g, key=lambda h: h["strength"])
        out.append(dict(top, bands=[BANDS[b][0] for b in sorted({h["band"] for h in g})]))
    return out


def _position(t, bpm):
    """Song time -> (bar, beat) to the nearest 16th; bar 1 starts at 0 s, 4/4."""
    beat_len = 60 / bpm
    bar = int(t // (4 * beat_len))
    beat = round(4 * ((t - bar * 4 * beat_len) / beat_len + 1)) / 4
    if beat >= 5:
        bar, beat = bar + 1, beat - 4
    return bar + 1, beat


def _tempo_fit(hits, bpm):
    """Line through the stable hits' errors: drift per second -> the tempo the cover was played at."""
    pts = [(h["cover_t"], h["err"]) for h in hits if h["status"] == "ok" and h["band"] in STABLE]
    if len(pts) < 8:
        return None
    x, y = np.array(pts).T
    slope, icpt = np.polyfit(x, y, 1)
    return dict(slope=float(slope), icpt=float(icpt), cover_bpm=bpm * (1 - slope),
                start_ms=1000 * icpt, end_ms=1000 * (icpt + slope * x.max()))


def _bars(hits, missed, mistakes):
    bars = {}

    def bar(n):
        return bars.setdefault(n, dict(bar=n, errs=[], drift=[], mistakes=0, extra=0, missed=0, fill=False))

    for h in hits:
        s = bar(h["bar"])
        s["fill"] |= h["in_fill"]
        if h["err"] is None:
            s["extra"] += 1
        elif h["status"] == "ok":
            s["errs"].append(h["err"])
            s["drift"].append(h["drift"])
    for m in missed:
        s = bar(m["bar"])
        s["missed"] += 1
        s["fill"] |= m["in_fill"]
    for m in mistakes:
        bar(m["bar"])["mistakes"] += 1
    return [dict(bar=s["bar"], matched=len(s["errs"]), mistakes=s["mistakes"], extra=s["extra"], missed=s["missed"],
                 drift_ms=1000 * float(np.median(s["drift"])) if s["drift"] else None, pattern_differs=s["fill"])
            for s in sorted(bars.values(), key=lambda s: s["bar"])]


def check_timing(ref, cover, start, bpm):
    """Compare the whole cover, placed at `start` s, with the original hit by hit."""
    sixteenth = 15 / bpm
    gap = min(0.06, 0.4 * sixteenth)
    cover_on = _band_onsets(cover, 0, len(cover), gap)
    ref_on = _band_onsets(ref, round((start - PAD) * SR), len(cover) + round(2 * PAD * SR), gap)
    hits, missed = _pair_hits(cover_on, ref_on, start, window=min(0.06, 0.45 * sixteenth))
    missed = [m for m in missed if start <= m["ref_t"] <= start + len(cover) / SR]
    _trend(hits)
    _classify(hits, missed, 60 / bpm)
    for h in hits:
        h["bar"], h["beat"] = _position(h["ref_t"] if h["ref_t"] is not None else h["song_t"], bpm)
    for m in missed:
        m["bar"], m["beat"] = _position(m["ref_t"], bpm)
    mistakes = _mistakes(hits)
    bars, tempo = _bars(hits, missed, mistakes), _tempo_fit(hits, bpm)
    for b in bars:  # drift against the cover's own steady tempo: rushing / dragging
        centre = (b["bar"] - 0.5) * 240 / bpm - start
        b["vs_tempo_ms"] = (None if b["drift_ms"] is None or tempo is None
                            else b["drift_ms"] - 1000 * (tempo["icpt"] + tempo["slope"] * centre))
    return dict(hits=hits, missed=missed, mistakes=mistakes, bars=bars, tempo=tempo, bpm=bpm, start=start)
