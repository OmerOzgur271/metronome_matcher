"""Find where the drum cover starts inside the extracted drums."""
import numpy as np

from .config import (ANCHOR_SEC, ANCHOR_WINDOW, BANDS, COARSE, FINE, MAX_CANDIDATES, SAME_PEAK, SEARCH_RADIUS,
                     SR, TOL, WEAK_HIT)
from .features import envelopes, local_onsets, long_envelopes, match, onsets
from .helpers import block_start, local_peaks, ncc, ncc_bands, parabolic_offset


def _refine_lag(cover_f, ref_f, centre, max_lag, mf):
    """Best fine lag near `centre`, with sub-frame offset and its NCC."""
    span = int(np.ceil(2.5 * COARSE[0] / FINE[0]))
    lags = range(max(0, centre - span), min(max_lag, centre + span) + 1)
    vals = {L: ncc_bands(cover_f, ref_f, 0, L, mf) for L in range(max(0, lags[0] - 1), min(max_lag, lags[-1] + 1) + 1)}
    best = max(lags, key=vals.get)
    delta = 0.0
    if best - 1 in vals and best + 1 in vals:  # parabolic sub-frame interpolation
        delta = parabolic_offset(vals[best - 1], vals[best], vals[best + 1])
    return best, delta, vals[best]


def _segment_drift(cover_f, ref_f, max_lag, mf, fs0, T):
    """Each slice of the cover finds its own best lag (±70 ms); they should all agree."""
    hop = FINE[0]
    K = int(np.clip(mf * hop / SR // 4, 2, 8))
    seg_len = mf // K
    segments = []
    for k in range(K):
        off = k * seg_len
        if cover_f[:, off:off + seg_len].mean() < 0.05:
            segments.append(None)
            continue
        sv = [ncc_bands(cover_f, ref_f, off, off + L, seg_len) for L in range(max_lag + 1)]
        segments.append(((fs0 + int(np.argmax(sv)) * hop) / SR - T) * 1000)
    return segments


def _band_hits(cover_f, ref_f, best, mf, fs0, T):
    """Per-band onset agreement between cover and reference at start T."""
    frame = FINE[0] / SR
    end = T + mf * frame
    bands = []
    for b, (name, _, _) in enumerate(BANDS):
        c_on = onsets(cover_f[b, :mf], frame, T)
        r_on = onsets(ref_f[b], frame, fs0 / SR)
        r_on = r_on[(r_on >= T - TOL) & (r_on <= end + TOL)]
        pairs = match(c_on, r_on)
        bands.append(dict(
            name=name, ncc=ncc(cover_f[b, :mf], ref_f[b, best:best + mf]),
            cover=len(c_on), ref=len(r_on), matched=len(pairs),
            f1=2 * len(pairs) / max(1, len(c_on) + len(r_on)),
            err_ms=float(np.median([(r - c) * 1000 for c, r in pairs])) if pairs else 0.0))
    return bands


def verify(ref, cover_f, coarse_t):
    """Refine one coarse candidate at fine resolution and check kick/snare/cymbal hits."""
    hop, win = FINE
    pad = round(0.07 * SR)
    fs0 = max(0, round(coarse_t * SR) - pad)
    ref_f = envelopes(ref, fs0, cover_f.shape[1] * hop + 2 * pad + win, hop, win)
    max_lag = 2 * pad // hop
    mf = min(cover_f.shape[1], ref_f.shape[1] - max_lag)
    centre = round((coarse_t * SR - fs0) / hop)
    best, delta, fine_ncc = _refine_lag(cover_f, ref_f, centre, max_lag, mf)
    T = (fs0 + (best + delta) * hop) / SR

    segments = _segment_drift(cover_f, ref_f, max_lag, mf, fs0, T)
    voiced = [s for s in segments if s is not None]
    bands = _band_hits(cover_f, ref_f, best, mf, fs0, T)
    active = [x for x in bands if x["cover"] >= 3 and x["ref"] >= 3]
    hit_score = np.mean([x["f1"] for x in active]) if active else 0.0
    return dict(T=T, fine_ncc=fine_ncc, hit_score=hit_score, bands=bands, segments=segments,
                consistency=np.mean([abs(s) <= 10 for s in voiced]) if voiced else 0.0,
                score=0.5 * fine_ncc + 0.5 * hit_score,
                ref_f=ref_f, ref_start=fs0 / SR, mf=mf)


def _coarse_curve(ref, cover, est, buffer, radius, match_n, log):
    """NCC vs candidate start time over est ± radius at coarse resolution."""
    crop = max(0.0, est - max(buffer, radius))
    s0 = round(crop * SR)
    log(f"3. Reference crop starts at {crop:.3f} s ({est - crop:.3f} s buffer before the estimate)")

    hop, win = COARSE
    lag_lo = max(0, int(np.floor((max(0.0, est - radius) * SR - s0) / hop)))
    lag_hi = int(np.ceil(((est + radius) * SR - s0) / hop))
    cover_c = envelopes(cover, 0, match_n, hop, win)
    ref_c = envelopes(ref, s0, lag_hi * hop + match_n + win, hop, win)
    m = min(cover_c.shape[1], ref_c.shape[1] - lag_hi)
    if m < 2 * SR / hop:
        raise ValueError("Not enough overlapping audio to compare (need at least 2 s).")
    times = np.array([(s0 + L * hop) / SR for L in range(lag_lo, lag_hi + 1)])
    curve = np.array([ncc_bands(cover_c, ref_c, 0, L, m) for L in range(lag_lo, lag_hi + 1)])
    return times, curve


def _check_candidates(ref, cover_f, times, curve, peaks):
    # Refine every near-tied peak and keep the one where the individual hits line up:
    # in a repetitive groove a one-beat shift can correlate almost as well.
    top = peaks[0]
    cands = []
    for i in peaks:
        if len(cands) >= MAX_CANDIDATES or curve[i] < curve[top] - max(0.05, 0.15 * abs(curve[top])):
            break
        if all(abs(c["coarse_t"] - times[i]) > SAME_PEAK for c in cands):
            cands.append(dict(verify(ref, cover_f, times[i]), coarse_t=times[i], coarse_ncc=curve[i]))
    return cands


def _margin(best, cands, times, curve, peaks):
    """Relative lead of the winner over the runner-up (candidate or coarse peak)."""
    rivals = [c for c in cands if c is not best]
    if rivals:
        return (best["score"] - max(c["score"] for c in rivals)) / max(1e-6, best["score"])
    top = peaks[0]
    others = [curve[i] for i in peaks if abs(times[i] - times[top]) > SAME_PEAK]
    return (curve[top] - others[0]) / max(1e-6, abs(curve[top])) if others else 1.0


def _confidence(best, margin):
    # Confidence: match quality scaled by uniqueness. A groove that fits equally well
    # one beat/bar away is ambiguous however good each match looks.
    parts = dict(
        correlation=float(np.clip((best["fine_ncc"] - 0.15) / 0.55, 0, 1)),
        hits=float(np.clip(best["hit_score"] / 0.8, 0, 1)),
        consistency=float(best["consistency"]),
        uniqueness=float(np.clip(margin / 0.25, 0, 1)),
    )
    quality = 0.4 * parts["correlation"] + 0.4 * parts["hits"] + 0.2 * parts["consistency"]
    return round(100 * quality * (0.4 + 0.6 * parts["uniqueness"])), parts


def _opening_hits(x, start, n):
    """Clear onsets per band in x[start:start + n], as song times."""
    hop, win = FINE
    out = []
    for e in long_envelopes(x, start, n, hop, win):
        times, strength = local_onsets(e, hop / SR, start / SR, min_gap=0.06)
        out.append(times[strength >= WEAK_HIT])
    return out


def _anchor(ref, cover, T):
    """Fine-tune T so the cover's opening hits land on the original's.

    The whole cover decides the bar; a player's timing wanders over a take, so its average can sit
    a little off the opening. Within ± ANCHOR_WINDOW, pick the shift that pairs the most of the
    first ANCHOR_SEC s of hits (from the cover's first hit), then centre it on their median error.
    Returns (T, first hit in the cover in s, paired, opening hits).
    """
    cov = _opening_hits(cover, 0, len(cover))
    first = min((c[0] for c in cov if len(c)), default=0.0)
    cov = [c[c <= first + ANCHOR_SEC] for c in cov]
    n = sum(len(c) for c in cov)
    if n < 4:
        return T, first, 0, n
    lo = round((T + first - ANCHOR_WINDOW - TOL) * SR)
    ref_on = _opening_hits(ref, max(0, lo), round((ANCHOR_SEC + 2 * (ANCHOR_WINDOW + TOL)) * SR))

    def pairs(shift):
        out = []
        for c, r in zip(cov, ref_on):
            if len(c) and len(r):
                d = r[None, :] - (c[:, None] + T + shift)
                j = np.abs(d).argmin(1)
                e = d[np.arange(len(c)), j]
                out += list(e[np.abs(e) <= TOL])
        return out

    shifts = np.arange(-ANCHOR_WINDOW, ANCHOR_WINDOW + 1e-9, 0.001)
    best = max(shifts, key=lambda s: (len(pairs(s)), -np.mean(np.abs(pairs(s))) if pairs(s) else 0, -abs(s)))
    errs = pairs(best)
    if not errs:
        return T, first, 0, n
    return T + best + float(np.median(errs)), first, len(errs), n


def align(ref, cover, bpm, block, buffer=1.0, radius=SEARCH_RADIUS, match_sec=30.0, log=print):
    est = block_start(bpm, block)
    if est >= len(ref) / SR:
        raise ValueError(f"Block {block} starts at {est:.3f} s but the reference is only {len(ref) / SR:.3f} s long.")
    log(f"1. Estimate: block {block} @ {bpm} BPM (4/4) -> {est:.3f} s")
    log(f"2. Loaded reference {len(ref) / SR:.3f} s, cover {len(cover) / SR:.3f} s")

    match_n = round(min(match_sec, len(cover) / SR) * SR)
    times, curve = _coarse_curve(ref, cover, est, buffer, radius, match_n, log)
    peaks = local_peaks(curve)
    top = peaks[0]
    log(f"4. Best correlation at {times[top]:.3f} s (NCC {curve[top]:.3f})")

    cover_f = envelopes(cover, 0, match_n, *FINE)
    cands = _check_candidates(ref, cover_f, times, curve, peaks)
    best = max(cands, key=lambda c: c["score"])
    hits = sum(b["matched"] for b in best["bands"])
    log(f"5. Refined to {best['T']:.4f} s; {hits} kick/snare/cymbal hits aligned; {len(cands)} candidate(s) checked")

    confidence, parts = _confidence(best, _margin(best, cands, times, curve, peaks))
    T, first, paired, opening = _anchor(ref, cover, best["T"])
    log(f"6. Fine-tuned on your opening hits ({ANCHOR_SEC:g} s from your first hit at {first:.3f} s): "
        f"{paired}/{opening} land on the original's, start moved {(T - best['T']) * 1000:+.1f} ms")
    log(f"7. Start {T:.3f} s | offset {(T - est) * 1000:+.0f} ms | confidence {confidence}/100")
    return dict(estimate=est, start=T, offset_ms=(T - est) * 1000, confidence=confidence, parts=parts,
                best=best, candidates=cands, curve=(times, curve), cover_f=cover_f, bpm=bpm,
                whole_cover_start=best["T"], first_hit=first, opening=(paired, opening))
