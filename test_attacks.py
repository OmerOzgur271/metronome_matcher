"""Checks that the attack analyser measures attack starts to the ms and labels who played what."""
import numpy as np

from aligner import block_start, check_timing
from aligner.attacks import analyse_attacks, attack_report, attack_table
from test_timing import BEAT, BPM, _cover


def test_attack_starts_are_measured_to_the_ms():
    start = block_start(BPM, 21)
    late_kick, early_snare = round(start + 10 * BEAT, 3), round(start + 21 * BEAT, 3)
    ref, cover = _cover(start, 40, {(late_kick, "kick"): 0.030, (early_snare, "snare"): -0.025})
    t = check_timing(ref, cover, start, BPM)
    rows = analyse_attacks(ref, cover, t)
    both = [r for r in rows if r["status"] == "matched"]

    def at(when, band):
        return min((r for r in both if r["band"] == band), key=lambda r: abs(r["orig"]["start"] - when))

    assert abs(at(late_kick, "Kick")["start_ms"] - 30) < 4, at(late_kick, "Kick")["start_ms"]
    assert abs(at(early_snare, "Snare")["start_ms"] + 25) < 4, at(early_snare, "Snare")["start_ms"]
    rest = np.array([r["start_ms"] for r in both
                     if min(abs(r["orig"]["start"] - late_kick), abs(r["orig"]["start"] - early_snare)) > 0.1])
    assert np.median(np.abs(rest)) < 4, np.median(np.abs(rest))  # the cover's own jitter is 3 ms
    assert all(r["you"]["start"] <= r["you"]["peak"] <= r["you"]["end"] for r in both)
    assert all(r["orig"]["length_ms"] > 0 for r in both)


def test_labels_and_table():
    start = block_start(BPM, 21)
    ref, cover = _cover(start, 40, {})
    t = check_timing(ref, cover, start, BPM)
    rows = analyse_attacks(ref, cover, t)
    assert {r["status"] for r in rows} <= {"matched", "extra", "missed"}
    assert all(r["position"].count(".") == 2 for r in rows)
    head, data = attack_table(rows)
    assert len(data) == len(rows) and all(len(d) == len(head) for d in data)
    assert "Drift through your take" in attack_report(rows, t)


if __name__ == "__main__":
    test_attack_starts_are_measured_to_the_ms()
    test_labels_and_table()
    print("ok")
