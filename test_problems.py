"""Checks that problem_ranges() finds and diagnoses bad sections of a cover."""
from aligner import block_start, check_timing, problem_ranges, problems_report
from test_timing import BEAT, BPM, _cover


def _section(start, first_bar, bars, drum, shift):
    """Shifts for every `drum` hit in bars first_bar .. first_bar + bars - 1 (bar 1 starts at 0 s)."""
    shifts = {}
    for i in range(int((first_bar - 1) * 4 / 0.5), int((first_bar - 1 + bars) * 4 / 0.5)):
        for d in ([drum] if drum else ["hat", "kick", "snare"]):
            shifts[(round(i * BEAT / 2, 3), d)] = shift
    return shifts


def test_finds_dragging_and_late_kick():
    # The cover covers bars 21-37.
    start = block_start(BPM, 21)
    shifts = {**_section(start, 24, 4, None, 0.030), **_section(start, 32, 4, "kick", 0.032)}
    ref, cover = _cover(start, 40, shifts)
    t = check_timing(ref, cover, start, BPM)
    ranges = problem_ranges(t)
    drag = [r for r in ranges if r["first"] <= 25 and r["last"] >= 26]
    kick = [r for r in ranges if r["first"] <= 33 and r["last"] >= 34]
    assert drag and "dragging" in drag[0]["tags"], ranges
    assert kick and any(tag.startswith("mostly kick") for tag in kick[0]["tags"]), ranges
    assert len(ranges) == 2, [(r["first"], r["last"], r["tags"]) for r in ranges]
    assert "#### #1" in problems_report(t, ranges)


def test_clean_cover_has_no_ranges():
    start = block_start(BPM, 21)
    ref, cover = _cover(start, 40, {})
    assert problem_ranges(check_timing(ref, cover, start, BPM)) == []


if __name__ == "__main__":
    test_finds_dragging_and_late_kick()
    test_clean_cover_has_no_ranges()
    print("ok")
