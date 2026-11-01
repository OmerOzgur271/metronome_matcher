"""Checks that the start is fine-tuned on the cover's opening hits, not the whole take's average."""
from aligner import block_start
from aligner.align import _anchor
from test_timing import BEAT, BPM, _cover


def test_opening_hits_set_the_start():
    start = block_start(BPM, 21)
    # Everything after the first 8 s of the cover is played 20 ms late: the whole-take average sits
    # ~15 ms late, but the opening is on time.
    late = {}
    for i in range(int((start + 8) / (BEAT / 2)), int((start + 41) / (BEAT / 2))):
        for drum in ("hat", "kick", "snare"):
            late[(round(i * BEAT / 2, 3), drum)] = 0.020
    ref, cover = _cover(start, 40, late)
    T, first, paired, opening = _anchor(ref, cover, start + 0.015)
    assert abs(T - start) < 0.003, (T - start, paired, opening)
    assert paired >= 0.8 * opening, (paired, opening)


if __name__ == "__main__":
    test_opening_hits_set_the_start()
    print("ok")
