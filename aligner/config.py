"""Shared constants for analysis and alignment."""

SR = 22050
BANDS = [("Kick", 30, 150), ("Snare", 150, 4000), ("Cymbal", 4000, 11025)]
COARSE = (128, 1024)   # hop, window  (~5.8 ms frames)
FINE = (32, 512)       # hop, window  (~1.45 ms frames)
TOL = 0.025            # s, a hit counts as aligned within this distance
SAME_PEAK = 0.04       # s, peaks closer than this are the same peak
SEARCH_RADIUS = 3.0    # s, look for the cover start within estimate ± this
MAX_CANDIDATES = 12    # near-tied coarse peaks to verify hit-by-hit
ANCHOR_SEC = 6.0       # s of the cover, from its first hit, that the start is fine-tuned on
ANCHOR_WINDOW = 0.06   # s, how far the opening hits may move the whole-cover start
MIX_PAD = 10.0         # s of song the final mix keeps before and after the cover

# Timing check (whole cover vs original, after the start is found)
MISTAKE_MS = 20        # a hit this far off its surroundings is a mistake; a section this far off is drifting
STABLE_BANDS = ("Kick", "Cymbal")  # follow tempo drift with these; the snare band also picks up fills/toms
DRIFT_WINDOW = 2.4     # s, local drift = median hit error within ± this
WEAK_HIT = 0.4         # onsets weaker than this × the band's typical hit are usually another drum leaking in# Source separation (demucs): the song is split into drums + everything else
DEMUCS_MODEL = "htdemucs_ft"
DEMUCS_SHIFTS = 7      # random time shifts averaged; more = cleaner and slower
DEMUCS_OVERLAP = 0.75
SNIPPET_MARGIN = 10.0  # s of song separated beyond the cover's search range on each side
