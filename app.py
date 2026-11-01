"""Drum cover aligner: give it your drum cover and the original song.

Splits the song into drums and drumless with demucs (htdemucs_ft, cached in separated/),
finds where the cover starts in the separated drums, finds the time ranges where the cover
is most off the original and how to fix them, then mixes the cover onto both the original
song and the drumless track (written to output/).

UI:   python app.py
CLI:  python app.py drumcover.mp3 gg.mp3 --bpm 100 --block 21
      python app.py drumcover.mp3 gg.mp3 --out-dir mixes/ --gain 0.9 --pad 10
      python app.py drumcover.mp3 gg.mp3 --crop drums_aligned.mp3 cover_aligned.mp3
      python app.py drumcover.mp3 gg.mp3 --chart errors.png  # error rate over the song
      python app.py drumcover.mp3 gg.mp3 --clips problems/  # problem parts with a click
"""
from aligner.cli import main

if __name__ == "__main__":
    main(description=__doc__)
