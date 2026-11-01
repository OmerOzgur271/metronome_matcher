"""Drum cover aligner: find where drumcover.mp3 starts inside extracted_drums.wav,
find the time ranges where the cover is most off the original and how to fix them,
crop both to the stretch they share as aligned mp3s, then optionally mix the cover
onto drumless.wav at that offset.

UI:   python app.py
CLI:  python app.py drumcover.mp3 extracted_drums.wav --bpm 100 --block 21
      python app.py drumcover.mp3 extracted_drums.wav --drumless drumless.wav -o final.wav
      python app.py drumcover.mp3 extracted_drums.wav --crop extracted_aligned.mp3 cover_aligned.mp3
      python app.py drumcover.mp3 extracted_drums.wav --chart errors.png  # error rate over the song
      python app.py drumcover.mp3 extracted_drums.wav --clips problems/  # problem parts with a click
"""
from aligner.cli import main

if __name__ == "__main__":
    main(description=__doc__)
