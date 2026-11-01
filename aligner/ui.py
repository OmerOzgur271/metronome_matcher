"""Gradio front end."""
import tempfile

from .align import align
from .clips import problem_clips
from .audio_io import load
from .config import MIX_PAD, SEARCH_RADIUS
from .export import export_aligned
from .helpers import block_start
from .mix import mix
from .problems import TOP, mmss, problem_ranges, problems_plot
from .report import overlay, plot, report, timing_report
from .timing import check_timing


def _temp_file(prefix, suffix):
    out = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False)
    out.close()
    return out.name


def _estimate_text(bpm, block):
    try:
        return f"Estimated start: **{block_start(bpm, int(block)):.3f} s** (bar = {240 / bpm:.3f} s)"
    except (TypeError, ZeroDivisionError):
        return ""


def ui():
    import gradio as gr

    def run(cover_file, ref_file, drumless_file, bpm, block, buffer, radius, match_sec, do_mix, cover_gain, pad,
            with_original):
        if not cover_file or not ref_file:
            raise gr.Error("Load both drumcover.mp3 and extracted_drums.wav.")
        if do_mix and not drumless_file:
            raise gr.Error("Load drumless.wav, or turn off Merge onto drumless.")
        steps = []
        ref, cover = load(ref_file), load(cover_file)
        r = align(ref, cover, bpm, int(block), buffer, radius, match_sec, log=steps.append)
        t = check_timing(ref, cover, r["start"], bpm)
        ranges = problem_ranges(t)
        steps.append(f"7. Timing check: {sum(h['err'] is not None for h in t['hits'])} hits compared, "
                     f"{len(ranges)} problem ranges, {len(t['mistakes'])} single wrong hits")
        clips = problem_clips(cover_file, ref_file, t, ranges, tempfile.mkdtemp(prefix="problem_clips_"),
                              original_level=0.5 if with_original else 0.0)
        players = [gr.Audio(value=path, visible=True,
                            label=f"#{i} · song {mmss(q['song_from'])}–{mmss(q['song_to'])} (bars {q['first']}–"
                                  f"{q['last']}) · {', '.join(q['tags'])}")
                   for i, (q, path) in enumerate(zip(ranges, clips), 1)]
        players += [gr.Audio(value=None, visible=False)] * (TOP - len(players))
        try:
            ref_mp3, cover_mp3, begin, length = export_aligned(
                ref_file, cover_file, r["start"], _temp_file("extracted_aligned_", ".mp3"),
                _temp_file("cover_aligned_", ".mp3"))
        except RuntimeError as exc:
            raise gr.Error(str(exc)) from exc
        steps.append(f"8. Cropped both to song {begin:.3f}–{begin + length:.3f} s ({length:.1f} s) as aligned mp3s")
        final = None
        if do_mix:
            final = _temp_file("final_mix_", ".wav")
            try:
                mix(drumless_file, cover_file, r["start"], final, cover_gain, pad)
            except RuntimeError as exc:
                raise gr.Error(str(exc)) from exc
            steps.append(f"9. Mixed drumcover onto drumless at {r['start']:.3f} s (level {cover_gain:.2f}), "
                         f"keeping {pad:g} s before and after it")
        return (report(r), "\n".join(steps), plot(r), overlay(ref, cover, r["start"]), final,
                ref_mp3, cover_mp3, timing_report(t), problems_plot(t, ranges), *players)

    with gr.Blocks(title="Drum Cover Aligner") as app:
        gr.Markdown(
            "# Drum Cover Aligner\n"
            "Finds where **drumcover.mp3** starts in **extracted_drums.wav** (4/4), "
            "then can mix it onto **drumless.wav** at that time."
        )
        with gr.Row():
            bpm = gr.Number(100, label="BPM")
            block = gr.Number(21, label="Block (bar) number", precision=0)
            est = gr.Markdown(f"Estimated start: **{block_start(100, 21):.3f} s**")
        with gr.Row():
            cover_file = gr.File(label="drumcover.mp3", type="filepath")
            ref_file = gr.File(label="extracted_drums.wav", type="filepath")
            drumless_file = gr.File(label="drumless.wav", type="filepath")
        with gr.Row():
            do_mix = gr.Checkbox(True, label="Merge drumcover onto drumless")
            cover_gain = gr.Slider(0, 2, value=1, step=0.05, label="Drum cover level")
            pad = gr.Number(MIX_PAD, label="Seconds of song to keep before and after the cover")
        with gr.Accordion("Advanced", open=False):
            buffer = gr.Number(1.0, label="Buffer before estimate (s)")
            radius = gr.Number(SEARCH_RADIUS, label="Search radius around estimate (s)")
            match_sec = gr.Number(30.0, label="Seconds of cover to compare")
        go = gr.Button("Find start time", variant="primary")
        result = gr.Markdown()
        steps = gr.Textbox(label="Steps", lines=8)
        fig = gr.Plot(label="Alignment")
        audio = gr.Audio(label="Listen: reference (left) + cover (right) at the found offset")
        final = gr.Audio(label="Final mix: drumless + drumcover, cover part ± the seconds kept", type="filepath")
        with gr.Row():
            ref_mp3 = gr.Audio(label="Extracted drums, cropped to the cover (mp3)", type="filepath")
            cover_mp3 = gr.Audio(label="Drum cover, aligned and cropped (mp3)", type="filepath")
        timing = gr.Markdown()
        problems_fig = gr.Plot(label="Error rate over the song: the numbered boxes are the worst ranges in the report")
        gr.Markdown("### Listen to the problem parts\nYour take with a click at the song's BPM, from a bar before "
                    "each range to a bar after it. The click is on the original's beats: if you hear yourself "
                    "behind it you're late, ahead of it you're early.")
        with_original = gr.Checkbox(False, label="Also play the original drums quietly in the clips")
        players = [gr.Audio(type="filepath", visible=False) for _ in range(TOP)]

        bpm.change(_estimate_text, [bpm, block], est)
        block.change(_estimate_text, [bpm, block], est)
        go.click(
            run,
            [cover_file, ref_file, drumless_file, bpm, block, buffer, radius, match_sec, do_mix, cover_gain, pad,
             with_original],
            [result, steps, fig, audio, final, ref_mp3, cover_mp3, timing, problems_fig, *players],
        )
    app.launch(inbrowser=True)
